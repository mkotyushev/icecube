
import gc
import math
from pathlib import Path
import numpy as np
import os
import torch
import pandas as pd
import gc, os
import numpy as np
from mock import patch
from typing import Any, Dict, List, Union, Optional
from pytorch_lightning.callbacks import (
    EarlyStopping, 
    GradientAccumulationScheduler,
    ModelCheckpoint,
    LearningRateMonitor
)
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import Module, Linear, ModuleList, BatchNorm1d
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.pool import knn_graph
from torch_geometric.typing import Adj, PairTensor
from torch_geometric.nn import EdgeConv
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.components.layers import DynEdgeConv
from graphnet.models.detector.icecube import IceCubeKaggle
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import AngleReconstructionCos, AngleReconstructionSinCos, AngleReconstructionSincosWithKappa, AzimuthReconstruction, AzimuthReconstructionWithKappa, DirectionReconstruction, DirectionReconstructionWithKappa, ZenithAzimuthReconstruction, ZenithReconstructionWithKappa
from graphnet.training.callbacks import ProgressBar
from graphnet.training.loss_functions import CosineLoss, CosineLoss3D, EuclidianDistanceLossCos, EuclidianDistanceLossSinCos, VonMisesFisher2DLoss, VonMisesFisher2DLossSinCos, VonMisesFisher3DLoss
from graphnet.training.labels import Direction
from graphnet.training.utils import make_dataloader
from graphnet.utilities.logging import get_logger
from torch.utils.data import DataLoader
from pytorch_lightning import Callback
from pytorch_lightning.loggers.logger import Logger
from simplex.models.simplex_models import SimplexNet, SimplexModule
from graphnet.models.model import Model
from pytorch_lightning.utilities import grad_norm
from graphnet.data.sqlite import SQLiteDataset
from graphnet.data.parquet import ParquetDataset, ParallelParquetTrainDataset


# Constants
logger = get_logger()


class ExpLRSchedulerPiece:
    def __init__(self, start_lr, stop_lr, decay=0.2):
        self.start_lr = start_lr
        self.scale = (start_lr - stop_lr) / (start_lr - start_lr * np.exp(-1.0 / decay))
        self.decay = decay

    def __call__(self, pct):
        # Parametrized so that in 0.0 it is start_lr and 
        # in 1.0 it is stop_lr
        # shift -> scale -> shift
        return \
            (
                self.start_lr * np.exp(-pct / self.decay) - 
                self.start_lr
            ) * self.scale + \
        self.start_lr
    

class ConstLRSchedulerPiece:
    def __init__(self, start_lr):
        self.start_lr = start_lr

    def __call__(self, pct):
        return self.start_lr
    

class LinearLRSchedulerPiece:
    def __init__(self, start_lr, stop_lr):
        self.start_lr = start_lr
        self.stop_lr = stop_lr

    def __call__(self, pct):
        return self.start_lr + pct * (self.stop_lr - self.start_lr)
    

class CosineLRSchedulerPiece:
    def __init__(self, start_lr, stop_lr):
        self.start_lr = start_lr
        self.stop_lr = stop_lr

    def __call__(self, pct):
        return self.stop_lr + (self.start_lr - self.stop_lr) * (1 + np.cos(np.pi * pct)) / 2


class PiecewiceFactorsLRScheduler(LRScheduler):
    """
    Piecewise learning rate scheduler.

    Each piece operates between two milestones. The first milestone is always 0.
    Given percent of the way through the current piece, piece yields the learning rate.
    Last piece is continued indefinitely for epoch > last milestone.
    """
    def __init__(self, optimizer, milestones, pieces, last_epoch=-1):
        assert len(milestones) - 1 == len(pieces)
        assert milestones[0] == 0
        assert all(milestones[i] < milestones[i + 1] for i in range(len(milestones) - 1))

        self.milestones = milestones
        self.pieces = pieces
        self._current_piece_index = 0

        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if (
            not self._current_piece_index == len(self.pieces) - 1 and 
            self.last_epoch > self.milestones[self._current_piece_index + 1]
        ):
            self._current_piece_index += 1

        pct = (
            self.last_epoch - 
            self.milestones[self._current_piece_index]
        ) / (
            self.milestones[self._current_piece_index + 1] - 
            self.milestones[self._current_piece_index]
        )

        return [
            self.pieces[self._current_piece_index](pct) * base_lr
            for base_lr in self.base_lrs
        ]


def make_scheduler_kwargs(config, train_dataloader):
    assert "scheduler_kwargs" in config and "pieces" in config["scheduler_kwargs"]

    # Pytorch schedulers are parametrized on number of steps
    # and not in percents, so scheduler_kwargs need to be
    # constructed here
    assert len(config["scheduler_kwargs"]["pieces"]) == 2, \
        "Only 2 pieces one-cycle LR is supported for now"

    if config['dataset_type'] == 'parallel_parquet':
        # There is only one epoch for that dataset type,
        # but consisting data of all epochs
        assert config['fit']['max_epochs'] == 1
        single_epoch_steps = math.ceil(
            len(train_dataloader) /
            config['parallel_parquet']['actual_max_epochs']
        )
        max_epochs = config['parallel_parquet']['actual_max_epochs']
        warmup_epochs = config['parallel_parquet']['warmup_epochs']
    else:
        # Usial case
        single_epoch_steps = len(train_dataloader)
        max_epochs = config['fit']['max_epochs']
        warmup_epochs = 0.5
    
    scheduler_kwargs = {
        # 0.5 epoch warmup piece + rest piece
        "milestones": [
            0,
            math.ceil(single_epoch_steps * warmup_epochs),
            math.ceil(single_epoch_steps * max_epochs),
        ],
        "pieces": config["scheduler_kwargs"]["pieces"],
    }

    return scheduler_kwargs


def build_model(
    config: Dict[str, Any], 
    train_dataloader: Any, 
    fix_points = None
) -> StandardModel:
    """Builds GNN from config"""
    # Building model
    detector = IceCubeKaggle(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        global_pooling_schemes=["min", "max", "mean"],
        fix_points=fix_points,
        **config['dynedge']
    )

    tasks = []
    if config["target"] == 'direction':
        task = DirectionReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=config["target"],
            loss_function=VonMisesFisher3DLoss(),
            loss_weight='loss_weight' if config['loss_weight'] else None,
            bias=config['dynedge']['bias'],
            fix_points=fix_points,
        )
        tasks.append(task)
        prediction_columns = [config["target"] + "_x", 
                              config["target"] + "_y", 
                              config["target"] + "_z", 
                              config["target"] + "_kappa" ]
        additional_attributes = ['zenith', 'azimuth', 'event_id']
    elif config["target"] == 'angles_legacy':
        task = ZenithReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=config['truth'][0],
            loss_function=VonMisesFisher2DLoss(),
            loss_weight='loss_weight' if config['loss_weight'] else None,
            bias=config['dynedge']['bias'],
            fix_points=fix_points,
        )
        tasks.append(task)
        task = AzimuthReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=config['truth'][1],
            loss_function=VonMisesFisher2DLoss(),
            loss_weight='loss_weight' if config['loss_weight'] else None,
            bias=config['dynedge']['bias'],
            fix_points=fix_points,
        )
        tasks.append(task)
        prediction_columns = [config['truth'][0] + '_pred', 
                              config['truth'][0] + '_kappa',
                              config['truth'][1] + '_pred', 
                              config['truth'][1] + '_kappa']
        additional_attributes = [*config['truth'], 'event_id']
    elif config["target"] == 'angles':
        task = ZenithAzimuthReconstruction(
            hidden_size=gnn.nb_outputs,
            target_labels=config['truth'],
            loss_function=CosineLoss(),
            loss_weight='loss_weight' if config['loss_weight'] else None,
            bias=config['dynedge']['bias'],
            fix_points=fix_points,
        )
        tasks.append(task)
        prediction_columns = [config['truth'][0] + '_pred', 
                              config['truth'][1] + '_pred']
        additional_attributes = [*config['truth'], 'event_id']
    # elif config["target"] == 'angles_from_3d':
    #     task = ZenithAzimuthReconstructionFrom3D(
    #         hidden_size=gnn.nb_outputs,
    #         target_labels=truth,
    #         loss_function=CosineLoss(),
    #         loss_weight='loss_weight' if config['loss_weight'] else None,
    #         bias=config['dynedge']['bias'],
    #         fix_points=fix_points,
    #     )
    #     tasks.append(task)
    #     prediction_columns = [truth[0] + '_pred', 
    #                           truth[1] + '_pred']
    #     additional_attributes = [*truth, 'event_id']
    elif config["target"] == 'angles_sincos':
        task = AngleReconstructionSincosWithKappa(
            half=True,
            hidden_size=gnn.nb_outputs,
            target_labels=config['truth'][0],
            loss_function=VonMisesFisher2DLossSinCos(),
            loss_weight='loss_weight' if config['loss_weight'] else None,
            bias=config['dynedge']['bias'],
            fix_points=fix_points,
        )
        tasks.append(task)
        task = AngleReconstructionSincosWithKappa(
            half=False,
            hidden_size=gnn.nb_outputs,
            target_labels=config['truth'][1],
            loss_function=VonMisesFisher2DLossSinCos(),
            loss_weight='loss_weight' if config['loss_weight'] else None,
            bias=config['dynedge']['bias'],
            fix_points=fix_points,
        )
        tasks.append(task)
        prediction_columns = [
            config['truth'][0] + '_sin', 
            config['truth'][0] + '_cos',
            config['truth'][0] + '_kappa',
            config['truth'][1] + '_sin', 
            config['truth'][1] + '_cos',
            config['truth'][1] + '_kappa',
        ]
        additional_attributes = [*config['truth'], 'event_id']
    elif config["target"] == 'angles_sincos_euclidean':
        task = AngleReconstructionSinCos(
            half=True,
            hidden_size=gnn.nb_outputs,
            target_labels=config['truth'][0],
            loss_function=EuclidianDistanceLossSinCos(),
            loss_weight='loss_weight' if config['loss_weight'] else None,
            bias=config['dynedge']['bias'],
            fix_points=fix_points,
        )
        tasks.append(task)
        task = AngleReconstructionSinCos(
            half=False,
            hidden_size=gnn.nb_outputs,
            target_labels=config['truth'][1],
            loss_function=EuclidianDistanceLossSinCos(),
            loss_weight='loss_weight' if config['loss_weight'] else None,
            bias=config['dynedge']['bias'],
            fix_points=fix_points,
        )
        tasks.append(task)
        prediction_columns = [
            config['truth'][0] + '_sin', 
            config['truth'][0] + '_cos',
            config['truth'][1] + '_sin', 
            config['truth'][1] + '_cos',
        ]
        additional_attributes = [*config['truth'], 'event_id']
    elif config["target"] in ['zenith_sincos_euclidean', 'zenith_sincos_euclidean_cancel_azimuth']:
        task = AngleReconstructionSinCos(
            half=True,
            hidden_size=gnn.nb_outputs,
            target_labels=config['truth'][0],
            loss_function=EuclidianDistanceLossSinCos(),
            loss_weight='loss_weight' if config['loss_weight'] else None,
            bias=config['dynedge']['bias'],
            fix_points=fix_points,
        )
        tasks.append(task)
        prediction_columns = [
            config['truth'][0] + '_sin', 
            config['truth'][0] + '_cos',
        ]
        additional_attributes = [*config['truth'], 'event_id']
        if config["target"] == 'zenith_sincos_euclidean_cancel_azimuth':
            additional_attributes.append('azimuth_pred')
    elif config["target"] == 'zenith_cos_euclidean':
        task = AngleReconstructionCos(
            hidden_size=gnn.nb_outputs,
            target_labels=config['truth'][0],
            loss_function=EuclidianDistanceLossCos(),
            loss_weight='loss_weight' if config['loss_weight'] else None,
            bias=config['dynedge']['bias'],
            fix_points=fix_points,
        )
        tasks.append(task)
        prediction_columns = [
            config['truth'][0] + '_cos',
        ]
        additional_attributes = [*config['truth'], 'event_id']
    elif config["target"] == 'zenith':
        task = ZenithReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=config['truth'][0],
            loss_function=VonMisesFisher2DLoss(),
            loss_weight='loss_weight' if config['loss_weight'] else None,
            bias=config['dynedge']['bias'],
            fix_points=fix_points,
        )
        tasks.append(task)
        prediction_columns = [config['truth'][0] + '_pred', 
                              config['truth'][0] + '_kappa']
        additional_attributes = [*config['truth'], 'event_id']
    elif config["target"] == 'azimuth':
        task = ZenithReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=config['truth'][1],
            loss_function=VonMisesFisher2DLoss(),
            loss_weight='loss_weight' if config['loss_weight'] else None,
            bias=config['dynedge']['bias'],
            fix_points=fix_points,
        )
        tasks.append(task)
        prediction_columns = [config['truth'][1] + '_pred', 
                              config['truth'][1] + '_kappa']
        additional_attributes = [*config['truth'], 'event_id']
    elif config["target"] == 'direction_cosine':
        task = DirectionReconstruction(
            hidden_size=gnn.nb_outputs,
            target_labels=config['truth'],
            loss_function=CosineLoss3D(),
            loss_weight='loss_weight' if config['loss_weight'] else None,
            bias=config['dynedge']['bias'],
            fix_points=fix_points,
        )
        tasks.append(task)
        prediction_columns = [
            'direction_x', 
            'direction_y',
            'direction_z',
        ]
        additional_attributes = [*config['truth'], 'event_id']


    scheduler_kwargs = make_scheduler_kwargs(config, train_dataloader)
    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=tasks,
        tasks_weiths=config["tasks_weights"] if "tasks_weights" in config else None,
        optimizer_class=config['optimizer_class'],
        optimizer_kwargs=config["optimizer_kwargs"],
        scheduler_class=PiecewiceFactorsLRScheduler,
        scheduler_kwargs=scheduler_kwargs,
        scheduler_config={
            "interval": "step",
        },
        **config["model_kwargs"],
    )
    model.prediction_columns = prediction_columns
    model.additional_attributes = additional_attributes
    
    return model

def load_pretrained_model(
    config: Dict[str,Any], 
    path: str = '/kaggle/input/dynedge-pretrained/dynedge_pretrained_batch_1_to_50/state_dict.pth',
    return_train_dataloader: bool = False
) -> StandardModel:
    train_dataloader, _ = make_dataloaders(config = config)
    
    if path.endswith('model.pth'):
        logger.info(f'Loading model from {path}')
        model = StandardModel.load(path)
    else:
        if config['train_mode'] == 'simplex_inference':
            model, _, _ = build_model_simplex(
                config=config, 
                state_dict_path=None,
                is_inference=True
            )
        else:
            model = build_model(config = config, 
                                train_dataloader = train_dataloader)
        #model._inference_trainer = Trainer(config['fit'])
        logger.info(f'Current model state dict keys: {model.state_dict().keys()}')

        if path.endswith('.ckpt'):
            logger.info(f'Loading checkpoint from {path}')
            model = StandardModel.load_from_checkpoint(
                path, 
                detector=model._detector, 
                gnn=model._gnn, 
                tasks=[task for task in model._tasks]
            )
        elif path.endswith('.pth'):
            logger.info(f'Loading state dict from {path}')
            state_dict = torch.load(path)
            model.load_state_dict(state_dict)
        else:
            raise ValueError(f'path must be a .pth or .ckpt file, got {path}')

    if return_train_dataloader:
        return model, train_dataloader
    else:
        return model

def make_dataloaders(config: Dict[str, Any], labels=None) -> List[Any]:
    """Constructs training and validation dataloaders for training with early stopping."""
    loss_weight_kwargs, max_n_pulses_kwargs = {}, {}
    if 'max_n_pulses' in config:
        max_n_pulses_kwargs = config['max_n_pulses']
    if 'loss_weight' in config:
        loss_weight_kwargs = config['loss_weight']

    dataset_kwargs = dict()
    if config['dataset_type'] == 'sqlite':
        dataset_class = SQLiteDataset
    elif config['dataset_type'] == 'parquet':
        dataset_class = ParquetDataset
    elif config['dataset_type'] == 'parallel_parquet':
        dataset_class = ParallelParquetTrainDataset
        dataset_kwargs = dict(
            geometry_path=config['parallel_parquet']['geometry_path'],
            meta_path=config['parallel_parquet']['meta_path'],
            filepathes=config['parallel_parquet']['filepathes'],
        )
    else:
        raise ValueError(f'Unknown dataset type {config["dataset_type"]}')

    train_dataloader = make_dataloader(
        dataset_class = dataset_class,
        db = config['path'],
        selection = None,
        pulsemaps = config['pulsemap'],
        features = config['features'],
        truth = config['truth'],
        batch_size = config['batch_size'],
        num_workers = config['num_workers'],
        shuffle = False,
        labels = labels,
        index_column = config['index_column'],
        truth_table = config['truth_table'],
        transforms = config['train_transforms'],
        **loss_weight_kwargs,
        **max_n_pulses_kwargs,
        **dataset_kwargs
    )
    
    validate_dataloader = make_dataloader(
        dataset_class = SQLiteDataset,
        db = config['inference_database_path'],
        selection = None,
        pulsemaps = config['pulsemap'],
        features = config['features'],
        truth = config['truth'],
        batch_size = config['batch_size'],
        num_workers = config['num_workers'],
        shuffle = False,
        labels = labels,
        index_column = config['index_column'],
        truth_table = config['truth_table'],
        max_n_pulses=config['max_n_pulses']['max_n_pulses'],
        max_n_pulses_strategy="clamp",
        transforms = config['val_transforms'],
    )
    return train_dataloader, validate_dataloader


def train_dynedge(
    model, 
    config, 
    train_dataloader, 
    validate_dataloader, 
    callbacks: Optional[List[Callback]] = None
):
    # Compile model
    torch.compile(model)

    # Training model
    default_callbacks = [
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
        ),
        ProgressBar(),
    ]
    if callbacks is None:
        callbacks = default_callbacks
    else:
        callbacks += default_callbacks

    model_checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        monitor="val_loss",
        save_top_k=3,
        save_on_train_epoch_end=False,
        save_last=True,
    )
    callbacks.append(model_checkpoint_callback)

    model.fit(
        train_dataloader,
        validate_dataloader,
        callbacks=callbacks,
        **config["fit"],
    )
    return model


def train_dynedge_from_scratch(config: Dict[str, Any], state_dict_path=None) -> StandardModel:
    """Builds and trains GNN according to config."""
    logger.info(f"features: {config['features']}")
    logger.info(f"truth: {config['truth']}")
    
    archive = os.path.join(config['base_dir'], "train_model_without_configs")
    run_name = f"dynedge_{config['target']}_{config['run_name_tag']}"

    labels = {'direction': Direction(azimuth_key=config['truth'][1], zenith_key=config['truth'][0])}
    train_dataloader, validate_dataloader = make_dataloaders(config = config, labels = labels)

    if state_dict_path is None:
        model = build_model(config, train_dataloader)
    else:
        model = load_pretrained_model(config, state_dict_path)

    return train_dynedge(model, config, train_dataloader, validate_dataloader)


def inference(model, config: Dict[str, Any], use_labels: bool) -> pd.DataFrame:
    labels = None
    if use_labels:
        labels = {'direction': Direction(azimuth_key=config['truth'][1], zenith_key=config['truth'][0])}
    """Applies model to the database specified in config['inference_database_path'] and saves results to disk."""
    _, validate_dataloader = make_dataloaders(config=config, labels=labels)
    
    # Get predictions
    with torch.no_grad():
        results = model.predict_as_dataframe(
            gpus = [0],
            dataloader = validate_dataloader,
            prediction_columns=model.prediction_columns,
            additional_attributes=model.additional_attributes,
            distribution_strategy='auto'
        )
    # Save predictions and model to file
    archive = os.path.join(config['base_dir'], "train_model_without_configs")
    run_name = f"dynedge_{config['target']}_{config['run_name_tag']}"
    db_name = config['path'].split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)
    logger.info(f"Writing results to {path}")
    os.makedirs(path, exist_ok=True)

    results.to_csv(f"{path}/results.csv")
    return results


def convert_to_3d(df: pd.DataFrame) -> pd.DataFrame:
    """Converts zenith and azimuth to 3D direction vectors"""
    df['true_x'] = np.cos(df['azimuth']) * np.sin(df['zenith'])
    df['true_y'] = np.sin(df['azimuth']) * np.sin(df['zenith'])
    df['true_z'] = np.cos(df['zenith'])
    return df


def calculate_angular_error(df : pd.DataFrame) -> pd.DataFrame:
    """Calcualtes the opening angle (angular error) between true and reconstructed direction vectors"""
    if 'direction_x' not in df.columns:
        if 'azimuth_pred' not in df.columns:
            # azimuth_pred = np.arctan2(df['azimuth_sin'], df['azimuth_cos'])
            # zenith_pred = np.arctan2(df['zenith_sin'], df['zenith_cos'])
            # df['direction_x'] = np.cos(azimuth_pred) * np.sin(zenith_pred)
            # df['direction_y'] = np.sin(azimuth_pred) * np.sin(zenith_pred)
            # df['direction_z'] = np.cos(zenith_pred)
            
            df['direction_x'] = df['azimuth_cos'] * df['zenith_sin']
            df['direction_y'] = df['azimuth_sin'] * df['zenith_sin']
            df['direction_z'] = df['zenith_cos']

            norm = np.sqrt(df['direction_x']**2 + df['direction_y']**2 + df['direction_z']**2)
            df['direction_x'] /= norm
            df['direction_y'] /= norm
            df['direction_z'] /= norm
        else:
            df['direction_x'] = np.cos(df['azimuth_pred']) * np.sin(df['zenith_pred'])
            df['direction_y'] = np.sin(df['azimuth_pred']) * np.sin(df['zenith_pred'])
            df['direction_z'] = np.cos(df['zenith_pred'])
    df['angular_error'] = np.arccos(df['true_x']*df['direction_x'] + df['true_y']*df['direction_y'] + df['true_z']*df['direction_z'])
    return df


# Common parts of BlockLinear and BlockBatchNorm1d
class BlockModule(Module):
    """Base class for block modules"""
    def freeze_first_n_blocks(self, n: int):
        """Freezes first n blocks"""
        for i, block in enumerate(self.blocks):
            if i < n:
                for param in block.parameters():
                    param.requires_grad = False
        return self

    def freeze_except_last_block(self):
        """Freezes all blocks except last"""
        self.freeze_first_n_blocks(len(self.blocks) - 1)
        return self

    def up_to_last_block_parameters(self):
        for i, block in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                for param in block.parameters():
                    yield param

    def last_block_parameters(self):
        for param in self.blocks[-1].parameters():
            yield param


class BlockLinear(BlockModule):
    output_aggregation = 'sum'

    """Block linear layer"""
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        device=None, 
        dtype=None, 
        block_sizes=None, 
        type='intermediate', 
        input_transform=None,
    ) -> None:
        super().__init__()

        if block_sizes is None:
            block_sizes = [(in_features, out_features)]

        assert sum(block_size[0] for block_size in block_sizes) == in_features, \
            "Sum of input features of all blocks must be equal to in_features"
        assert sum(block_size[1] for block_size in block_sizes) == out_features, \
            "Sum of output features of all blocks must be equal to out_features"
        if type == 'input':
            assert all(block_size[0] == in_features for block_size in block_sizes), \
                "Input features of all blocks must be equal to in_features for type == input"
        if type == 'output':
            assert all(block_size[1] == out_features for block_size in block_sizes), \
                "Output features of all blocks must be equal to out_features for type == output"

        self.in_features = in_features 
        self.out_features = out_features 
        self.type = type
        self.input_transform = input_transform
        
        self.blocks = ModuleList()
        for block_size in block_sizes:
            self.blocks.append(
                Linear(
                    block_size[0], 
                    block_size[1], 
                    bias=bias,
                    device=device,
                    dtype=dtype
                )
            )

    def set_type(self, type):
        if type == 'input':
            assert all(
                linear.in_features == self.in_features 
                for linear in self.blocks
            )
        elif type == 'output':
            assert all(
                linear.out_features == self.out_features 
                for linear in self.blocks
            )
        self.type = type
        return self

    def set_input_transform(self, input_transform):
        self.input_transform = input_transform
        return self

    @property
    def bias(self):
        if self.blocks[0].bias is None:
            return None
        return torch.cat([linear.bias for linear in self.blocks], dim=0)
    
    @property
    def weight(self):
        return torch.block_diag(*[linear.weight for linear in self.blocks])
    
    def add_block(self, in_features_block: int, out_features_block: int, init: str = 'default'):
        """Adds a new block"""
        assert init in ['default', 'zero'], \
            "init must be one of ['default', 'zero']"

        linear = Linear(
            in_features_block, 
            out_features_block, 
            bias=self.bias is not None,
            device=self.blocks[0].weight.device,
            dtype=self.blocks[0].weight.dtype
        )
        self.blocks.append(linear)

        # # Re-normalize weights of existing blocks
        # scale = (len(self.blocks) - 1) / len(self.blocks)
        # with torch.no_grad():
        #     for linear in self.blocks[:-1]:
        #         linear.weight.data *= scale
        #         if linear.bias is not None:
        #             linear.bias.data *= scale

        if self.type != 'input':
            self.in_features += in_features_block
        if self.type != 'output':
            self.out_features += out_features_block

        # reinitialize added layer weights to match current in_features
        if init == 'zero':
            linear.weight.data.zero_()
            if linear.bias is not None:
                linear.bias.data.zero_()
        elif init == 'default':
            # Default initialization is xavier_uniform
            pass

        return self

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass"""
        if self.input_transform is not None:
            with torch.no_grad():
                x = self.input_transform(
                    x, 
                    [linear.in_features for linear in self.blocks]
                )
        
        # stacked horizontally
        if self.type == 'input':
            assert x.shape[1] == self.in_features
            x = [linear(x) for linear in self.blocks]
            x = torch.cat(x, dim=1)
        # stacked vertically
        elif self.type == 'output':
            in_sizes = [linear.in_features for linear in self.blocks]
            assert x.shape[1] == sum(in_sizes)
            assert all(
                linear.out_features == self.blocks[0].out_features 
                for linear in self.blocks
            )
            x = torch.split(x, in_sizes, dim=1)
            assert len(x) == len(self.blocks)
            x = torch.stack([linear(x_) for linear, x_ in zip(self.blocks, x)], dim=0)
            if BlockLinear.output_aggregation == 'mean':
                x = torch.mean(x, dim=0)
            elif BlockLinear.output_aggregation == 'sum':
                x = torch.sum(x, dim=0)
            else:
                raise ValueError(
                    f"BlockLinear.output_aggregation must be one of ['mean', 'sum'], "
                    f'got {BlockLinear.output_aggregation}'
                )
        # block-diagonal
        else:
            in_sizes = [linear.in_features for linear in self.blocks]
            assert x.shape[1] == sum(in_sizes), \
                f"x.shape[1] ({x.shape[1]}) must be equal to sum(in_sizes) ({sum(in_sizes)})"
            x = torch.split(x, in_sizes, dim=1)
            assert len(x) == len(self.blocks), \
                f"len(x) ({len(x)}) must be equal to len(self.blocks) ({len(self.blocks)})"
            x = [linear(x_) for linear, x_ in zip(self.blocks, x)]
            x = torch.cat(x, dim=1)
        return x


class BlockBatchNorm1d(BlockModule):
    """BatchNorm1d with blocks"""
    def __init__(
        self, 
        num_features: int, 
        block_sizes=None, 
        eps: float = 1e-5, 
        momentum: float = 0.1, 
        affine: bool = True, 
        track_running_stats: bool = True,
        device: Optional[Union[str, int, torch.device]] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        if block_sizes is None:
            block_sizes = [num_features]
        assert sum(block_size for block_size in block_sizes) == num_features, \
            "Sum of num features of all blocks must be equal to num_features"

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device
        self.dtype = dtype

        self.blocks = ModuleList()
        for block_size in block_sizes:
            self.blocks.append(
                BatchNorm1d(
                    block_size, 
                    eps=eps, 
                    momentum=momentum, 
                    affine=affine, 
                    track_running_stats=track_running_stats,
                    device=device,
                    dtype=dtype
                )
            )

    def add_block(self, num_features_block: int, init: str = 'default'):
        """Adds a new block"""
        assert init in ['default', 'zero'], \
            "init must be one of ['default', 'zero']"

        batchnorm = BatchNorm1d(
            num_features_block, 
            eps=self.eps, 
            momentum=self.momentum, 
            affine=self.affine, 
            track_running_stats=self.track_running_stats,
            device=self.device,
            dtype=self.dtype
        )
        self.blocks.append(batchnorm)

        self.num_features += num_features_block

        # Reinitialize added layer weights to match current in_features
        # Note: bias initialization is zero by default
        if init == 'zero':
            batchnorm.weight.data.zero_()
        elif init == 'default':
            # Default initialization is uniform 0, 1
            pass

        return self

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass"""
        assert x.shape[1] == self.num_features, \
            f"Input must have {self.num_features} features, got {x.shape[1]}"
        x = torch.split(
            x, 
            [batchnorm.num_features for batchnorm in self.blocks], 
            dim=1
        )
        x = torch.cat(
            [
                batchnorm(x_) 
                for batchnorm, x_ in zip(self.blocks, x)
            ], 
            dim=1
        )
        return x


def permute_cat_block_output(x, n_cat, block_sizes):
    """Permute output of cat blocks
                               <--- n_cats --->
        |b0_c0, b1_c0, ..., bM_c0|b0_c1, b1_c1, ..., bM_c1|...|b0_cN, b1_cN, ..., bM_cN|
        to 
                               <--- n_blocks --->
        |b0_c0, b0_c1, ..., b0_cN|b1_c0, b1_c1, ..., b1_cN|...|bM_c0, bM_c1, ..., bM_cN|
    """
    # x is (batch_size, n_cat * sum(ith block_size))

    # block_size here is n_cat * ith block_size of previous layer
    block_sizes_before_cat = [block_size // n_cat for block_size in block_sizes]

    # list of (batch_size, sum(ith block_size)) 
    # tensors with len n_cat
    x = torch.split(
        x, 
        x.shape[1] // n_cat, 
        dim=1
    )

    # list of lists of (batch_size, ith block_size) tensors 
    # with len n_cat and len n_blocks
    x = [torch.split(x_, block_sizes_before_cat, dim=1) for x_ in x]

    # list of (n_blocks * n_cat) tensors with len n_blocks
    x = [torch.cat(x_, dim=1) for x_ in zip(*x)]

    # (batch_size, sum(ith block_size) * n_cat)
    x = torch.cat(x, dim=1)  

    return x


def pre_input_dummy(x, block_sizes, message=''):
    """Dummy view for pre-input"""
    torch.set_printoptions(profile="full")
    if message:
        print(message)
    print(x[0].shape)
    print(x[0])
    torch.set_printoptions(profile="default") # reset
    return x


def edgeconv_input_view(x, block_sizes):
    """View for edgeconv input"""
    # x = pre_input_dummy(x, block_sizes, message='edgeconv_input_view pre')
    x = permute_cat_block_output(x, 2, block_sizes)
    # x = pre_input_dummy(x, block_sizes, message='edgeconv_input_view post')
    return x

len_in_features = 17
def postprocessing_input_view(x, block_sizes):
    """View for postprocessing input"""
    # x = pre_input_dummy(x, block_sizes, message='postprocessing_input_view pre')

    block_sizes = (block_sizes[0] - len_in_features, *block_sizes[1:])
    x_to_permute = x[:, len_in_features:]
    x_to_permute = permute_cat_block_output(x_to_permute, 4, block_sizes)
    x = torch.cat([x[:, :len_in_features], x_to_permute], dim=1)
    
    # x = pre_input_dummy(x, block_sizes, message='postprocessing_input_view post')
    
    return x


def readout_input_view(x, block_sizes):
    """View for readout input"""
    # x = pre_input_dummy(x, block_sizes, message='readout_input_view pre')
    x = permute_cat_block_output(x, 3, block_sizes)
    # x = pre_input_dummy(x, block_sizes, message='readout_input_view post')
    return x


def print_layer_norms(model):
    for name, module in model.named_modules():
        if isinstance(module, BlockLinear):
            print(name, torch.norm(module.weight))


class BlockStandardModel(StandardModel):
    def __init__(self, *args, **kwargs):
        # Explicit gradient clipping params are required
        # for manual optimmization
        self.gradient_clip_val = kwargs.pop('gradient_clip_val', None)
        self.gradient_clip_algorithm = kwargs.pop('gradient_clip_algorithm', None)

        super().__init__(*args, **kwargs)

        # Disable automatic optimization: not supported
        # for multiple optimizers
        self.automatic_optimization = False
    
    def training_step(self, train_batch: Data, batch_idx: int) -> Tensor:
        """Perform training step."""
        # Get loss
        loss = super().training_step(train_batch, batch_idx)

        # Get optimizers and schedulers
        optimizer_up_to_last, optimizer_last = self.optimizers()
        scheduler_up_to_last, scheduler_last = self.lr_schedulers()
        
        # Zero gradients
        optimizer_up_to_last.zero_grad()
        optimizer_last.zero_grad()

        # Backward pass
        self.manual_backward(loss)

        # Clip gradients
        if self.gradient_clip_val is not None:
            self.clip_gradients(
                optimizer_up_to_last, 
                gradient_clip_val=self.gradient_clip_val, 
                gradient_clip_algorithm=self.gradient_clip_algorithm
            )
            self.clip_gradients(
                optimizer_last, 
                gradient_clip_val=self.gradient_clip_val, 
                gradient_clip_algorithm=self.gradient_clip_algorithm
            )

        # Update weights
        optimizer_up_to_last.step()
        optimizer_last.step()

        # Update learning rate
        scheduler_up_to_last.step()
        scheduler_last.step()

        # Log learning rate
        self.log('lr_up_to_last', scheduler_up_to_last.get_last_lr()[0])
        self.log('lr_last', scheduler_last.get_last_lr()[0])

    def up_to_last_block_parameters(self):
        for module in self.modules(): 
            if isinstance(module, BlockModule):
                yield from module.up_to_last_block_parameters()

    def last_block_parameters(self):
        for module in self.modules(): 
            if isinstance(module, BlockModule):
                yield from module.last_block_parameters()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Constant rate scheduler & optimizer for blocks up to last
        and configured scheduler & optimizer for last block."""
        assert self._scheduler_class is not None

        # Configure optimizer & scheduler 
        # for all blocks except last
        up_to_last_scheduler_kwargs = {
            "milestones": [
                0,
                self._scheduler_kwargs["milestones"][-1]
            ],
            "pieces": [
                ConstLRSchedulerPiece(
                    self._scheduler_kwargs["pieces"][0].start_lr
                ),
            ],
        }
        optimizer_up_to_last = self._optimizer_class(
            self.up_to_last_block_parameters(), **self._optimizer_kwargs
        )
        scheduler_up_to_last = {
            "scheduler": self._scheduler_class(
                optimizer_up_to_last, **up_to_last_scheduler_kwargs
            ),
            **self._scheduler_config,
        }

        # Configure optimizer & scheduler
        # for last block
        optimizer_last = self._optimizer_class(
            self.last_block_parameters(), **self._optimizer_kwargs
        )
        scheduler_last = {
            "scheduler": self._scheduler_class(
                optimizer_last, **self._scheduler_kwargs
            ),
            **self._scheduler_config,
        }

        return \
            [optimizer_up_to_last, optimizer_last], \
            [scheduler_up_to_last, scheduler_last]


def train_dynedge_blocks(
    config: Dict[str, Any], 
    n_blocks: int, 
    model_save_dir: Path, 
    state_dict_path: Path = None
) -> StandardModel:
    """Trains DynEdge with n_blocks blocks
    
    Basically, for each block we train a new model with same architecture,
    joined with previous models by task head.
    """
    assert n_blocks > 0, f'n_blocks must be > 0, got {n_blocks}'

    # Build empty block model
    labels = {'direction': Direction(azimuth_key=config['truth'][1], zenith_key=config['truth'][0])}
    train_dataloader, validate_dataloader = make_dataloaders(config=config, labels=labels)
    BlockLinear.output_aggregation = config['block']['output_aggregation']
    with \
        patch('icecube_utils.StandardModel', BlockStandardModel), \
        patch('torch.nn.Linear', BlockLinear), \
        patch('torch.nn.BatchNorm1d', BlockBatchNorm1d):
        model = build_model(config, train_dataloader)

    # Load block 0 weights if provided
    if state_dict_path is not None:
        base_model = load_pretrained_model(
            config, 
            str(state_dict_path)
        )
        model.load_state_dict(
            {
                key: value 
                for key, (_, value) in zip(
                    model.state_dict().keys(), 
                    base_model.state_dict().items()
                )
            }
        )
    
        del base_model
        gc.collect()
    # Or train from scratch
    else:
        model = train_dynedge(
            model,
            config,
            train_dataloader, 
            validate_dataloader
        )
        model.save_state_dict(
            str(model_save_dir / 'state_dict_n_blocks_1.pth')
        )

    # Set input and output blocks types (handled differently)
    for name, module in model.named_modules(): 
        if isinstance(module, BlockLinear):
            if name == '_gnn._conv_layers.0.nn.0.linear':
                module.set_type('input')
            elif name.startswith('_tasks') and name.endswith('_affine'):
                module.set_type('output')
            
            if name == '_gnn._post_processing.0.linear':
                module.set_input_transform(postprocessing_input_view)
            elif (
                name != '_gnn._conv_layers.0.nn.0.linear' and 
                '_gnn._conv_layers' in name and 
                'nn.0.linear' in name
            ):
                module.set_input_transform(edgeconv_input_view)
            elif name == '_gnn._readout.0.linear':
                module.set_input_transform(readout_input_view)
            # else:
            #     module.set_input_transform(pre_input_dummy)
    
    initial_block_sizes = {
        name: (
            (block.in_features, block.out_features)
            if isinstance(block, BlockLinear) else
            block.num_features
        )
        for name, block in model.named_modules()
        if isinstance(block, BlockModule)
    }
    # with torch.no_grad():
    #     model(next(iter(train_dataloader)))

    # Train n_blocks - 1 blocks additionaly to first one
    for i in range(n_blocks - 1):
        # Add block
        for name, module in model.named_modules(): 
            if isinstance(module, BlockLinear):
                in_features_block, out_features_block = \
                    initial_block_sizes[name]
                
                if name == '_gnn._post_processing.0.linear':
                    in_features_block = in_features_block - len_in_features

                in_features_block = int(in_features_block * config['block']['block_size_scale'])
                out_features_block = int(out_features_block * config['block']['block_size_scale'])

                if module.type == 'input':
                    in_features_block = module.blocks[0].in_features
                elif module.type == 'output':
                    out_features_block = module.blocks[0].out_features

                with torch.no_grad():
                    module.add_block(
                        in_features_block=in_features_block, 
                        out_features_block=out_features_block,
                        init='zero' if config['block']['zero_new'] else 'default'
                    )
            elif isinstance(module, BlockBatchNorm1d):
                num_features_block = initial_block_sizes[name]
                num_features_block = int(num_features_block * config['block']['block_size_scale'])

                with torch.no_grad():
                    module.add_block(
                        num_features_block=num_features_block,
                        init='default'
                    )
        
        print_layer_norms(model)
        
        # # Replace the task layers with simple freshly initalized 
        # # linear layers of suitable in_features
        # for i in range(len(model._tasks)):
        #     model._tasks[i]._affine = BlockLinear(
        #         in_features=
        #             model._tasks[i]._affine.in_features + 
        #             initial_linear_block_sizes[f'_tasks.{i}._affine'][0],
        #         out_features=model._tasks[i]._affine.out_features,
        #         bias=model._tasks[i]._affine.bias is not None,
        #         device=model._tasks[i]._affine.weight.device,
        #         dtype=model._tasks[i]._affine.weight.dtype,
        #         block_sizes=None,
        #         # intermediate here is on purpose so as it is simple linear layer
        #         type='intermediate',
        #         input_transform=None
        #     )

        # Train
        model = train_dynedge(
            model,
            config,
            train_dataloader, 
            validate_dataloader
        )
        model.save_state_dict(
            str(model_save_dir / f'state_dict_n_blocks_{i + 2}.pth')
        )

    return model


class Transform:
    def __init__(self, features) -> None:
        self.feature_to_index = {feature: i for i, feature in enumerate(features)}

    def __call__(self, input, target=None):
        input_shape = input.shape
        input, target = self.transform(input, target)
        assert input.shape == input_shape

        if target is not None:
            assert 0 <= target['zenith'] <= np.pi, target['zenith']
            assert 0 <= target['azimuth'] <= 2 * np.pi, target['azimuth']
            return input, target

        return input


class RandomTransform(Transform):
    def __init__(self, features, p=0.5):
        super().__init__(features)
        self.p = p
    
    def __call__(self, input, target=None):
        if np.random.rand() < self.p:
            return super().__call__(input, target)
        return input, target


def angles_to_xyz(azimuth, zenith):
    x = np.cos(azimuth) * np.sin(zenith)
    y = np.sin(azimuth) * np.sin(zenith)
    z = np.cos(zenith)
    return x, y, z


def xyz_to_angles(x, y, z):
    norm = np.linalg.norm([x, y, z], ord=2)
    x, y, z = x / norm, y / norm, z / norm

    azimuth = np.arctan2(y, x)
    if azimuth < 0:
        azimuth += 2 * np.pi
    
    zenith = np.arccos(z)
    
    return azimuth, zenith


class FlipTimeTransform(RandomTransform):
    """Inverse time transform"""
    def transform(self, input, target=None):
        # Inveret time
        for feature in ['charge', 'auxiliary']:
            index = self.feature_to_index[feature]
            input[:, index] = input[::-1, index]
        
        # Flip direction
        if target is not None:
            azimuth, zenith = target['azimuth'], target['zenith']
            x, y, z = angles_to_xyz(azimuth, zenith)
            x, y, z = -x, -y, -z
            target['azimuth'], target['zenith'] = xyz_to_angles(x, y, z)

        return input, target
    

class FlipCoordinateTransform(RandomTransform):
    """Flip one of axes transform"""
    def __init__(self, features, p, coordinate):
        super().__init__(features, p)
        assert coordinate in ['x', 'y', 'z']
        self.coordinate = coordinate
        self.coordinate_index = self.feature_to_index[coordinate]
    
    def transform(self, input, target=None):
        input[:, self.coordinate_index] = -input[:, self.coordinate_index]

        if target is not None:
            x, y, z = angles_to_xyz(target['azimuth'], target['zenith'])
            if self.coordinate == 'x':
                x = -x
            elif self.coordinate == 'y':
                y = -y
            elif self.coordinate == 'z':
                z = -z
            target['azimuth'], target['zenith'] = xyz_to_angles(x, y, z)
        
        return input, target
    

def rotate_2d(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def reflect_2d(x, y, k, b):
    # TODO check if it works for other quaters than 2nd
    # Get angle
    alpha = np.pi / 2 + np.arctan(k)

    # Shift to origin
    x_shift, y_shift = \
        b * np.cos(np.pi / 2 - alpha) * np.cos(alpha), \
        b * np.cos(np.pi / 2 - alpha) * np.sin(alpha)
    x, y = x - x_shift, y - y_shift

    # Rotate
    x, y = rotate_2d((0, 0), (x, y), -alpha)

    # Reflect
    x = -x

    # Rotate back
    x, y = rotate_2d((0, 0), (x, y), alpha)

    # Shift back
    x, y = x + x_shift, y + y_shift

    return x, y

class FlipOverXYLineTransform(RandomTransform):
    def __init__(self, features, p, k, b):
        super().__init__(features, p)
        self.k = k
        self.b = b

    def transform(self, input, target=None):
        x = input[:, self.feature_to_index['x']]
        y = input[:, self.feature_to_index['y']]
        z = input[:, self.feature_to_index['z']]

        # Flip over line
        x, y = reflect_2d(x, y, self.k, self.b)

        # Flip direction
        if target is not None:
            azimuth, zenith = target['azimuth'], target['zenith']
            x_angles, y_angles, z_angles = angles_to_xyz(azimuth, zenith)
            x_angles, y_angles = reflect_2d(x_angles, y_angles, self.k, 0)
            target['azimuth'], target['zenith'] = xyz_to_angles(x_angles, y_angles, z_angles)

        input[:, self.feature_to_index['x']] = x
        input[:, self.feature_to_index['y']] = y
        input[:, self.feature_to_index['z']] = z

        return input, target


def rotate(v, k, angle):
    """Rotate vectors v around axis k by angle (right-hand rule).
    v: batch of 3D vectors to rotate, shape (B, 3)
    k: 3D unit-vector, shape (3,)
    angle: rotation angle in radians, shape (1,)
    """
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    assert np.isclose(np.linalg.norm(k, ord=2), 1.0)
    k = k[None, :]  # 1 x 3
    cos, sin = np.cos(angle), np.sin(angle)
    return (
        v * cos +  # B x 3
        np.cross(k, v) * sin +  # B x 3
        k * (np.dot(k, v.T)).T * (1 - cos)  # 1 x 3 * (1 x 3 . (B x 3).T).T = B x 3
    )


def rotate_azimuth(v, angle):
    """Rotate vectors given by coordinates x, y, z 
    by azimuth angle ~ around z-axis by angle (right-hand rule).
    v: batch of 3D vectors to rotate, shape (B, 3)
    angle: rotation angle in radians, shape (1,)
    """
    k = np.array([0, 0, 1])  # 3
    v_new = rotate(v, k, angle)  # B x 3
    return v_new


def rotate_zenith(v, w1, w2, angle):
    """Rotate vectors given by coordinates x, y, z 
    by zenith angle (right-hand rule).
    v: batch of 3D vectors to rotate, shape (B, 3)
    w1, w2: base vector of current direction, shape (3,) each
    angle: rotation angle in radians, shape (1,)
    """
    k = np.cross(w1, w2)  # 3
    k = k / np.linalg.norm(k, ord=2)
    v_new = rotate(v, k, angle)
    return v_new


class RotateAngleTransform(RandomTransform):
    """Rotate angle transform"""
    def __init__(self, features, p, angle):
        super().__init__(features, p)
        self.angle = angle

        # TODO: make it more general
        assert self.feature_to_index['x'] == 0
        assert self.feature_to_index['y'] == 1
        assert self.feature_to_index['z'] == 2

    def transform(self, input, target):
        assert target is not None, 'Target is required for rotation'
        v = input[:, :3]

        if self.angle == 'azimuth':
            azimuth = target['azimuth']
            new_azimuth = np.random.rand() * 2 * np.pi
            if np.isclose(azimuth, new_azimuth):
                return input, target
            
            angle = new_azimuth - azimuth
            k = np.array([0, 0, 1])

            target['azimuth'] = new_azimuth
        elif self.angle == 'zenith':
            zenith, azimuth = target['zenith'], target['azimuth']
            new_zenith = np.random.rand() * np.pi
            if np.isclose(zenith, new_zenith):
                return input, target

            angle = new_zenith - zenith

            # Distinct vectors in a plane of rotation
            w1 = np.array(angles_to_xyz(azimuth, zenith))
            w2 = np.array(angles_to_xyz(azimuth, zenith + angle / 2))
            k = np.cross(w1, w2)
            k = k / np.linalg.norm(k, ord=2)

            target['zenith'] = new_zenith

        v = rotate(v, k, angle)
        input = np.concatenate((v, input[:, 3:]), axis=1)

        return input, target


class OneOfTransform:
    """Apply one of transforms"""
    def __init__(self, transforms):
        self.transforms = transforms

    def transform(self, input, target=None):
        transform = np.random.choice(self.transforms)
        return transform.transform(input, target)


class CancelAzimuthByPredictionTransform(Transform):
    """Cancel azimuth by auxilary azimuth_pred feature"""
    def __init__(self, features, gt=False):
        super().__init__(features)

        self.gt = gt

        # TODO: make it more general
        assert self.feature_to_index['x'] == 0
        assert self.feature_to_index['y'] == 1
        assert self.feature_to_index['z'] == 2

    def transform(self, input, target):
        if self.gt:
            angle = target['azimuth']
        else:
            angle = target['azimuth_pred']
        v = input[:, :3]
        k = np.array([0, 0, 1])
        v = rotate(v, k, -angle)
        input = np.concatenate((v, input[:, 3:]), axis=1)
        return input, target


# Replace all the models' callstacks so they can forward coeffs_t
# down to simplex linear layers

# https://stackoverflow.com/questions/10874432/possible-to-change-function-name-in-definition
# necessary to make torch_geometric inspection modules happy again after mocking
def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator

@rename('forward')
def StandardModelGraphnet_forward(self, data: Data, coeffs_t) -> List[Union[Tensor, Data]]:
    """Forward pass, chaining model components."""
    if self._coarsening:
        data = self._coarsening(data)
    data = self._detector(data)
    x = self._gnn(data, coeffs_t)
    preds = [task(x, coeffs_t) for task in self._tasks]
    return preds
    

@rename('forward')
def DynEdgeGraphnet_forward(self, data: Data, coeffs_t) -> Tensor:
    """Apply learnable forward pass."""
    # Convenience variables
    x, edge_index, batch, n_pulses = data.x, data.edge_index, data.batch, data.n_pulses

    global_variables = self._calculate_global_variables(
        x,
        edge_index,
        batch,
        torch.log10(n_pulses),
    )

    # Distribute global variables out to each node
    if not self._add_global_variables_after_pooling:
        distribute = (
            batch.unsqueeze(dim=1) == torch.unique(batch).unsqueeze(dim=0)
        ).type(torch.float)

        global_variables_distributed = torch.sum(
            distribute.unsqueeze(dim=2)
            * global_variables.unsqueeze(dim=0),
            dim=1,
        )

        x = torch.cat((x, global_variables_distributed), dim=1)

    # DynEdge-convolutions
    skip_connections = [x]
    for conv_layer in self._conv_layers:
        x, edge_index = conv_layer(x, edge_index, batch, coeffs_t)
        skip_connections.append(x)

    # Skip-cat
    x = torch.cat(skip_connections, dim=1)

    # Post-processing
    x = self._post_processing(x, coeffs_t)

    # (Optional) Global pooling
    if self._global_pooling_schemes:
        x = self._global_pooling(x, batch=batch)
        if self._add_global_variables_after_pooling:
            x = torch.cat(
                [
                    x,
                    global_variables,
                ],
                dim=1,
            )

    # Read-out
    x = self._readout(x, coeffs_t)

    return x


@rename('forward')
def DynEdgeConvGraphnet_forward(
    self, x: Tensor, edge_index: Adj, batch: Optional[Tensor] = None, 
    coeffs_t: Optional[Tensor] = None
) -> Tensor:
    """Forward pass."""
    # Standard EdgeConv forward pass
    x = super(DynEdgeConv, self).forward(x, edge_index, coeffs_t)

    # Recompute adjacency
    edge_index = knn_graph(
        x=x[:, self.features_subset],
        k=self.nb_neighbors,
        batch=batch,
    ).to(self.device)

    return x, edge_index


@rename('forward')
def EdgeConvGraphnet_forward(
        self, 
        x: Union[Tensor, PairTensor], 
        edge_index: Adj, 
        coeffs_t: Optional[Tensor] = None
    ) -> Tensor:
    if isinstance(x, Tensor):
        x: PairTensor = (x, x)
    # propagate_type: (x: PairTensor)
    return self.propagate(edge_index, x=x, size=None, coeffs_t=coeffs_t)


@rename('message')
def EdgeConvGraphnet_message(self, x_i: Tensor, x_j: Tensor, coeffs_t: Optional[Tensor] = None) -> Tensor:
    return self.nn(torch.cat([x_i, x_j - x_i], dim=-1), coeffs_t)


@rename('forward')
def SequentialGraphnet_forward(self, input, coeffs_t):
    for module in self:
        if isinstance(module, SimplexModule):
            input = module(input, coeffs_t)
        else:
            input = module(input)
    return input


@rename('forward')
def Task_forward(self, x: Union[Tensor, Data], coeffs_t) -> Union[Tensor, Data]:
    """Forward pass."""
    self._regularisation_loss = 0  # Reset
    x = self._affine(x, coeffs_t)
    x = self._forward(x)
    return self._transform_prediction(x)


class SimplexNetGraphnet(Model):
    def __init__(
        self, 
        architecture, 
        n_verts, 
        LMBD, 
        nsample,
        base_model=None, 
        optimizer_class: type = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
        simplex_volume_loss_enabled: bool = True,
        infrerence_sampling_average: str = 'angles',
        infrerence_sampling_topk: int = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['base_model'])

        with torch.no_grad():
            self.reg_pars = []
            for ii in range(0, n_verts + 2):
                fix_pts = [True]*(ii + 1)
                start_vert = len(fix_pts)

                out_dim = 10
                simplex_model = SimplexNet(out_dim, architecture, n_vert=start_vert,
                                    fix_points=fix_pts)
                simplex_model = simplex_model.cuda()
                
                log_vol = (simplex_model.total_volume() + 1e-4).log()
                
                self.reg_pars.append(max(float(LMBD)/log_vol, 1e-8))
                del simplex_model
        
        fix_pts = [True]
        n_vert = len(fix_pts)
        self.simplex_model = SimplexNet(10, architecture, n_vert=n_vert,
                           fix_points=fix_pts).cuda()
        self.prediction_columns = self.simplex_model.net.prediction_columns
        self.additional_attributes = self.simplex_model.net.additional_attributes
        if base_model is not None:
            self.simplex_model.import_base_parameters(base_model, 0)

        self.n_verts = n_verts
        self.nsample = nsample
        self.infrerence_sampling_topk = infrerence_sampling_topk
        self.current_vol_reg = self.reg_pars[0]
        self.simplex_volume_loss_enabled = simplex_volume_loss_enabled
        assert infrerence_sampling_average in ['angles', 'direction']
        self.infrerence_sampling_average = infrerence_sampling_average

        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or dict()
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or dict()
        self._scheduler_config = scheduler_config or dict()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the model's optimizer(s)."""
        optimizer = self._optimizer_class(
            self.parameters(), **self._optimizer_kwargs
        )
        config = {
            "optimizer": optimizer,
        }
        if self._scheduler_class is not None:
            scheduler = self._scheduler_class(
                optimizer, **self._scheduler_kwargs
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        **self._scheduler_config,
                    },
                }
            )
        return config

    def _get_batch_size(self, data: Data) -> int:
        return torch.numel(torch.unique(data.batch))
    
    def load_state_dict(self, path_or_state_dict: Union[str, Dict], strict: bool = True) -> "Model":
        """Load model state dict from `path`."""
        if isinstance(path_or_state_dict, str):
            state_dict = torch.load(path_or_state_dict)
        else:
            state_dict = path_or_state_dict
        state_dict = {k.replace('simplex_model.', ''): v for k, v in state_dict.items()}
        self.simplex_model.load_state_dict(state_dict, strict=strict)
        return self

    def inference(self) -> None:
        """Activate inference mode."""
        self.simplex_model.eval()
        self.simplex_model.net.inference()

    def train(self, mode: bool = True) -> "Model":
        """Deactivate inference mode."""
        super().train(mode)
        self.simplex_model = self.simplex_model.train(mode)
        self.simplex_model.net.train(mode)
        return self

    def shared_step(self, batch: Data, batch_idx: int) -> Tensor:
        """Perform shared step.

        Applies the forward pass and the following loss calculation, shared
        between the training and validation step.
        """
        loss = None
        for _ in range(self.nsample):
            output = self.simplex_model(batch)
            if loss is None:
                loss = self.simplex_model.net.compute_loss(output, batch)
            else:
                loss = loss + self.simplex_model.net.compute_loss(output, batch)
        loss = loss / self.nsample

        volume_loss_pos = 0
        if self.simplex_volume_loss_enabled:
            vol = self.simplex_model.total_volume()
            log_vol = (vol + 1e-4).log()

            volume_loss_pos = self.current_vol_reg * log_vol
            loss = loss - volume_loss_pos
        return loss, -volume_loss_pos

    def training_step(self, batch, batch_idx):
        loss, volume_loss = self.shared_step(batch, batch_idx)
        self.log(
            "train_loss",
            loss,
            batch_size=self._get_batch_size(batch),
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "train_volume_loss",
            volume_loss,
            batch_size=self._get_batch_size(batch),
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )

        return {
            'loss': loss,
            'volume_loss': volume_loss,
        }
    
    def validation_step(self, batch: Data, batch_idx: int) -> Tensor:
        """Perform validation step."""
        loss, volume_loss = self.shared_step(batch, batch_idx)
        self.log(
            "val_loss",
            loss,
            batch_size=self._get_batch_size(batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "val_volume_loss",
            volume_loss,
            batch_size=self._get_batch_size(batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    def forward(self, data: Data) -> List[Union[Tensor, Data]]:
        """Perform forward pass."""
        # Sample
        output = []
        for _ in range(self.nsample):
            if not output:
                for task_output in self.simplex_model(data):
                    output.append([task_output])
            else:
                for i, task_output in enumerate(self.simplex_model(data)):
                    output[i].append(task_output)

        # Average over samples (in angles space)
        for i in range(len(output)):
            if self.infrerence_sampling_average == 'angles':
                task_output = output[i]  # list of nsamples, each is (nbatch, 4)
                task_output = torch.stack(task_output, dim=0)  # (nsamples, nbatch, 4)

                # Select top-k by kappa
                if self.infrerence_sampling_topk is not None:
                    kappa = task_output[0][:, 3]  # (nsamples, nbatch)
                    _, topk = kappa.topk(self.infrerence_sampling_topk, dim=0)  # (nbatch, k)
                    task_output = task_output[topk, torch.arange(task_output.shape[1])]  # (k, nbatch)

                # Normalize
                x, y, z = \
                    task_output[:, :, 0], \
                    task_output[:, :, 1], \
                    task_output[:, :, 2]  # each is (k, nbatch)
                norm = (x ** 2 + y ** 2 + z ** 2).sqrt()
                x = x / norm
                y = y / norm
                z = z / norm

                # Convert to angles & average
                azimuth = torch.atan2(y, x).mean(dim=0)  # (nbatch)
                zenith = torch.acos(z).mean(dim=0)  # (nbatch)

                # Convert back to cartesian
                x = torch.cos(azimuth) * torch.sin(zenith)  # (nbatch)
                y = torch.sin(azimuth) * torch.sin(zenith)  # (nbatch)
                z = torch.cos(zenith)  # (nbatch)
                kappa = task_output[:, :, 3].mean(dim=0)  # (nbatch)

                # Save back
                output[i] = torch.stack([x, y, z, kappa], dim=1)
            elif self.infrerence_sampling_average == 'direction':
                output[i] = torch.stack(output[i], dim=0).mean(dim=0)

        return output
    
    def add_vert(self, vertex_index: int):
        self.current_vol_reg = self.reg_pars[vertex_index]
        self.simplex_model.add_vert()

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        *,
        max_epochs: int = 10,
        gpus: Optional[Union[List[int], int]] = None,
        callbacks: Optional[List[Callback]] = None,
        ckpt_path: Optional[str] = None,
        logger: Optional[Logger] = None,
        log_every_n_steps: int = 1,
        gradient_clip_val: Optional[float] = None,
        distribution_strategy: Optional[str] = "ddp",
        **trainer_kwargs: Any,
    ) -> None:
        """Fit `Model` using `pytorch_lightning.Trainer`."""
        self.train(mode=True)
        for vertex_index in range(1, self.n_verts + 1):
            self.add_vert(vertex_index)
            self.simplex_model = self.simplex_model.cuda()
            super().fit(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                max_epochs=max_epochs,
                gpus=gpus,
                callbacks=callbacks,
                ckpt_path=ckpt_path,
                logger=logger,
                log_every_n_steps=log_every_n_steps,
                gradient_clip_val=gradient_clip_val,
                distribution_strategy='auto',
                **trainer_kwargs
            )

    def enable_simplex_volume_loss(self, enable: bool):
        self.simplex_volume_loss_enabled = enable

    # def on_before_optimizer_step(self, optimizer):
    #     print(f'grad_norm: {grad_norm(self, norm_type=2)}')

class EnableSimplexVolumeLossCallback(Callback):
    def __init__(self, enable_on_step: int, reset_on_fit_start: bool = True):
        self.enable_on_step = enable_on_step
        self.steps = 0
        self.reset_on_fit_start = reset_on_fit_start
    
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        """Called when the train batch ends.

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.
        """
        if self.steps >= self.enable_on_step:
            pl_module.enable_simplex_volume_loss(True)
        self.steps += 1

    def on_fit_start(self, trainer, pl_module) -> None:
        if self.reset_on_fit_start:
            pl_module.enable_simplex_volume_loss(False)
            self.steps = 0


class EarlyStoppingTrainStepCallback(Callback):
    mode_dict = {"min": torch.lt, "max": torch.gt}

    def __init__(self, monitor, start_step, mode='min', min_delta=0.0, patience=0):
        self.monitor = monitor

        assert start_step >= 0, f"EarlyStopping requires start_step >= 0, got {start_step}"
        self.start_step = start_step

        assert mode in self.mode_dict, f"mode {mode} is unknown!"
        self.monitor_op = self.mode_dict[mode]

        assert min_delta >= 0, f"EarlyStopping requires min_delta >= 0, got {min_delta}"
        self.min_delta = torch.tensor(min_delta)
        self.min_delta *= (1 if self.monitor_op == torch.gt else -1)

        assert patience >= 0, f"EarlyStopping requires patience >= 0, got {patience}"
        self.patience = patience

        self.best = None
        self.wait = 0
        self.steps = 0
    
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        """Called when the train batch ends.

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.
        """
        if self.steps >= self.start_step:
            if self.best is None:
                self.best = outputs[self.monitor]
            else:
                if self.monitor_op(
                    outputs[self.monitor] - self.min_delta, 
                    self.best
                ):
                    self.best = outputs[self.monitor]
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        trainer.should_stop = True
        self.steps += 1

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.best = None
        self.wait = 0
        self.steps = 0


@patch('icecube_utils.StandardModel.forward', StandardModelGraphnet_forward)
@patch('icecube_utils.DynEdge.forward', DynEdgeGraphnet_forward)
@patch('graphnet.models.gnn.dynedge.DynEdgeConv.forward', DynEdgeConvGraphnet_forward)
@patch('graphnet.models.components.layers.EdgeConv.forward', EdgeConvGraphnet_forward)
@patch('graphnet.models.components.layers.EdgeConv.message', EdgeConvGraphnet_message)
@patch('torch.nn.Sequential.forward', SequentialGraphnet_forward)
@patch('graphnet.models.task.reconstruction.Task.forward', Task_forward)
def build_model_simplex(
    config, 
    state_dict_path=None,
    is_inference=False
):
    base_model = None
    if state_dict_path is not None:
        base_model = load_pretrained_model(
            config, 
            str(state_dict_path)
        )
    labels = None
    if is_inference:
        labels = {'direction': Direction(azimuth_key=config['truth'][1], zenith_key=config['truth'][0])}
    train_dataloader, validate_dataloader = make_dataloaders(config=config, labels=labels)
    
    def build_model_wrapper(n_output, fix_points, **architecture_kwargs):
        model = build_model(config, train_dataloader, fix_points)
        return model

    scheduler_kwargs = make_scheduler_kwargs(config, train_dataloader)
    simplex_model_wrapper = SimplexNetGraphnet(
        architecture=build_model_wrapper, 
        n_verts=config['simplex']['n_verts'], 
        LMBD=config['simplex']['LMBD'], 
        nsample=config['simplex']['nsample'],
        base_model=base_model, 
        optimizer_class=config['optimizer_class'],
        optimizer_kwargs=config["optimizer_kwargs"],
        scheduler_class=PiecewiceFactorsLRScheduler,
        scheduler_kwargs=scheduler_kwargs,
        scheduler_config={
            "interval": "step",
        },
        simplex_volume_loss_enabled=False,
        infrerence_sampling_average=config['simplex']['infrerence_sampling_average'],
        infrerence_sampling_topk=config['simplex']['infrerence_sampling_topk'],
    )

    # If building for inference, add vertices
    if is_inference:
        for vertex_index in range(1, simplex_model_wrapper.n_verts + 1):
            simplex_model_wrapper.add_vert(vertex_index)

    return simplex_model_wrapper, train_dataloader, validate_dataloader


@patch('icecube_utils.StandardModel.forward', StandardModelGraphnet_forward)
@patch('icecube_utils.DynEdge.forward', DynEdgeGraphnet_forward)
@patch('graphnet.models.gnn.dynedge.DynEdgeConv.forward', DynEdgeConvGraphnet_forward)
@patch('graphnet.models.components.layers.EdgeConv.forward', EdgeConvGraphnet_forward)
@patch('graphnet.models.components.layers.EdgeConv.message', EdgeConvGraphnet_message)
@patch('torch.nn.Sequential.forward', SequentialGraphnet_forward)
@patch('graphnet.models.task.reconstruction.Task.forward', Task_forward)
def train_dynedge_simplex(
    config,
    state_dict_path
):
    simplex_model_wrapper, train_dataloader, validate_dataloader = \
        build_model_simplex(
            config, 
            state_dict_path,
            is_inference=False
        )
    start_simplex_steps = 100
    simplex_model_wrapper = train_dynedge(
        simplex_model_wrapper,
        config,
        train_dataloader, 
        validate_dataloader,
        callbacks=[
            EnableSimplexVolumeLossCallback(start_simplex_steps, reset_on_fit_start=True),
            EarlyStoppingTrainStepCallback(
                monitor='volume_loss',
                start_step=start_simplex_steps + 1,
                mode='min',
                min_delta=0.0,
                patience=5000,  # 100 log steps
            )
        ]
    )
    return simplex_model_wrapper


@patch('icecube_utils.StandardModel.forward', StandardModelGraphnet_forward)
@patch('icecube_utils.DynEdge.forward', DynEdgeGraphnet_forward)
@patch('graphnet.models.gnn.dynedge.DynEdgeConv.forward', DynEdgeConvGraphnet_forward)
@patch('graphnet.models.components.layers.EdgeConv.forward', EdgeConvGraphnet_forward)
@patch('graphnet.models.components.layers.EdgeConv.message', EdgeConvGraphnet_message)
@patch('torch.nn.Sequential.forward', SequentialGraphnet_forward)
@patch('graphnet.models.task.reconstruction.Task.forward', Task_forward)
def inference_simplex(model, config: Dict[str, Any], use_labels: bool) -> pd.DataFrame:
    return inference(model, config, use_labels)

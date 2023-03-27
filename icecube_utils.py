
import gc
from pathlib import Path
import numpy as np
import os
import torch
import pandas as pd
import gc, os
import numpy as np
from mock import patch
from typing import Any, Dict, List
from pytorch_lightning.callbacks import (
    EarlyStopping, 
    GradientAccumulationScheduler,
    ModelCheckpoint,
    LearningRateMonitor
)
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import Module, Linear, ModuleList
from torch import Tensor
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeKaggle
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import AngleReconstructionCos, AngleReconstructionSinCos, AngleReconstructionSincosWithKappa, AzimuthReconstruction, AzimuthReconstructionWithKappa, DirectionReconstruction, DirectionReconstructionWithKappa, ZenithAzimuthReconstruction, ZenithReconstructionWithKappa
from graphnet.training.callbacks import ProgressBar
from graphnet.training.loss_functions import CosineLoss, CosineLoss3D, EuclidianDistanceLossCos, EuclidianDistanceLossSinCos, VonMisesFisher2DLoss, VonMisesFisher2DLossSinCos, VonMisesFisher3DLoss
from graphnet.training.labels import Direction
from graphnet.training.utils import make_dataloader
from graphnet.utilities.logging import get_logger
from graphnet.data.sqlite import SQLiteDataset
from graphnet.data.parquet import ParquetDataset, SequentialParquetDataset


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
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, pct):
        return self.lr
    

class LinearLRSchedulerPiece:
    def __init__(self, start_lr, stop_lr):
        self.start_lr = start_lr
        self.stop_lr = stop_lr

    def __call__(self, pct):
        return self.start_lr + pct * (self.stop_lr - self.start_lr)


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
        bias=config['bias'],
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
            bias=config['bias'],
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
            bias=config['bias'],
            fix_points=fix_points,
        )
        tasks.append(task)
        task = AzimuthReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=config['truth'][1],
            loss_function=VonMisesFisher2DLoss(),
            loss_weight='loss_weight' if config['loss_weight'] else None,
            bias=config['bias'],
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
            bias=config['bias'],
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
    #         bias=config['bias'],
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
            bias=config['bias'],
            fix_points=fix_points,
        )
        tasks.append(task)
        task = AngleReconstructionSincosWithKappa(
            half=False,
            hidden_size=gnn.nb_outputs,
            target_labels=config['truth'][1],
            loss_function=VonMisesFisher2DLossSinCos(),
            loss_weight='loss_weight' if config['loss_weight'] else None,
            bias=config['bias'],
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
            bias=config['bias'],
            fix_points=fix_points,
        )
        tasks.append(task)
        task = AngleReconstructionSinCos(
            half=False,
            hidden_size=gnn.nb_outputs,
            target_labels=config['truth'][1],
            loss_function=EuclidianDistanceLossSinCos(),
            loss_weight='loss_weight' if config['loss_weight'] else None,
            bias=config['bias'],
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
            bias=config['bias'],
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
            bias=config['bias'],
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
            bias=config['bias'],
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
            bias=config['bias'],
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
            bias=config['bias'],
            fix_points=fix_points,
        )
        tasks.append(task)
        prediction_columns = [
            'direction_x', 
            'direction_y',
            'direction_z',
        ]
        additional_attributes = [*config['truth'], 'event_id']


    assert "scheduler_kwargs" in config and "pieces" in config["scheduler_kwargs"]

    # Pytorch schedulers are parametrized on number of steps
    # and not in percents, so scheduler_kwargs need to be
    # constructed here
    assert len(config["scheduler_kwargs"]["pieces"]) == 2, \
        "Only 2 pieces one-cycle LR is supported for now"
    scheduler_kwargs = {
        # 0.5 epoch warmup piece + rest piece
        "milestones": [
            0,
            len(train_dataloader) / 2,
            len(train_dataloader) * config["fit"]["max_epochs"],
        ],
        "pieces": config["scheduler_kwargs"]["pieces"],
    }

    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=tasks,
        tasks_weiths=config["tasks_weights"] if "tasks_weights" in config else None,
        optimizer_class=AdamW,
        optimizer_kwargs=config["optimizer_kwargs"],
        scheduler_class=PiecewiceFactorsLRScheduler,
        scheduler_kwargs=scheduler_kwargs,
        scheduler_config={
            "interval": "step",
        },
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

def make_dataloaders(config: Dict[str, Any]) -> List[Any]:
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
    elif config['dataset_type'] == 'sequential_parquet':
        dataset_class = SequentialParquetDataset
        dataset_kwargs = dict(
            geometry_path=config['sequential_parquet']['geometry_path'],
            meta_path=config['sequential_parquet']['meta_path'],
            shuffle_outer=config['sequential_parquet']['shuffle_outer'],
            shuffle_inner=config['sequential_parquet']['shuffle_inner'],
        )
    else:
        raise ValueError(f'Unknown dataset type {config["dataset_type"]}')


    if config['dataset_type'] == 'sequential_parquet':
        dataset_kwargs['filepathes'] = config['sequential_parquet']['train_path'].glob('**/*.parquet')
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
        labels = {'direction': Direction(azimuth_key=config['truth'][1], zenith_key=config['truth'][0])},
        index_column = config['index_column'],
        truth_table = config['truth_table'],
        transforms = config['train_transforms'],
        **loss_weight_kwargs,
        **max_n_pulses_kwargs,
        **dataset_kwargs
    )
    
    if config['dataset_type'] == 'sequential_parquet':
        dataset_kwargs['filepathes'] = config['sequential_parquet']['val_path'].glob('**/*.parquet')
        dataset_kwargs.pop('shuffle_outer', None)
        dataset_kwargs.pop('shuffle_inner', None)
    validate_dataloader = make_dataloader(
        dataset_class = dataset_class,
        db = config['inference_database_path'],
        selection = None,
        pulsemaps = config['pulsemap'],
        features = config['features'],
        truth = config['truth'],
        batch_size = config['batch_size'],
        num_workers = config['num_workers'],
        shuffle = False,
        labels = {'direction': Direction(azimuth_key=config['truth'][1], zenith_key=config['truth'][0])},
        index_column = config['index_column'],
        truth_table = config['truth_table'],
        max_n_pulses=config['max_n_pulses']['max_n_pulses'],
        max_n_pulses_strategy="clamp",
        transforms = config['val_transforms'],
        **dataset_kwargs
    )
    return train_dataloader, validate_dataloader


def train_dynedge(model, config, train_dataloader, validate_dataloader):
    # Compile model
    torch.compile(model)

    # Training model
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        GradientAccumulationScheduler(
            scheduling={0: config['accumulate_grad_batches']}
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
        ),
        ProgressBar(),
    ]

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

    train_dataloader, validate_dataloader = make_dataloaders(config = config)

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
    # Make Dataloader
    test_dataloader = make_dataloader(
        db = config['inference_database_path'],
        selection = None, # Entire database
        pulsemaps = config['pulsemap'],
        features = config['features'],
        truth = config['truth'],
        batch_size = config['batch_size'],
        num_workers = config['num_workers'],
        shuffle = False,
        labels = labels,
        index_column = config['index_column'],
        truth_table = config['truth_table'],
        max_n_pulses = config['max_n_pulses']['max_n_pulses'],
        max_n_pulses_strategy='clamp',
        transforms = config['val_transforms'],
    )
    
    # Get predictions
    with torch.no_grad():
        results = model.predict_as_dataframe(
            gpus = [0],
            dataloader = test_dataloader,
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


class BlockLinear(Module):
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
        
        self.linears = ModuleList()
        for block_size in block_sizes:
            self.linears.append(
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
                for linear in self.linears
            )
        elif type == 'output':
            assert all(
                linear.out_features == self.out_features 
                for linear in self.linears
            )
        self.type = type
        return self

    def set_input_transform(self, input_transform):
        self.input_transform = input_transform
        return self

    @property
    def bias(self):
        if self.linears[0].bias is None:
            return None
        return torch.cat([linear.bias for linear in self.linears], dim=0)
    
    @property
    def weight(self):
        return torch.block_diag(*[linear.weight for linear in self.linears])
    
    def freeze_first_n_blocks(self, n: int):
        """Freezes first n blocks"""
        for i, linear in enumerate(self.linears):
            if i < n:
                linear.weight.requires_grad = False
                if linear.bias is not None:
                    linear.bias.requires_grad = False
        return self

    def freeze_except_last_block(self):
        """Freezes all blocks except last"""
        self.freeze_first_n_blocks(len(self.linears) - 1)
        return self

    def add_block(self, in_features_block: int, out_features_block: int, init: str = 'xavier_uniform'):
        """Adds a new block"""
        assert init in ['xavier_uniform', 'zero'], \
            "init must be one of ['xavier_uniform', 'zero']"

        linear = Linear(
            in_features_block, 
            out_features_block, 
            bias=self.bias is not None,
            device=self.linears[0].weight.device,
            dtype=self.linears[0].weight.dtype
        )
        self.linears.append(linear)

        # # Re-normalize weights of existing blocks
        # scale = (len(self.linears) - 1) / len(self.linears)
        # with torch.no_grad():
        #     for linear in self.linears[:-1]:
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
        elif init == 'xavier_uniform':
            # Default initialization is xavier_uniform
            pass

        return self

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass"""
        if self.input_transform is not None:
            with torch.no_grad():
                x = self.input_transform(x, len(self.linears))
        
        # stacked horizontally
        if self.type == 'input':
            assert x.shape[1] == self.in_features
            x = [linear(x) for linear in self.linears]
            x = torch.cat(x, dim=1)
        # stacked vertically
        elif self.type == 'output':
            in_sizes = [linear.in_features for linear in self.linears]
            assert x.shape[1] == sum(in_sizes)
            assert all(
                linear.out_features == self.linears[0].out_features 
                for linear in self.linears
            )
            x = torch.split(x, in_sizes, dim=1)
            assert len(x) == len(self.linears)
            x = torch.stack([linear(x_) for linear, x_ in zip(self.linears, x)], dim=0)
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
            in_sizes = [linear.in_features for linear in self.linears]
            assert x.shape[1] == sum(in_sizes)
            x = torch.split(x, in_sizes, dim=1)
            assert len(x) == len(self.linears)
            x = [linear(x_) for linear, x_ in zip(self.linears, x)]
            x = torch.cat(x, dim=1)
        return x


def permute_cat_block_output(x, n_cat, n_blocks):
    """Permute output of cat blocks"""
    return x.reshape(
        x.shape[0], n_cat, n_blocks, -1
    ).permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1])


def pre_input_dummy(x, n_blocks):
    """Dummy view for pre-input"""
    torch.set_printoptions(profile="full")
    print(x[0])
    torch.set_printoptions(profile="default") # reset
    return x


def edgeconv_input_view(x, n_blocks):
    """View for edgeconv input"""
    # x = pre_input_dummy(x, n_blocks)
    return permute_cat_block_output(x, 2, n_blocks)


def postprocessing_input_view(x, n_blocks):
    """View for postprocessing input"""
    # x = pre_input_dummy(x, n_blocks)

    # torch.set_printoptions(profile="full")
    # print(x[0])
    x_to_permute = x[:, 17:]
    x_to_permute = permute_cat_block_output(x_to_permute, 4, n_blocks)
    x = torch.cat([x[:, :17], x_to_permute], dim=1)
    # print(x[0])
    # torch.set_printoptions(profile="default") # reset
    return x


def readout_input_view(x, n_blocks):
    """View for readout input"""
    # x = pre_input_dummy(x, n_blocks)
    return permute_cat_block_output(x, 3, n_blocks)


def print_layer_norms(model):
    for name, module in model.named_modules():
        if isinstance(module, BlockLinear):
            print(name, torch.norm(module.weight))


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
    train_dataloader, validate_dataloader = make_dataloaders(config=config)
    BlockLinear.output_aggregation = config['block_output_aggregation']
    with patch('torch.nn.Linear', BlockLinear):
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

    # Set initial sizes to add blocks of these sizes
    initial_linear_block_sizes = {
        name: (module.in_features, module.out_features) 
        for name, module in model.named_modules()
        if isinstance(module, BlockLinear)
    }

    # Set input and output blocks types (handled differently)
    for name, module in model.named_modules(): 
        if isinstance(module, BlockLinear):
            if name == '_gnn._conv_layers.0.nn.0':
                module.set_type('input')
            elif name.startswith('_tasks') and name.endswith('_affine'):
                module.set_type('output')
            
            if name == '_gnn._post_processing.0':
                module.set_input_transform(postprocessing_input_view)
            elif (
                name != '_gnn._conv_layers.0.nn.0' and 
                '_gnn._conv_layers' in name and 
                'nn.0' in name
            ):
                module.set_input_transform(edgeconv_input_view)
            elif name == '_gnn._readout.0':
                module.set_input_transform(readout_input_view)
            # else:
            #     module.set_input_transform(pre_input_dummy)
    
    # with torch.no_grad():
    #     model(next(iter(train_dataloader)))

    # Train n_blocks - 1 blocks additionaly to first one
    for i in range(n_blocks - 1):
        print_layer_norms(model)

        # Add block and freeze all the previous blocks
        for name, module in model.named_modules(): 
            if isinstance(module, BlockLinear):
                in_features_block, out_features_block = initial_linear_block_sizes[name]
                
                if name == '_gnn._post_processing.0':
                    in_features_block = in_features_block - 17
                
                with torch.no_grad():
                    module.add_block(
                        in_features_block=in_features_block, 
                        out_features_block=out_features_block,
                        init='zero' if config['zero_new_block'] else 'xavier_uniform'
                    )
                    module.freeze_except_last_block()
        
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

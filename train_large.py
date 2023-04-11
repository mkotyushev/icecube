import argparse
import math
import pandas as pd
import torch
import random, os
import numpy as np
import torch
from graphnet.data.constants import FEATURES, TRUTH
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.strategies.ddp import DDPStrategy
from graphnet.data.sqlite.sqlite_dataset import SQLiteDataset
from lion_pytorch import Lion

from icecube_utils import (
    CancelAzimuthByPredictionTransform,
    ExpLRSchedulerPiece,
    FlipOverXYLineTransform,
    LinearLRSchedulerPiece,
    CosineLRSchedulerPiece,
    OneOfTransform,
    RotateAngleTransform,
    train_dynedge_blocks,
    train_dynedge_from_scratch,
    FlipTimeTransform,
    FlipCoordinateTransform,
    train_dynedge_simplex,
)

# example usage:
# python train_large.py --model-save-dir /weights/test --max-epochs 1 --size-multiplier 1.0 --batch-size 512 --accumulate-grad-batches 1 --seed 0 --weight-loss-by-inverse-n-pulses-log --max-n-pulses-strategy random --mode small --enable-augmentations --lr-onecycle-factors 1e-02 1 1e-02 --lr-schedule-type linear
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dict-path', type=Path, default=None)
    parser.add_argument('--model-save-dir', type=Path, default=None)
    parser.add_argument('--max-epochs', type=float, default=10)
    parser.add_argument('--size-multiplier', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--loss-weight-strategy', 
        type=str, 
        choices=['inverse_n_pulses_log', 'zenith_count'],
        default=None
    )
    parser.add_argument(
        '--max-n-pulses-strategy', 
        type=str, 
        choices=['clamp', 'random_sequential', 'random'], 
        default='clamp'
    )
    parser.add_argument('--n-blocks', type=int, default=None)
    parser.add_argument('--zero-new-block', action='store_true')
    parser.add_argument('--block-output-aggregation', type=str, choices=['mean', 'sum'], default='sum')
    parser.add_argument('--enable-augmentations', action='store_true')
    parser.add_argument('--lr-onecycle-factors', type=float, nargs=3, default=[1e-02, 1, 1e-02])
    parser.add_argument('--lr-schedule-type', type=str, default='linear', choices=['linear', 'exp', 'cos'])
    parser.add_argument('--train-mode', type=str, default='default', choices=['default', 'block', 'simplex'])
    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--dropout', type=float, required=False, default=None)
    parser.add_argument('--optim', type=str, default='adamw', choices=['sgd', 'adam', 'adamw', 'lion'])
    parser.add_argument(
        '--target', 
        type=str, 
        default='direction', 
        choices=[
            'direction', 
            'angles_sincos_euclidean', 
            's2', 
            'zenith'
        ]
    )
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')


def first_last_pulse_index_to_loss_weight(first_last_pulse_index):
    first, last = first_last_pulse_index[0][0], first_last_pulse_index[0][1]
    if first == last:
        return [[1.0]]
    return [[1 / np.log((last - first) + 2)]]


class WeightLossByZenithCount:
    def __init__(self, weights_info_tsv_path):
        weights_info = pd.read_csv(weights_info_tsv_path, sep='\t')
        self.weights = weights_info['weights'].values
        self.zenith_bins = weights_info['zenith_bins'].values

    def __call__(self, zenith):
        zenith = zenith[0][0]
        index = np.digitize(zenith, self.zenith_bins) - 1
        index = np.clip(index, 0, len(self.weights) - 1)
        return [[self.weights[index]]]


features = FEATURES.KAGGLE
truth = ['zenith', 'azimuth']

# fold_0_val.db is obtained batch_656.parquet
# train_path/*.parquet are assumed to be batches up to 655 (inclusive)
config = {
        'dataset_type': 'parallel_parquet',
        # 'dataset_type': 'sqlite',
        # Pathes for large machine
        'parallel_parquet': {
            'train_path': Path('/workspace/icecube/data/parquet/train'),
            'meta_path': Path('/workspace/icecube/data/parquet/train_meta'),
            'geometry_path': Path('/workspace/icecube/data/sensor_geometry.csv'),
        },
        "path": '/workspace/icecube/data/fold_0.db',
        "inference_database_path": '/workspace/icecube/data/fold_0_val.db',

        # # Pathes for small machine
        # 'parallel_parquet': {
        #     'train_path': Path('/workspace/data2/train'),
        #     'meta_path': Path('/workspace/data2/train_meta'),
        #     'geometry_path': Path('/workspace/icecube/data/dataset/sensor_geometry.csv'),
        # },
        # "path": '/workspace/data2/batch_14.db',
        # "inference_database_path": '/workspace/data2/batch_656.db',

        "pulsemap": 'pulse_table',
        "truth_table": 'meta_table',
        "features": features,
        "truth": truth,
        # "tasks_weights": [0.75, 0.25],
        "index_column": 'event_id',
        "run_name_tag": 'my_example',
        "batch_size": 100,
        "num_workers": 10,
        "target": 'direction',
        "early_stopping_patience": 5,
        "fit": {
            "max_epochs": 10,
            "gpus": [0],
            "distribution_strategy": 'auto',
            "precision": '16-mixed', 
            "log_every_n_steps": 50,
            "val_check_interval": 0.05,  # originally was 0.5 on 10% of data, for parallel_parquet div by 10
            # "detect_anomaly": True,
            # "num_sanity_val_steps": 0,
            # "limit_train_batches": 100,
            # "limit_val_batches": 100,
            # "profiler": "simple",
            # "profiler": AdvancedProfiler(dirpath=".", filename="perf_logs"),
        },
        'base_dir': 'training',
        'dynedge': {},
        # applicable only for dataset_type == 'sqlite',
        # dataset_type == 'parquet' always shuffles data
        # and moreover order is not really deterministic
        'shuffle_train': True,
        'optimizer_class': torch.optim.AdamW,
        'optimizer_kwargs': {
            "lr": 1e-03, 
            "eps": 1e-03
        },
        "scheduler_kwargs": {
            "pieces": [
                LinearLRSchedulerPiece(1e-2, 1),
                LinearLRSchedulerPiece(1, 1e-2),
            ]
        },
        'max_n_pulses': {
            'max_n_pulses': 200,
            'max_n_pulses_strategy': 'clamp'
        },
        'loss_weight': {},
        'block': {
            'zero_new': False,
            'output_aggregation': 'sum',
            'block_size_scale': 0.5,
        },
        'train_transforms': [],
        'val_transforms': [],
        'simplex': {
            'n_verts': 3,
            'LMBD': 1e-10,
            'nsample': 5,
            'infrerence_sampling_average': 'angles',
            'infrerence_sampling_topk': None
        },
        'dynedge': {
            'bias': True,
            'bn': True,
            'dropout': None,
        },
        'model_kwargs': {}
}


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)

    if args.verbose:
        config['model_kwargs']['log_norm_verbose'] = True

    config['target'] = args.target

    if args.optim == 'sgd':
        config['optimizer_class'] = torch.optim.SGD
        config['optimizer_kwargs'] = {
            "lr": 1e-03, 
        }
    elif args.optim == 'adam':
        config['optimizer_class'] = torch.optim.Adam
        config['optimizer_kwargs'] = {
            "lr": 1e-03, 
        }
    elif args.optim == 'adamw':
        config['optimizer_class'] = torch.optim.AdamW
        config['optimizer_kwargs'] = {
            "lr": 1e-03, 
            "eps": 1e-03, 
        }
    elif args.optim == 'lion':
        config['optimizer_class'] = Lion
        config['optimizer_kwargs'] = {
            "lr": 1e-03, 
            "use_triton": False
        }

    config['train_mode'] = args.train_mode
    config['batch_size'] = args.batch_size
    config['dynedge']['dynedge_layer_sizes'] = [
        (int(x * args.size_multiplier), int(y * args.size_multiplier)) 
        for x, y in [(128, 256), (336, 256), (336, 256), (336, 256)]
    ]
    config['dynedge']['bn'] = args.bn
    config['dynedge']['dropout'] = args.dropout
    config['max_n_pulses']['max_n_pulses_strategy'] = args.max_n_pulses_strategy

    # Convert patience from epochs to validation checks
    config['early_stopping_patience'] = int(
        config['early_stopping_patience'] / 
        config['fit']['val_check_interval']
    )

    # Replace max_epochs with iterating files max_epochs times
    # to apply files shuffilg for parallel_parquet for each epoch
    # and change related parameters accordingly
    # TODO: fix it breaks limit_train_batches
    if config['dataset_type'] == 'parallel_parquet':
        config['fit']['max_epochs'] = 1

        if args.train_mode == 'simplex':  # Simplex run on limited subset of train
            filepathes = sorted(
                [
                    filepath
                    for filepath in config['parallel_parquet']['train_path'].glob('**/*.parquet')
                    if int(filepath.stem.split('_')[1]) < 132
                ]
            )
            config['fit']['val_check_interval'] = 1
        else:
            filepathes = sorted(
                list(
                    config['parallel_parquet']['train_path'].glob('**/*.parquet')
                )
            )
        config['parallel_parquet']['filepathes'] = []

        for i in range(math.floor(args.max_epochs)):
            random.shuffle(filepathes)
            config['parallel_parquet']['filepathes'] += filepathes[:]
        # If max_epochs is float
        if args.max_epochs - math.floor(args.max_epochs) > 0:
            random.shuffle(filepathes)
            n_files_in_fractional_epoch = math.ceil(
                len(filepathes) * (args.max_epochs - math.floor(args.max_epochs))
            )
            config['parallel_parquet']['filepathes'] += filepathes[:n_files_in_fractional_epoch]

        config['fit']['val_check_interval'] = config['fit']['val_check_interval'] / args.max_epochs

        config['parallel_parquet']['actual_max_epochs'] = args.max_epochs
        # originally was 0.5 on 10% of data, for parallel_parquet div by 10
        config['parallel_parquet']['warmup_epochs'] = 0.05

    config['fit']['gradient_clip_val'] = 100 * args.size_multiplier ** 2
    config['fit']['gradient_clip_algorithm'] = 'norm'

    # Set LR schedule
    if args.lr_schedule_type == 'linear':
        config['scheduler_kwargs']['pieces'] = [
            LinearLRSchedulerPiece(args.lr_onecycle_factors[0], args.lr_onecycle_factors[1]),
            LinearLRSchedulerPiece(args.lr_onecycle_factors[1], args.lr_onecycle_factors[2]),
        ]
    elif args.lr_schedule_type == 'exp':
        config['scheduler_kwargs']['pieces'] = [
            LinearLRSchedulerPiece(args.lr_onecycle_factors[0], args.lr_onecycle_factors[1]),
            ExpLRSchedulerPiece(args.lr_onecycle_factors[1], args.lr_onecycle_factors[2], decay=0.2),
        ]
    elif args.lr_schedule_type == 'cos':
        config['scheduler_kwargs']['pieces'] = [
            LinearLRSchedulerPiece(args.lr_onecycle_factors[0], args.lr_onecycle_factors[1]),
            CosineLRSchedulerPiece(args.lr_onecycle_factors[1], args.lr_onecycle_factors[2]),
        ]

    if args.loss_weight_strategy == 'inverse_n_pulses_log':
        config['loss_weight'] = {
            'loss_weight_table': 'meta_table',
            'loss_weight_columns': ['first_pulse_index', 'last_pulse_index'],
            'loss_weight_transform': first_last_pulse_index_to_loss_weight,
        }
    elif args.loss_weight_strategy == 'zenith_count':
        config['loss_weight'] = {
            'loss_weight_table': 'meta_table',
            'loss_weight_columns': ['zenith'],
            'loss_weight_transform': WeightLossByZenithCount('./weights_info.tsv'),
        }

    if args.enable_augmentations:
        if config['target'] in ['zenith_sincos_euclidean_cancel_azimuth', 'zenith']:
            config['train_transforms'] = [
                FlipCoordinateTransform(features=features, p=0.5, coordinate='x'),
                FlipCoordinateTransform(features=features, p=0.5, coordinate='y'),
                FlipCoordinateTransform(features=features, p=0.5, coordinate='z'),
                # TODO: make a proper rotation for zenith for transforms
                # RotateAngleTransform(features=features, p=0.5, angle='zenith'),
            ]
        else:
            config['train_transforms'] = [
                FlipOverXYLineTransform(features=features, p=0.5, k=-1.2334245570477371, b=22.436265094233192),
            ]

    # TODO: make a proper rotation for zenith: 
    # - limit rotation angle to some sensible range
    # - rotate in world coordinates like now
    # - rotate back in local coordinates of each vertical array of sensors in (rotated) x, y plane
    # - scale x and y by cos(rotation angle) -- that is why we limit the rotation angle
    # Reason: sensors geometry is tilted when rotating in world coordinates
    if config['target'] == 'zenith_sincos_euclidean_cancel_azimuth':
        config['truth'] = config['truth'] + ['azimuth_pred']
        config['train_transforms'].insert(
            0, 
            CancelAzimuthByPredictionTransform(
                features=features, 
                gt=True
            )
        )
        config['val_transforms'].insert(
            0, 
            CancelAzimuthByPredictionTransform(
                features=features, 
                gt=False
            )
        )

    # Zero new block after adding
    config['zero_new_block'] = args.zero_new_block
    config['block_output_aggregation'] = args.block_output_aggregation

    wandb_logger = WandbLogger(
        project='icecube',
        save_dir='./wandb',
        log_model=False,
    )
    wandb_logger.experiment.config.update(config, allow_val_change=True)
    config['fit']['logger'] = wandb_logger

    # Continue traininig: set trainer state
    # only if WANDB_RESUME env var is set to 'must'
    # otherwise, the model will be initialized with checkpoint,
    # but trainer state will be reset
    if (
        args.state_dict_path is not None and 
        args.state_dict_path.suffix == '.ckpt' and 
        'WANDB_RESUME' in os.environ and 
        os.environ['WANDB_RESUME'] == 'must'
    ):
        config['fit']['ckpt_path'] = args.state_dict_path
    
    if args.train_mode == 'block':
        assert args.n_blocks is not None
        
        # Gradient clipping is model kwargs due to manual optimization
        config['model_kwargs']['gradient_clip_val'] = config['fit']['gradient_clip_val']
        config['model_kwargs']['gradient_clip_algorithm'] = config['fit']['gradient_clip_algorithm']
        del config['fit']['gradient_clip_val'], config['fit']['gradient_clip_algorithm']
    
        config['fit']['distribution_strategy'] = 'auto'
        model = train_dynedge_blocks(
            config, 
            args.n_blocks, 
            args.model_save_dir,
            args.state_dict_path
        )
    elif args.train_mode == 'simplex':
        assert args.state_dict_path is not None
        model = train_dynedge_simplex(
            config, 
            args.state_dict_path
        )
    else:
        model = train_dynedge_from_scratch(
            config=config, 
            state_dict_path=None if args.state_dict_path is None else str(args.state_dict_path)
        )

    model.save_state_dict(str(args.model_save_dir / 'state_dict.pth'))

import argparse
import torch
import random, os
import numpy as np
import torch
from graphnet.data.constants import FEATURES, TRUTH
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.strategies.ddp import DDPStrategy
from graphnet.data.sqlite.sqlite_dataset import SQLiteDataset, SQLiteDatasetMaxNPulses

from icecube_utils import (
    CancelAzimuthByPredictionTransform,
    OneOfTransform,
    RotateAngleTransform,
    train_dynedge_blocks,
    train_dynedge_from_scratch,
    FlipTimeTransform,
    FlipCoordinateTransform
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dict-path', type=Path, default=None)
    parser.add_argument('--model-save-dir', type=Path, default=None)
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--size-multiplier', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--accumulate-grad-batches', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--weight-loss-by-inverse-n-pulses-log', action='store_true')
    parser.add_argument(
        '--max-n-pulses-strategy', 
        type=str, 
        choices=['clamp', 'random_sequential', 'random'], 
        default='clamp'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['small', 'large', 'large_contd'], 
        default='small'
    )
    parser.add_argument('--n-blocks', type=int, default=None)
    parser.add_argument('--zero-new-block', action='store_true')
    parser.add_argument('--block-output-aggregation', type=str, choices=['mean', 'sum'], default='sum')
    parser.add_argument('--enable-augmentations', action='store_true')
    parser.add_argument('--lr-onecycle-factors', type=float, nargs=3, default=[1e-02, 1, 1e-02])
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
    return [[1 / np.log((last - first) + 1)]]


features = FEATURES.KAGGLE
truth = ['zenith', 'azimuth']

config = {
        # "path": '/workspace/icecube/data/batch_1.db',
        # "inference_database_path": '/workspace/icecube/data/batch_51.db',
        "path": '/workspace/icecube/data/fold_0.db',
        "inference_database_path": '/workspace/icecube/data/fold_0_val.db',
        "pulsemap": 'pulse_table',
        "truth_table": 'meta_table',
        "features": features,
        "truth": truth,
        # "tasks_weights": [0.75, 0.25],
        "index_column": 'event_id',
        "run_name_tag": 'my_example',
        "batch_size": 100,
        "accumulate_grad_batches": 1,
        "num_workers": 10,
        "target": 'zenith',
        # "target": 'angles_sincos_euclidean',
        # "target": 'direction',
        "early_stopping_patience": 5,
        "fit": {
            "max_epochs": 10,
            "gpus": [0],
            "distribution_strategy": DDPStrategy(find_unused_parameters=False),
            "precision": '16-mixed', 
            "log_every_n_steps": 50,
            "val_check_interval": 0.2,
            # "limit_train_batches": 100,
            # "limit_val_batches": 100,
            # "profiler": "simple",
            # "profiler": AdvancedProfiler(dirpath=".", filename="perf_logs"),
        },
        'base_dir': 'training',
        'bias': True,
        'dynedge': {},
        'shuffle_train': True,
        'optimizer_kwargs': {
            "lr": 1e-03, 
            "eps": 1e-03
        },
        "scheduler_kwargs": {
            "factors": [1e-02, 1, 1e-02],
        },
        'max_n_pulses': {
            'max_n_pulses': 200,
            'max_n_pulses_strategy': 'clamp'
        },
        'loss_weight': {
            'loss_weight_table': 'meta_table',
            'loss_weight_columns': ['first_pulse_index', 'last_pulse_index'],
            'loss_weight_transform': first_last_pulse_index_to_loss_weight,
        },
        'dataset_class': SQLiteDataset,
        # 'dataset_class': SQLiteDatasetMaxNPulses,
        'zero_new_block': False,
        'block_output_aggregation': 'sum',
        'train_transforms': [],
        'val_transforms': [],
}


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)

    config['fit']['max_epochs'] = args.max_epochs
    config['batch_size'] = args.batch_size
    config['accumulate_grad_batches'] = args.accumulate_grad_batches
    config['dynedge']['dynedge_layer_sizes'] = [
        (x * args.size_multiplier, y * args.size_multiplier) 
        for x, y in [(128, 256), (336, 256), (336, 256), (336, 256)]
    ]
    config['max_n_pulses']['max_n_pulses_strategy'] = args.max_n_pulses_strategy

    if args.mode == 'large':
        config['fit']['val_check_interval'] = 0.1
        config['fit']['max_steps'] = 10000
    elif args.mode == 'large_contd':
        config['fit']['val_check_interval'] = 0.1
        config['fit']['max_steps'] = -1
    elif args.mode == 'small':
        config['fit']['val_check_interval'] = 0.5
        config['fit']['max_steps'] = -1
    else:
        raise ValueError(f'Unknown mode {args.mode}')

    # Set LR one-cycle factors
    config['scheduler_kwargs']['factors'] = args.lr_onecycle_factors
    
    # Convert patience from epochs to validation checks
    config['early_stopping_patience'] = int(
        config['early_stopping_patience'] / 
        config['fit']['val_check_interval']
    )

    if args.weight_loss_by_inverse_n_pulses_log:
        config['loss_weight'] = {
            'loss_weight_table': 'meta_table',
            'loss_weight_columns': ['first_pulse_index', 'last_pulse_index'],
            'loss_weight_transform': first_last_pulse_index_to_loss_weight,
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
                FlipCoordinateTransform(features=features, p=0.5, coordinate='x'),
                FlipCoordinateTransform(features=features, p=0.5, coordinate='y'),
                FlipCoordinateTransform(features=features, p=0.5, coordinate='z'),
                RotateAngleTransform(features=features, p=0.5, angle='azimuth'),
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
    
    if args.n_blocks is not None:
        config['fit']['distribution_strategy'] = 'auto'
        model = train_dynedge_blocks(
            config, 
            args.n_blocks, 
            args.model_save_dir,
            args.state_dict_path
        )
    else:
        model = train_dynedge_from_scratch(
            config=config, 
            state_dict_path=None if args.state_dict_path is None else str(args.state_dict_path)
        )

    model.save_state_dict(str(args.model_save_dir / 'state_dict.pth'))

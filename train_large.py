import argparse
import torch
import random, os
import numpy as np
import torch
from graphnet.data.constants import FEATURES, TRUTH
from pathlib import Path

from icecube_utils import (
    train_dynedge_from_scratch
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


features = FEATURES.KAGGLE
truth = TRUTH.KAGGLE

config = {
        "path": '/workspace/icecube/data/batch_1.db',
        "inference_database_path": '/workspace/icecube/data/batch_51.db',
        "pulsemap": 'pulse_table',
        "truth_table": 'meta_table',
        "features": features,
        "truth": truth,
        "index_column": 'event_id',
        "run_name_tag": 'my_example',
        "batch_size": 100,
        "accumulate_grad_batches": 1,
        "num_workers": 6,
        "target": 'direction',
        "early_stopping_patience": 5,
        "fit": {
            "max_epochs": 10,
            "gpus": [0],
            "distribution_strategy": None,
            "precision": 16, 
        },
        'train_selection': '/workspace/icecube/data/train_selection_max_200_pulses.csv',
        'validate_selection': '/workspace/icecube/data/validate_selection_max_200_pulses.csv',
        'test_selection': None,
        'base_dir': 'training',
        'bias': False,
        'dynedge': {
            'max_pulses': 200,
        }
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
    model = train_dynedge_from_scratch(
        config=config, 
        state_dict_path=None if args.state_dict_path is None else str(args.state_dict_path)
    )

    model.save(args.model_save_dir / 'model.pth')
    model.save_state_dict(args.model_save_dir / 'state_dict.pth')

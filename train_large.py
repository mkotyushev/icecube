import argparse
import torch
import random, os
import numpy as np
import torch
from graphnet.data.constants import FEATURES, TRUTH

from icecube_utils import (
    train_dynedge_from_scratch
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type=str, choices=['large', 'small'])
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
        "num_workers": 6,
        "target": 'direction',
        "early_stopping_patience": 5,
        "fit": {
            "max_epochs": 10,
            "gpus": [0],
            "distribution_strategy": None,
        },
        'train_selection': '/workspace/icecube/data/train_selection_max_200_pulses.csv',
        'validate_selection': '/workspace/icecube/data/validate_selection_max_200_pulses.csv',
        'test_selection': None,
        'base_dir': 'training',
        'dynedge': {}
}


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)

    if args.model_type == 'large':
        config['dynedge']['dynedge_layer_sizes'] = [(128 * 2, 256 * 2), (336 * 2, 256 * 2), (336 * 2, 256 * 2), (336 * 2, 256 * 2)]
    elif args.model_type == 'small':
        config['dynedge']['dynedge_layer_sizes'] = [(128, 256), (336, 256), (336, 256), (336, 256)]
    model = train_dynedge_from_scratch(config=config)

    model.save(f'weights/dynedge_pretrained_{args.model_type}_{args.seed}/model.pth')
    model.save_state_dict(f'weights/dynedge_pretrained_{args.model_type}_{args.seed}/state_dict.pth')

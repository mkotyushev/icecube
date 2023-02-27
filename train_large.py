import torch
import random, os
import numpy as np
import torch
from graphnet.data.constants import FEATURES, TRUTH

from icecube_utils import (
    train_dynedge_from_scratch
)

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
        "path": '/workspaces/icecube3/batch_1.db',
        "inference_database_path": '/workspaces/icecube3/batch_51.db',
        "pulsemap": 'pulse_table',
        "truth_table": 'meta_table',
        "features": features,
        "truth": truth,
        "index_column": 'event_id',
        "run_name_tag": 'my_example',
        "batch_size": 200,
        "num_workers": 6,
        "target": 'direction',
        "early_stopping_patience": 5,
        "fit": {
            "max_epochs": 1,
            # 'max_steps': 0,
            "gpus": [0],
            "distribution_strategy": None,
        },
        'train_selection': '/workspaces/icecube3/train_selection_max_200_pulses.csv',
        'validate_selection': '/workspaces/icecube3/validate_selection_max_200_pulses.csv',
        'test_selection': None,
        'base_dir': 'training',
        'dynedge': {
                'dynedge_layer_sizes': [(128 * 2, 256 * 2), (336 * 2, 256 * 2), (336 * 2, 256 * 2), (336 * 2, 256 * 2)]
        }
}


if __name__ == '__main__':
    seed_everything(0)
    model = train_dynedge_from_scratch(config=config)

    model.save('dynedge_pretrained_large/model.pth')
    model.save_state_dict('dynedge_pretrained_large/state_dict.pth')

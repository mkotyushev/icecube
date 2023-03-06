from pathlib import Path
from copy import deepcopy
from graphnet.data.constants import FEATURES, TRUTH
from icecube_utils import make_dataloader

from icecube_utils import (
    load_pretrained_model,
    make_dataloader,
    map_model,
)
from graphnet.training.labels import Direction
from parameters import get_parser
from train_large import config

features = FEATURES.KAGGLE
truth = TRUTH.KAGGLE


def get_args():
    parser = get_parser()
    parser.add_argument('--size-multiplier', type=int, default=1)
    parser.add_argument('--from-model-state-dict-path', type=Path, required=True)
    parser.add_argument('--to-model-state-dict-path', type=Path, required=True)
    parser.add_argument('--mapped-model-save-dir', type=Path, required=True)
    args = parser.parse_args()

    args.gpu_id = 0
    args.proper_marginals = True
    args.skip_last_layer = True
    args.skip_personal_idx = False
    args.act_num_samples = 20
    args.width_ratio = 1
    args.dataset = 'icecube'

    return args


def main(args):
    models = {}

    config_from = deepcopy(config)
    config_from['dynedge']['dynedge_layer_sizes'] = [(128, 256), (336, 256), (336, 256), (336, 256)]

    models['from'] = load_pretrained_model(
        config=config_from, 
        state_dict_path=str(args.from_model_state_dict_path)
    )

    config_to = deepcopy(config)
    config_to['dynedge']['dynedge_layer_sizes'] = [
        (x * args.size_multiplier, y * args.size_multiplier) 
        for x, y in [(128, 256), (336, 256), (336, 256), (336, 256)]
    ]

    models['to'] = load_pretrained_model(
        config=config_to, 
        state_dict_path=str(args.to_model_state_dict_path)
    )

    # Map from model to to model
    train_dataloader = make_dataloader(db=config_from['path'],
        selection=None, # Entire database
        pulsemaps=config_from['pulsemap'],
        features=features,
        truth=truth,
        batch_size=args.act_num_samples,
        num_workers=config_from['num_workers'],
        shuffle=False,
        labels={'direction': Direction()},
        index_column=config_from['index_column'],
        truth_table=config_from['truth_table'],
    )

    models['mapped'] = map_model(
        args,
        models['from'], 
        models['to'], 
        train_dataloader, 
        args.act_num_samples
    )
    models['mapped'].save_state_dict(str(args.mapped_model_save_dir / 'state_dict.pth'))


if __name__ == '__main__':
    args = get_args()
    main(args)

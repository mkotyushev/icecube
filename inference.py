from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from copy import deepcopy
import pandas as pd
import torch
from graphnet.data.constants import FEATURES, TRUTH

from icecube_utils import (
    inference, 
    convert_to_3d,
    calculate_angular_error,
    load_pretrained_model
)
from parameters import get_parser
from train_large import config as base_config

features = FEATURES.KAGGLE
truth = TRUTH.KAGGLE


def get_args():
    parser = get_parser()
    parser.add_argument('--pdf-save-path', type=str, default='mapping_results.pdf')
    parser.add_argument('--size-multiplier', type=int, default=1)
    parser.add_argument('--from-model-state-dict-path', type=str, required=True)
    parser.add_argument('--to-model-state-dict-path', type=str, required=True)
    parser.add_argument('--mapped-model-state-dict-path', type=str, required=True)
    args = parser.parse_args()

    args.gpu_id = 0
    args.proper_marginals = True
    args.skip_last_layer = True
    args.skip_personal_idx = False
    args.act_num_samples = 20
    args.width_ratio = 1
    args.dataset = 'icecube'

    return args


# class InferenceModel(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self._detector = model._detector
#         self._gnn = model._gnn
#         self._task = model._task

#     def forward(self, data):
#         return self.model(data, data.x, data.edge_index, data.batch


def main(args):
    results = {}

    model_name_to_state_dict_paths = {
        'from': args.from_model_state_dict_path,
        'to': args.to_model_state_dict_path,
        'mapped': args.mapped_model_state_dict_path,
    }

    for model_name, state_dict_path in model_name_to_state_dict_paths.items():
        print(f'Running {model_name} model')
        if Path(f'results/{model_name}.h5').exists():
            results[model_name] = pd.read_hdf(f'results/{model_name}.h5', key='df')
        else:
            config = deepcopy(base_config)
            if model_name in {'to', 'mapped'}:
                config['dynedge']['dynedge_layer_sizes'] = [
                    (x * args.size_multiplier, y * args.size_multiplier) 
                    for x, y in [(128, 256), (336, 256), (336, 256), (336, 256)]
                ]
                config['batch_size'] = 200

            model, _ = load_pretrained_model(
                config=config, 
                state_dict_path=state_dict_path,
                return_train_dataloader=True,
            )

            results[model_name] = calculate_angular_error(
                convert_to_3d(
                    inference(
                        model.cuda(), 
                        config
                    )
                )
            )
            results[model_name].to_hdf(f'results/{model_name}.h5', key='df', mode='w')
            del model
            torch.cuda.empty_cache()

    # Plot results
    with PdfPages(args.pdf_save_path) as pdf:
        fig = plt.figure(figsize = (6,6))
        for model_name, result in results.items():
            plt.hist(result['angular_error'], 
                bins = np.arange(0,np.pi*2, 0.05), 
                histtype = 'step', 
                label = f'{model_name} mean AE: {np.round(result["angular_error"].mean(),2)}'
            )
            plt.xlabel('Angular Error [rad.]', size = 15)
            plt.ylabel('Counts', size = 15)
        plt.title(f'{model_name}, Angular Error Distribution (Batch 51)', size = 15)
        plt.legend(frameon=False, fontsize=15)
        pdf.savefig(fig)


if __name__ == '__main__':
    args = get_args()
    main(args)

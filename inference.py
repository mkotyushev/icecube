import argparse
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
    inference_simplex,
    load_pretrained_model
)
from train_large import config as base_config

features = FEATURES.KAGGLE
truth = TRUTH.KAGGLE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdf_save_path', type=str, default='mapping_results.pdf')
    parser.add_argument('size_multiplier', type=float, default=1.0)
    parser.add_argument('from_model_state_dict_path', type=str)
    parser.add_argument('to_model_state_dict_path', type=str)
    parser.add_argument('mapped_model_state_dict_path', type=str)
    parser.add_argument('--simplex_model_state_dict_path', type=str, required=False, default=None)
    args = parser.parse_args()

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
        'simplex': args.simplex_model_state_dict_path,
    }

    base_config['dataset_type'] = 'sqlite'

    for model_name, state_dict_path in model_name_to_state_dict_paths.items():
        print(f'Running {model_name} model')
        if Path(f'results/{model_name}.h5').exists():
            results[model_name] = pd.read_hdf(f'results/{model_name}.h5', key='df')
        else:
            config = deepcopy(base_config)
            if model_name in {'to', 'mapped'}:
                config['dynedge']['dynedge_layer_sizes'] = [
                    (int(x * args.size_multiplier), int(y * args.size_multiplier)) 
                    for x, y in [(128, 256), (336, 256), (336, 256), (336, 256)]
                ]
            infrence_fn = inference
            if model_name == 'simplex' and state_dict_path is not None:
                config['train_mode'] = 'simplex_inference'
                config['simplex']['nsample'] = 20
                config['simplex']['infrerence_sampling_average'] = 'direction'
                infrence_fn = inference_simplex

            config['batch_size'] = 512

            model = load_pretrained_model(
                config=config, 
                path=state_dict_path,
                return_train_dataloader=False,
            )

            results[model_name] = calculate_angular_error(
                convert_to_3d(
                    infrence_fn(
                        model.cuda(), 
                        config,
                        True
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
                label = f'{model_name} mean AE: {result["angular_error"].mean():.3f}'
            )
            plt.xlabel('Angular Error [rad.]', size = 15)
            plt.ylabel('Counts', size = 15)
        plt.title(f'{model_name}, Angular Error Distribution (Batch 51)', size = 15)
        plt.legend(frameon=False, fontsize=15)
        pdf.savefig(fig)


if __name__ == '__main__':
    args = parse_args()
    main(args)

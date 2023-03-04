import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_pdf import PdfPages
from copy import deepcopy
from graphnet.data.constants import FEATURES, TRUTH
from icecube_utils import make_dataloader

from icecube_utils import (
    load_pretrained_model,
    inference, 
    make_dataloader,
    map_model,
    convert_to_3d,
    calculate_angular_error
)
from graphnet.training.labels import Direction
from parameters import get_parser
from train_large import config

features = FEATURES.KAGGLE
truth = TRUTH.KAGGLE


def get_args():
    parser = get_parser()
    parser.add_argument('--pdf-save-path', type=str, default='mapping_results.pdf')
    args = parser.parse_args('--gpu-id 1 --model-name mlpnet --n-epochs 10 --save-result-file sample.csv \
    --sweep-name exp_sample --exact --correction --ground-metric euclidean --weight-stats \
    --activation-histograms --activation-mode raw --geom-ensemble-type acts --sweep-id 21 \
    --act-num-samples 200 --ground-metric-normalize none --activation-seed 21 \
    --prelu-acts --recheck-acc --load-models ./mnist_models --ckpt-type final \
    --past-correction --not-squared --dist-normalize --print-distances --to-download'.split())

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

    # Load pretrained models
    config_small = deepcopy(config)
    config_small['dynedge']['dynedge_layer_sizes'] = [(128, 256), (336, 256), (336, 256), (336, 256)]

    models['small'] = load_pretrained_model(
        config=config_small, 
        state_dict_path='/workspace/icecube/weights/dynedge_pretrained_batch_1_to_50/state_dict.pth'
    )

    config_large = deepcopy(config)
    config_large['dynedge']['dynedge_layer_sizes'] = [(128 * 2, 256 * 2), (336 * 2, 256 * 2), (336 * 2, 256 * 2), (336 * 2, 256 * 2)]

    models['large_bad'] = load_pretrained_model(
        config=config_large, 
        state_dict_path='/workspace/icecube/weights/dynedge_pretrained_large_0_epochs_1/state_dict.pth'
    )

    # Map small model to large model
    train_dataloader = make_dataloader(db=config_small['path'],
        selection=None, # Entire database
        pulsemaps=config_small['pulsemap'],
        features=features,
        truth=truth,
        batch_size=args.act_num_samples,
        num_workers=config_small['num_workers'],
        shuffle=False,
        labels={'direction': Direction()},
        index_column=config_small['index_column'],
        truth_table=config_small['truth_table'],
    )

    models['small_mapped'] = map_model(
        args,
        models['small'], 
        models['large_bad'], 
        train_dataloader, 
        args.act_num_samples
    )
    torch.cuda.empty_cache()

    # Inference
    results = {}
    for model_name, model in models.items():
        results[model_name] = calculate_angular_error(
            convert_to_3d(
                inference(
                    model, 
                    config_large
                )
            )
        )
        torch.cuda.empty_cache()

    # Plot results
    with PdfPages(args.pdf_save_path) as pdf:
        fig = plt.figure(figsize = (6,6))
        for name, result in results.items():
            plt.hist(result['angular_error'], 
                    bins = np.arange(0,np.pi*2, 0.05), 
                    histtype = 'step', 
                    label = f'{name} mean AE: {np.round(result["angular_error"].mean(),2)}')
            plt.xlabel('Angular Error [rad.]', size = 15)
            plt.ylabel('Counts', size = 15)
        plt.title(f'{name}, Angular Error Distribution (Batch 51)', size = 15)
        plt.legend(frameon=False, fontsize=15)
        pdf.savefig(fig)


if __name__ == '__main__':
    args = get_args()
    main(args)

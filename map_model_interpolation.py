import argparse
import torch
from pathlib import Path
from tqdm import tqdm

from graphnet.utilities.logging import get_logger
from train_large import config
from icecube_utils import build_model, load_pretrained_model, make_dataloaders


logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('size_multiplier', type=float, default=2.0)
    parser.add_argument('from_model_state_dict_path', type=Path)
    parser.add_argument('mapped_model_save_dir', type=Path)
    return parser.parse_args()


def interpolate_linear_layer(
    linear_layer: torch.nn.Linear, 
    in_features_new: int, 
    out_features_new: int, 
    mode: str = 'bilinear'
) -> torch.nn.Linear:
    """
    Performs bilinear interpolation on the weights of a PyTorch Linear layer to obtain a new Linear layer
    with mapped input and output sizes.

    Args:
        linear_layer (nn.Linear): PyTorch Linear layer whose weights are to be interpolated.
        in_features_new (int): New input size for the Linear layer.
        out_features_new (int): New output size for the Linear layer.
        mode (str): Interpolation mode. Defaults to 'bilinear'.

    Returns:
        nn.Linear: New Linear layer with mapped input and output sizes.
    """
    # Extract weight and bias from the original linear layer
    weight = linear_layer.weight
    bias = linear_layer.bias

    # Extract input and output sizes from the original linear layer
    in_features = weight.size(1)
    out_features = weight.size(0)

    # Perform bilinear interpolation for weight
    weight_interpolated = torch.nn.functional.interpolate(
        weight.view(1, 1, out_features, in_features),
        size=(out_features_new, in_features_new),
        mode=mode,
        align_corners=False
    ).view(out_features_new, in_features_new)

    # Scale the interpolated weight
    scale = (weight_interpolated.norm() /  weight.norm()) ** 0.5
    weight_interpolated = scale * weight_interpolated

    # Perform linear interpolation for bias
    if bias is not None:
        bias_interpolated = torch.nn.functional.interpolate(
            bias.view(1, 1, 1, -1),
            size=(1, out_features_new),
            mode=mode,
            align_corners=False
        ).view(out_features_new)

        # Scale the interpolated bias
        scale = (bias_interpolated.norm() /  bias.norm()) ** 0.5
        bias_interpolated = bias_interpolated * scale
    else:
        bias_interpolated = None

    # Create a new Linear layer with interpolated weight and bias
    linear_layer_mapped = torch.nn.Linear(
        in_features_new, 
        out_features_new, 
        bias=(bias_interpolated is not None)
    )
    linear_layer_mapped.weight.data.copy_(weight_interpolated)
    if bias_interpolated is not None:
        linear_layer_mapped.bias.data.copy_(bias_interpolated)

    return linear_layer_mapped


def interpolate_batchnorm_layer(batchnorm_layer, num_features_new):
    """
    Performs bilinear interpolation on the weights of a PyTorch BatchNorm layer to obtain a new BatchNorm layer
    with mapped input and output sizes.

    Args:
        batchnorm_layer (nn.BatchNorm1d): PyTorch BatchNorm layer whose weights are to be interpolated.
        num_features_new (int): New number of features for the BatchNorm layer.

    Returns:
        nn.BatchNorm1d: New BatchNorm layer with mapped input and output sizes.
    """
    params_to_interpolate, params_interpolated = {
        'weight': batchnorm_layer.weight,
        'bias': batchnorm_layer.bias,
        'running_mean': batchnorm_layer.running_mean,
        'running_var': batchnorm_layer.running_var
    }, {}
    for param_name, param in params_to_interpolate.items():
        num_features = param.size(0)

        # Perform linear interpolation for param
        param_interpolated = torch.nn.functional.interpolate(
            param.view(1, 1, num_features),
            size=(num_features_new,),
            mode='linear',
            align_corners=False
        ).view(num_features_new)

        # Scale the interpolated param
        scale = (param_interpolated.norm() /  param.norm()) ** 0.5
        param_interpolated = param_interpolated * scale

        params_interpolated[param_name] = param_interpolated

    # Create a new BatchNorm layer with interpolated weight and bias
    batchnorm_layer_mapped = torch.nn.BatchNorm1d(
        num_features=num_features_new
    )
    batchnorm_layer_mapped.weight.data.copy_(params_interpolated['weight'])
    batchnorm_layer_mapped.bias.data.copy_(params_interpolated['bias'])
    batchnorm_layer_mapped.running_mean.data.copy_(params_interpolated['running_mean'])
    batchnorm_layer_mapped.running_var.data.copy_(params_interpolated['running_var'])
    batchnorm_layer_mapped.num_batches_tracked.data.copy_(batchnorm_layer.num_batches_tracked)

    return batchnorm_layer_mapped


def main(args):
    config['dataset_type'] = 'sqlite'
    config['train_mode'] = 'default'

    # Model from
    model_from = load_pretrained_model(
        config=config, 
        path=str(args.from_model_state_dict_path)
    )
    for name, param in model_from.named_parameters():
        logger.info(f'model from {name}: {param.size()}')
    
    # Model to
    config['dynedge']['repeat_input'] = int(args.size_multiplier)
    config['dynedge']['dynedge_layer_sizes'] = [
        (int(x * args.size_multiplier), int(y * args.size_multiplier)) 
        for x, y in [(128, 256), (336, 256), (336, 256), (336, 256)]
    ]
    train_dataloader, _ = make_dataloaders(config)
    model_to = build_model(config, train_dataloader)
    for name, param in model_to.named_parameters():
        logger.info(f'model to {name}: {param.size()}')

    named_modules = list(model_from.named_modules())
    skip_layers_contains = {
        '_gnn._post_processing.1',
        '_gnn._readout.0',
        '_tasks.0._affine'
    }

    for name, module in tqdm(named_modules):
        # Some layers have same size for any multiplier
        if any((skip_name_contains in name) for skip_name_contains in skip_layers_contains):
            setattr(model_to, name, module)
            continue

        if isinstance(module, torch.nn.Linear):
            in_features_new = int(args.size_multiplier * module.in_features)
            out_features_new = int(args.size_multiplier * module.out_features)

            # Input layer has same input size
            if '_gnn._conv_layers.0.nn.0' in name:
                in_features_new = module.in_features
            # First postprocessing layer has same output size
            elif '_gnn._post_processing.0' in name:
                out_features_new = module.out_features

            module_mapped = interpolate_linear_layer(
                module, 
                in_features_new, 
                out_features_new
            )
            setattr(model_to, name, module_mapped)
        elif isinstance(module, torch.nn.BatchNorm1d):
            num_features = int(args.size_multiplier * module.num_features)
            if '_gnn._post_processing.0' in name:
                num_features = module.num_features

            module_mapped = interpolate_batchnorm_layer(
                module, 
                num_features
            )
            setattr(model_to, name, module_mapped)

    # Validate loading
    state_dict = model_to.state_dict()
    model_to.load_state_dict(state_dict, strict=True)

    # Print mapped model
    for name, param in model_to.named_parameters():
        logger.info(f'model mapped {name}: {param.size()}')

    # Save
    args.mapped_model_save_dir.mkdir(parents=True, exist_ok=True)
    model_to.save_state_dict(str(args.mapped_model_save_dir / 'state_dict.pth'))


if __name__ == '__main__':
    args = parse_args()
    main(args)

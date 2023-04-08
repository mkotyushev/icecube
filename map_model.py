import sys
import torch
from collections import OrderedDict
from pathlib import Path
from copy import deepcopy
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.parquet.parquet_dataset import ParquetDataset
from graphnet.data.sqlite.sqlite_dataset import SQLiteDataset
from icecube_utils import make_dataloader

from icecube_utils import (
    load_pretrained_model,
    make_dataloader
)
from graphnet.training.labels import Direction
from parameters import get_parser
from train_large import config


from ground_metric import GroundMetric

from compute_activations import (
    normalize_tensor,
    save_activations
)
from wasserstein_ensemble import (
    _check_activation_sizes,
    _compute_marginals, 
    _get_current_layer_transport_map, 
    _get_layer_weights, 
    _get_neuron_importance_histogram, 
    _get_updated_acts_v0, 
    _process_ground_metric_from_acts, 
    get_activation_distance_stats, 
    get_histogram,
    update_model
)

from graphnet.utilities.logging import get_logger

# Constants
logger = get_logger()

features = FEATURES.KAGGLE
truth = TRUTH.KAGGLE


def get_args(args=None):
    parser = get_parser()
    parser.add_argument('size_multiplier', type=float, default=1.0)
    parser.add_argument('from_model_state_dict_path', type=Path)
    parser.add_argument('to_model_state_dict_path', type=Path)
    parser.add_argument('mapped_model_save_dir', type=Path)
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    args.model_name = 'mlpnet'
    args.n_epochs = 10
    args.save_result_file = 'sample.csv'
    args.sweep_name = 'exp_sample'
    args.exact = True
    args.correction = True
    args.ground_metric = 'euclidean'
    args.weight_stats = True
    args.activation_histograms = True
    args.activation_mode = 'raw'
    args.geom_ensemble_type = 'acts'
    args.sweep_id = 21
    args.ground_metric_normalize = 'none'
    args.activation_seed = 21
    args.prelu_acts = True
    args.recheck_acc = True
    args.load_models = './mnist_models'
    args.ckpt_type = 'final'
    args.past_correction = True
    args.not_squared = True
    args.dist_normalize = True
    args.print_distances = True
    args.to_download = True

    args.gpu_id = 0
    args.proper_marginals = True
    args.skip_last_layer = False
    args.skip_personal_idx = False
    args.act_num_samples = 256
    args.width_ratio = 1
    args.dataset = 'icecube'
    args.disable_bias = False

    return args



def compute_activations_across_models_v1(args, models, train_loader, num_samples, mode='mean',
                                         dump_activations=False, dump_path=None, layer_names=None):

    torch.manual_seed(args.activation_seed)

    # hook that computes the mean activations across data samples
    def get_activation(activation, name):
        def hook(model, input, output):
            # logger.info("num of samples seen before", num_samples_processed)
            # logger.info("output is ", output.detach())
            if name not in activation:
                activation[name] = []

            if isinstance(output, tuple):
                activation[name].append(output[0].detach())
            else:
                activation[name].append(output.detach())

        return hook

    # Prepare all the models
    activations = {}
    forward_hooks = []

    # assert args.disable_bias
    # handle below for bias later on!
    # logger.info("list of model named params ", list(models[0].named_parameters()))
    for idx, model in enumerate(models):

        # Initialize the activation dictionary for each model
        activations[idx] = {}
        layer_hooks = []
        # Set forward hooks for all layers inside a model
        for name, layer in model.named_modules():
            if name not in layer_names:
                continue
            layer_hooks.append(layer.register_forward_hook(get_activation(activations[idx], name)))
            logger.info(f"set forward hook for layer named: {name}")

        forward_hooks.append(layer_hooks)
        # Set the model in train mode
        model.train()

    # Run the same data samples ('num_samples' many) across all the models
    num_samples_processed = 0
    num_personal_idx = 0
    for batch_idx, data in enumerate(train_loader):
        if num_samples_processed >= num_samples:
            break
        if args.gpu_id != -1:
            data = data.cuda(args.gpu_id)

        # if args.skip_personal_idx and int(target.item()) == args.personal_class_idx:
        #     continue

        # if int(target.item()) == args.personal_class_idx:
        #     num_personal_idx += 1

        for idx, model in enumerate(models):
            model(data)

        num_samples_processed += len(data)

    logger.info(f"num_personal_idx {num_personal_idx}")
    setattr(args, 'num_personal_idx', num_personal_idx)

    relu = torch.nn.LeakyReLU()

    # Combine the activations generated across the number of samples to form importance scores
    # The importance calculated is based on the 'mode' flag: which is either of 'mean', 'std', 'meanstd'

    # model_cfg = myutils.get_model_layers_cfg(args.model_name)
    for idx in range(len(models)):
        for lnum, layer in enumerate(activations[idx]):
            logger.info('***********')
            activations[idx][layer] = torch.concatenate(activations[idx][layer], dim=0)
            logger.info(
                "min of act: {}, max: {}, mean: {}".format(
                    torch.min(activations[idx][layer]), 
                    torch.max(activations[idx][layer]), 
                    torch.mean(activations[idx][layer])
                )
            )
            # assert (activations[idx][layer] >= 0).all()

            if not args.prelu_acts and not lnum == (len(activations[idx])-1):
                # logger.info("activation was ", activations[idx][layer])
                logger.info("applying relu ---------------")
                activations[idx][layer] = relu(activations[idx][layer])
                # logger.info("activation now ", activations[idx][layer])
                logger.info(
                    "after RELU: min of act: {}, max: {}, mean: {}".format(
                        torch.min(activations[idx][layer]),
                        torch.max(activations[idx][layer]),
                        torch.mean(activations[idx][layer])
                    )
                )
            elif args.model_name == 'vgg11_nobias' and args.pool_acts and len(activations[idx][layer].shape)>3:
                if args.pool_relu:
                    logger.info("applying relu ---------------")
                    activations[idx][layer] = relu(activations[idx][layer])

                activations[idx][layer] = activations[idx][layer].squeeze(1)

                # unsqueeze back at axis 1
                activations[idx][layer] = activations[idx][layer].unsqueeze(1)

                logger.info("checking stats after pooling")
                logger.info(
                    "min of act: {}, max: {}, mean: {}".format(
                        torch.min(activations[idx][layer]),
                        torch.max(activations[idx][layer]),
                        torch.mean(activations[idx][layer])
                    )
                )

            if mode == 'mean':
                activations[idx][layer] = activations[idx][layer].mean(dim=0)
            elif mode == 'std':
                activations[idx][layer] = activations[idx][layer].std(dim=0)
            elif mode == 'meanstd':
                activations[idx][layer] = activations[idx][layer].mean(dim=0) * activations[idx][layer].std(dim=0)

            if args.standardize_acts:
                mean_acts = activations[idx][layer].mean(dim=0)
                std_acts = activations[idx][layer].std(dim=0)
                logger.info(
                    "shape of mean, std, and usual acts are: {}, {}, {}".format(
                        mean_acts.shape, 
                        std_acts.shape, 
                        activations[idx][layer].shape
                    )
                )
                activations[idx][layer] = (activations[idx][layer] - mean_acts)/(std_acts + 1e-9)
            elif args.center_acts:
                mean_acts = activations[idx][layer].mean(dim=0)
                logger.info(
                    "shape of mean and usual acts are: {}, {}".format(
                        mean_acts.shape, 
                        activations[idx][layer].shape
                    )
                )
                activations[idx][layer] = (activations[idx][layer] - mean_acts)
            elif args.normalize_acts:
                logger.info("normalizing the activation vectors")
                activations[idx][layer] = normalize_tensor(activations[idx][layer])
                logger.info(
                    "min of act: {}, max: {}, mean: {}".format(
                        torch.min(activations[idx][layer]),
                        torch.max(activations[idx][layer]),
                        torch.mean(activations[idx][layer])
                    )
                )

            logger.info(
                "activations for idx {} at layer {} have the following shape {}".format(
                    idx, layer, activations[idx][layer].shape))
            logger.info('-----------')

    # Dump the activations for all models onto disk
    if dump_activations and dump_path is not None:
        for idx in range(len(models)):
            save_activations(idx, activations[idx], dump_path)

    # Remove the hooks (as this was intefering with prediction ensembling)
    for idx in range(len(forward_hooks)):
        for hook in forward_hooks[idx]:
            hook.remove()


    return activations


def process_activations(args, activations, layer0_name, layer1_name, has_bn=False):
    if has_bn:
        # If the model has BN layers
        # we need to use activation from BN output
        # for all the layers in linear block: linear itself and BN

        # It is assumed that each linear layer is followed by BN layer
        layer0_name = layer0_name.replace('.linear', '.bn')
        layer1_name = layer1_name.replace('.linear', '.bn')

        layer0_name = layer0_name \
            .replace('.running_mean', '.bias') \
            .replace('.running_var', '.bias')
        layer1_name = layer1_name \
            .replace('.running_mean', '.bias') \
            .replace('.running_var', '.bias')

    # Bias and weights use same activations
    layer0_name = layer0_name \
        .replace('.weight', '') \
        .replace('.bias', '')
    layer1_name = layer1_name \
        .replace('.weight', '') \
        .replace('.bias', '')

    activations_0 = activations[0][layer0_name].squeeze()
    activations_1 = activations[1][layer1_name].squeeze()

    # assert activations_0.shape == activations_1.shape
    _check_activation_sizes(args, activations_0, activations_1)

    if args.same_model != -1:
        # sanity check when averaging the same model (with value being the model index)
        assert (activations_0 == activations_1).all()
        logger.info("Are the activations the same? ", (activations_0 == activations_1).all())

    if len(activations_0.shape) == 2:
        activations_0 = activations_0.t()
        activations_1 = activations_1.t()
    elif len(activations_0.shape) > 2:
        reorder_dim = [l for l in range(1, len(activations_0.shape))]
        reorder_dim.append(0)
        logger.info(f"reorder_dim is {reorder_dim}")
        activations_0 = activations_0.permute(*reorder_dim).contiguous()
        activations_1 = activations_1.permute(*reorder_dim).contiguous()

    return activations_0, activations_1


def get_acts_wassersteinized_layers_modularized(
    args, 
    networks_named_params, 
    activations, 
    eps=1e-7, 
    test_loader=None,
    input_dim=17,
    has_bn=False,
):
    '''
    Average based on the activation vector over data samples. Obtain the transport map,
    and then based on which align the nodes and average the weights!
    Like before: two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.
    Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*
    :param networks: list of networks
    :param activations: If not None, use it to build the activation histograms.
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    '''

    avg_aligned_layers, aligned_layers, T_vars = OrderedDict(), OrderedDict(), OrderedDict()
    T_var = None

    previous_layer_shape = None
    num_layers = len(networks_named_params)
    ground_metric_object = GroundMetric(args)

    if args.update_acts or args.eval_aligned:
        model0_aligned_layers = []

    if args.gpu_id==-1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id))

    idx = 0
    incoming_layer_aligned = True # for input
    while idx < num_layers:
        ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) = networks_named_params[idx]
        logger.info("\n--------------- At layer index {} name {} ------------- \n ".format(idx, layer0_name))
        logger.info(f"Previous layer shape is {previous_layer_shape}, current layer shape is {fc_layer0_weight.shape}")
        previous_layer_shape = fc_layer1_weight.shape

        # BN affine params are handled the same way as bias
        is_bias = \
            ('running_mean' in layer0_name or 'running_var' in layer0_name) or \
            ('bn.weight' in layer0_name or 'bn.bias' in layer0_name) or \
            'bias' in layer0_name
        is_first_layer = (idx == 0) if args.disable_bias else (idx <= 1)
        is_last_layer = (idx == (num_layers - 1)) if args.disable_bias else (idx >= (num_layers - 2))

        # No need for bias to align input
        if is_bias:
            incoming_layer_aligned = True

        activations_0, activations_1 = process_activations(args, activations, layer0_name, layer1_name, has_bn=has_bn)

        assert activations_0.shape[0] == fc_layer0_weight.shape[0], \
            f'activations_0.shape[0] {activations_0.shape[0]} fc_layer0_weight.shape[0] {fc_layer0_weight.shape[0]}'
        assert activations_1.shape[0] == fc_layer1_weight.shape[0], \
            f'activations_1.shape[0] {activations_1.shape[0]} fc_layer1_weight.shape[0] {fc_layer1_weight.shape[0]}'

        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        torch.cuda.empty_cache()
        get_activation_distance_stats(activations_0, activations_1, layer0_name)

        layer0_shape = fc_layer0_weight.shape
        layer_shape = fc_layer1_weight.shape

        assert len(layer0_shape) == len(layer_shape)
        assert len(layer0_shape) == 2 or len(layer0_shape) == 1

        fc_layer0_weight_data = _get_layer_weights(fc_layer0_weight, is_conv=False)
        fc_layer1_weight_data = _get_layer_weights(fc_layer1_weight, is_conv=False)

        # Align input with previous layer
        if is_first_layer or incoming_layer_aligned:
            logger.info(f"is_first_layer or incoming_layer_aligned")
            aligned_wt = fc_layer0_weight_data
        else:
            logger.info(f"Input correction")
            logger.info(f"\tshape of layer: model 0 {fc_layer0_weight_data.shape}")
            logger.info(f"\tshape of layer: model 1 {fc_layer1_weight_data.shape}")
            logger.info(f"\tshape of activations: model 0 {activations_0.shape}")
            logger.info(f"\tshape of activations: model 1 {activations_1.shape}")
            logger.info(f"\tshape of previous transport map {T_var.shape}")
            logger.info(f"\tshape of fc_layer0_weight_data {fc_layer0_weight_data.shape}")
            logger.info(f"\tshape of T_var {T_var.shape}")

            if fc_layer0_weight_data.shape[1] != T_var.shape[0]:
                if fc_layer0_weight_data.shape[1] == 2 * T_var.shape[0]:
                    # Handles cat connections of DynEdgeConv nn's first layer: same T_var twise
                    temp_T_var = [T_var, T_var]
                elif fc_layer0_weight_data.shape[1] == 3 * T_var.shape[0]:
                    # Handles cat connections of DynEdgeConv readout for 3 global poolings
                    temp_T_var = [T_var, T_var, T_var]
                else:
                    # Inputs to postproc layer are concatenated from
                    # input + 4 conv layers output

                    # Here, T_var's of conv layers are the same for 
                    # - linear weight 
                    # - linear bias (if present)
                    # - bn weight (if present)
                    # - bn bias (if present)
                    # So we use T_var's of linear weight
                    postproc_T_vars = [
                        torch.eye(input_dim).cuda(),
                        T_vars['_gnn._conv_layers.0.nn.1.linear.weight'],
                        T_vars['_gnn._conv_layers.1.nn.1.linear.weight'],
                        T_vars['_gnn._conv_layers.2.nn.1.linear.weight'],
                        T_vars['_gnn._conv_layers.3.nn.1.linear.weight']
                    ]

                    if fc_layer0_weight_data.shape[1] == sum(x.shape[0] for x in postproc_T_vars):
                        # Handles cat connections of DynEdge cat of conv layers: 
                        # input layer + 4 T_var's of conv layers' biases
                        temp_T_var = postproc_T_vars
                    else:
                        raise ValueError(
                            f'size mismatch for DynEdge: '
                            f'fc_layer0_weight_data.shape == {fc_layer0_weight_data.shape}, '
                            f'T_var.shape == {T_var.shape}, '
                            f'sum(x.shape[0] for x in postproc_T_vars) == {sum(x.shape[0] for x in postproc_T_vars)}'
                        )
                temp_T_var = torch.block_diag(*temp_T_var)
                aligned_wt = torch.matmul(fc_layer0_weight_data, temp_T_var)
            else:
                aligned_wt = torch.matmul(fc_layer0_weight_data, T_var)
            logger.info(f"\t shape of aligned_wt {aligned_wt.shape}")

        # Calculate activation histograms
        if args.importance is None or (idx == num_layers -1):
            mu = get_histogram(args, 0, mu_cardinality, layer0_name)
            nu = get_histogram(args, 1, nu_cardinality, layer1_name)
        else:
            # mu = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
            mu = _get_neuron_importance_histogram(args, fc_layer0_weight_data, is_conv=False)
            nu = _get_neuron_importance_histogram(args, fc_layer1_weight_data, is_conv=False)
            assert args.proper_marginals
        logger.info(f"shape of mu is {mu.shape}, shape of nu is {nu.shape}")

        # Ground metrics
        M0, M1 = _process_ground_metric_from_acts(
            args, 
            False,
            ground_metric_object,
            [activations_0, activations_1]
        )

        logger.info(
            f"# of ground metric features in 0 is {(activations_0.view(activations_0.shape[0], -1)).shape[1]}", 
        )
        logger.info(
            f"# of ground metric features in 1 is {(activations_1.view(activations_1.shape[0], -1)).shape[1]}", 
        )

        if not args.gromov:
            logger.info(f"M0.shape is {M0.shape}")
        else:
            logger.info(f"M0.shape is {M0.shape}, M1.shape is {M1.shape}")

        if args.skip_last_layer and is_last_layer:
            logger.info(f"Skipping beacuse last layer {layer0_name}")
            if args.skip_last_layer_type == 'average':
                logger.info("\tSkipping last layer: average")
                if args.ensemble_step != 0.5:
                    logger.info("taking baby steps (even in skip) ! ")
                    avg_aligned_layers[layer0_name] = ((1-args.ensemble_step) * aligned_wt +
                                            args.ensemble_step * fc_layer1_weight)
                else:
                    avg_aligned_layers[layer0_name] = (((aligned_wt + fc_layer1_weight)/2))
                aligned_layers[layer0_name] = aligned_wt
            elif args.skip_last_layer_type == 'second':
                logger.info("\tSkipping last layer: second")
                avg_aligned_layers[layer0_name] = (fc_layer1_weight)
                aligned_layers[layer0_name] = fc_layer1_weight
        else:
            logger.info(f"Not skipping because last layer {layer0_name}")
            T_var = _get_current_layer_transport_map(args, mu, nu, M0, M1, idx=idx, layer_shape=layer_shape, eps=eps, layer_name=layer0_name)
            T_var, marginals = _compute_marginals(args, T_var, device, eps=eps)
            logger.info(f'T_var.shape is {T_var.shape}')

            scale_outer = 1.0
            if layer0_name.startswith('_gnn._post_processing.0'):
                scale_outer = (fc_layer0_weight_data.shape[1] - input_dim) / (fc_layer1_weight_data.shape[1] - input_dim)
            else:
                scale_outer = fc_layer0_weight_data.shape[1] / fc_layer1_weight_data.shape[1]
            
            scale_inner = 1.0
            if not is_bias:
                scale_inner = scale_outer
            else:
                scale_inner = 1.0
            
            if not is_bias or not (is_first_layer or is_last_layer):
                scale_outer = scale_outer ** 0.5
                scale_inner = scale_inner ** 0.5
            
            logger.info(f'scale_inner is {scale_inner}')
            logger.info(f'scale_outer is {scale_outer}')

            T_vars[layer0_name] = T_var * scale_inner

            logger.info(
                f"Ratio of trace to the matrix sum: {torch.trace(T_var) / torch.sum(T_var)}"
            )
            logger.info(
                "Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var * scale_inner), torch.sum(T_var * scale_inner))
            )
            setattr(args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var * scale_inner) / torch.sum(T_var * scale_inner)).item())

            logger.info(f"Shape of aligned_wt is {aligned_wt.shape}")
            logger.info(f"Shape of fc_layer0_weight_data is {fc_layer0_weight_data.shape}")
            if args.past_correction:
                logger.info(f"Output correction")
                if is_bias:
                    t_fc0_model = torch.matmul(aligned_wt, T_var * scale_inner)
                else:
                    t_fc0_model = torch.matmul((T_var * scale_inner).t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
            else:
                logger.info(f"No output correction")
                t_fc0_model = torch.matmul((T_var * scale_inner).t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))
            logger.info(f"Shape of t_fc0_model is {t_fc0_model.shape}")

            # Average the weights of aligned first layers
            if args.ensemble_step != 0.5:
                logger.info("taking baby steps! ")
                geometric_fc = (1 - args.ensemble_step) * t_fc0_model + \
                            args.ensemble_step * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
            else:
                geometric_fc = (t_fc0_model + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)) / 2
            avg_aligned_layers[layer0_name] = (geometric_fc)
            aligned_layers[layer0_name] = t_fc0_model
            incoming_layer_aligned = False
            
            T_var = T_var * scale_outer

            # remove cached variables to prevent out of memory
            activations_0 = None
            activations_1 = None
            mu = None
            nu = None
            fc_layer0_weight_data = None
            fc_layer1_weight_data = None
            M0 = None
            M1 = None

        idx += 1
    return avg_aligned_layers, aligned_layers, T_vars


def map_model(args, model_from, model_to, dataloader, n_samples, size_multiplier):
    has_bn = False
    for name, _ in model_from.named_parameters():
        if 'bn' in name:
            has_bn = True
            break
    if has_bn:
        # If BN are used, we need to compute the activations of the model
        # after the BN layers.
        
        # Also: no BN for last layer, so for it activations are computed
        # after linear layers
        layer_names = {
            name.replace('.bias', '') 
            for name, _ in model_from.named_parameters()
            if '.bn.bias' in name or '_affine.bias' in name
        }
    else:
        # If no BN are used, we can compute the activations of the model
        # after linear layers
        layer_names = {
            name.replace('.bias', '') 
            for name, _ in model_from.named_parameters()
            if '.bias' in name
        }
    models = [
        model_from.cuda(),
        model_to.cuda(), 
    ]
    with torch.no_grad():
        activations = compute_activations_across_models_v1(
            args, 
            models,
            dataloader,
            n_samples,
            mode='raw',
            layer_names=layer_names
        )
        
        # Remove the num_batches_tracked from the state_dicts
        model_from_params = models[0].state_dict()
        model_to_params = models[1].state_dict()
        num_batches_tracked = {
            name: param 
            for name, param in model_from_params.items() 
            if 'num_batches_tracked' in name
        }
        for name in num_batches_tracked:
            del model_from_params[name]
            del model_to_params[name]

        networks_named_params = list(zip(model_from_params.items(), model_to_params.items()))

        _, mapped_state_dict, _ = get_acts_wassersteinized_layers_modularized(
            args, networks_named_params, activations, test_loader=None, has_bn=has_bn)

    # TODO fix scaling
    mapped_state_dict = {
        k: ((v * (size_multiplier) ** 0.5) if k.startswith('_gnn._conv_layers.0.nn.0') else v) 
        for k, v in mapped_state_dict.items()
    }
    # Add the num_batches_tracked back to the resulting state_dict
    mapped_state_dict.update(num_batches_tracked)
        
    mapped_state_dict = {
        k: v.squeeze() 
        for k, v in mapped_state_dict.items()
    }

    model_mapped = deepcopy(model_to)
    # TODO: map running_mean and running_var of BN layers 
    # together with weights and biases. Now they are not mapped 
    # because not present in .named_parameters().
    model_mapped.load_state_dict(mapped_state_dict, strict=False)

    return model_mapped


def main(args):
    models = {}

    config['dataset_type'] = 'sqlite'
    config['train_mode'] = 'default'

    config_from = deepcopy(config)
    config_from['dynedge']['dynedge_layer_sizes'] = [(128, 256), (336, 256), (336, 256), (336, 256)]

    models['from'] = load_pretrained_model(
        config=config_from, 
        path=str(args.from_model_state_dict_path)
    )

    config_to = deepcopy(config)
    config_to['dynedge']['dynedge_layer_sizes'] = [
        (int(x * args.size_multiplier), int(y * args.size_multiplier)) 
        for x, y in [(128, 256), (336, 256), (336, 256), (336, 256)]
    ]

    models['to'], train_dataloader = load_pretrained_model(
        config=config_to, 
        path=str(args.to_model_state_dict_path),
        return_train_dataloader=True
    )

    models['mapped'] = map_model(
        args,
        models['from'], 
        models['to'], 
        train_dataloader, 
        args.act_num_samples,
        args.size_multiplier
    )
    models['mapped'].save_state_dict(str(args.mapped_model_save_dir / 'state_dict.pth'))


if __name__ == '__main__':
    args = get_args()
    main(args)


import numpy as np
import pandas as pd
import os
import torch
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List
from pytorch_lightning.callbacks import (
    EarlyStopping, 
    GradientAccumulationScheduler,
    ModelCheckpoint
)
from torch.optim.adamw import AdamW
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeKaggle
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from graphnet.training.labels import Direction
from graphnet.training.utils import make_dataloader
from graphnet.utilities.logging import get_logger
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


# Constants
features = FEATURES.KAGGLE
truth = TRUTH.KAGGLE

logger = get_logger()


def build_model(
    config: Dict[str, Any], 
    train_dataloader: Any, 
    fix_points = None
) -> StandardModel:
    """Builds GNN from config"""
    # Building model
    detector = IceCubeKaggle(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        global_pooling_schemes=["min", "max", "mean"],
        bias=config['bias'],
        fix_points=fix_points,
        **config['dynedge']
    )

    if config["target"] == 'direction':
        task = DirectionReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=config["target"],
            loss_function=VonMisesFisher3DLoss(),
            loss_weight='loss_weight' if 'loss_weight' in config else None,
            bias=config['bias'],
            fix_points=fix_points,
        )
        prediction_columns = [config["target"] + "_x", 
                              config["target"] + "_y", 
                              config["target"] + "_z", 
                              config["target"] + "_kappa" ]
        additional_attributes = ['zenith', 'azimuth', 'event_id']

    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=AdamW,
        optimizer_kwargs=config["optimizer_kwargs"],
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                len(train_dataloader) / 2,
                len(train_dataloader) * config["fit"]["max_epochs"],
            ],
            "factors": config["scheduler_kwargs"]["factors"],
        },
        scheduler_config={
            "interval": "step",
        },
    )
    model.prediction_columns = prediction_columns
    model.additional_attributes = additional_attributes
    
    return model

def load_pretrained_model(
    config: Dict[str,Any], 
    state_dict_path: str = '/kaggle/input/dynedge-pretrained/dynedge_pretrained_batch_1_to_50/state_dict.pth',
    return_train_dataloader: bool = False
) -> StandardModel:
    train_dataloader, _ = make_dataloaders(config = config)
    model = build_model(config = config, 
                        train_dataloader = train_dataloader)
    #model._inference_trainer = Trainer(config['fit'])
    model.load_state_dict(state_dict_path)
    model.prediction_columns = [config["target"] + "_x", 
                              config["target"] + "_y", 
                              config["target"] + "_z", 
                              config["target"] + "_kappa" ]
    model.additional_attributes = ['zenith', 'azimuth', 'event_id']

    if return_train_dataloader:
        return model, train_dataloader
    else:
        return model

def make_dataloaders(config: Dict[str, Any]) -> List[Any]:
    """Constructs training and validation dataloaders for training with early stopping."""
    loss_weight_kwargs = {}
    if 'loss_weight' in config:
        loss_weight_kwargs = config['loss_weight']

    train_dataloader = make_dataloader(db = config['path'],
                                            selection = None,
                                            pulsemaps = config['pulsemap'],
                                            features = features,
                                            truth = truth,
                                            batch_size = config['batch_size'],
                                            num_workers = config['num_workers'],
                                            shuffle = config['shuffle_train'],
                                            labels = {'direction': Direction()},
                                            index_column = config['index_column'],
                                            truth_table = config['truth_table'],
                                            **loss_weight_kwargs
                                            )
    
    validate_dataloader = make_dataloader(db = config['inference_database_path'],
                                            selection = None,
                                            pulsemaps = config['pulsemap'],
                                            features = features,
                                            truth = truth,
                                            batch_size = config['batch_size'],
                                            num_workers = config['num_workers'],
                                            shuffle = False,
                                            labels = {'direction': Direction()},
                                            index_column = config['index_column'],
                                            truth_table = config['truth_table'],
                                            **loss_weight_kwargs
                                            )
    return train_dataloader, validate_dataloader

def train_dynedge_from_scratch(config: Dict[str, Any], state_dict_path=None) -> StandardModel:
    """Builds and trains GNN according to config."""
    logger.info(f"features: {config['features']}")
    logger.info(f"truth: {config['truth']}")
    
    archive = os.path.join(config['base_dir'], "train_model_without_configs")
    run_name = f"dynedge_{config['target']}_{config['run_name_tag']}"

    train_dataloader, validate_dataloader = make_dataloaders(config = config)

    if state_dict_path is None:
        model = build_model(config, train_dataloader)
    else:
        model = load_pretrained_model(config, state_dict_path)

    # Training model
    callbacks = [
        GradientAccumulationScheduler(
            scheduling={0: config['accumulate_grad_batches']}
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
        ),
        ProgressBar(),
    ]

    if (
        'checkpoint_during_train_epoch_interval' in config and 
        config['checkpoint_during_train_epoch_interval'] is not None
    ):
        every_n_train_steps = int(
            len(train_dataloader) * 
            config['checkpoint_during_train_epoch_interval']
        )
        model_checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            every_n_train_steps=every_n_train_steps,
        )
    else:
        model_checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            every_n_epochs=1
        )
    callbacks.append(model_checkpoint_callback)

    model.fit(
        train_dataloader,
        validate_dataloader,
        callbacks=callbacks,
        **config["fit"],
    )
    return model

def inference(model, config: Dict[str, Any]) -> pd.DataFrame:
    """Applies model to the database specified in config['inference_database_path'] and saves results to disk."""
    # Make Dataloader
    test_dataloader = make_dataloader(db = config['inference_database_path'],
                                            selection = None, # Entire database
                                            pulsemaps = config['pulsemap'],
                                            features = features,
                                            truth = truth,
                                            batch_size = config['batch_size'],
                                            num_workers = config['num_workers'],
                                            shuffle = False,
                                            labels = {'direction': Direction()},
                                            index_column = config['index_column'],
                                            truth_table = config['truth_table'],
                                            )
    
    # Get predictions
    with torch.no_grad():
        results = model.predict_as_dataframe(
            gpus = [0],
            dataloader = test_dataloader,
            prediction_columns=model.prediction_columns,
            additional_attributes=model.additional_attributes,
        )
    # Save predictions and model to file
    archive = os.path.join(config['base_dir'], "train_model_without_configs")
    run_name = f"dynedge_{config['target']}_{config['run_name_tag']}"
    db_name = config['path'].split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)
    logger.info(f"Writing results to {path}")
    os.makedirs(path, exist_ok=True)

    results.to_csv(f"{path}/results.csv")
    return results


def compute_activations_across_models_v1(args, models, train_loader, num_samples, mode='mean',
                                         dump_activations=False, dump_path=None, layer_names=None):

    torch.manual_seed(args.activation_seed)

    # hook that computes the mean activations across data samples
    def get_activation(activation, name):
        def hook(model, input, output):
            # print("num of samples seen before", num_samples_processed)
            # print("output is ", output.detach())
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

    assert args.disable_bias
    # handle below for bias later on!
    # print("list of model named params ", list(models[0].named_parameters()))
    param_names = [tupl[0].replace('.weight', '') for tupl in models[0].named_parameters()]
    for idx, model in enumerate(models):

        # Initialize the activation dictionary for each model
        activations[idx] = {}
        layer_hooks = []
        # Set forward hooks for all layers inside a model
        for name, layer in model.named_modules():
            if name not in layer_names:
                continue
            layer_hooks.append(layer.register_forward_hook(get_activation(activations[idx], name)))
            print("set forward hook for layer named: ", name)

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



    print("num_personal_idx ", num_personal_idx)
    setattr(args, 'num_personal_idx', num_personal_idx)

    relu = torch.nn.ReLU()
    maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    avgpool = torch.nn.AvgPool2d(kernel_size=1, stride=1)

    # Combine the activations generated across the number of samples to form importance scores
    # The importance calculated is based on the 'mode' flag: which is either of 'mean', 'std', 'meanstd'

    # model_cfg = myutils.get_model_layers_cfg(args.model_name)
    for idx in range(len(models)):
        cfg_idx = 0
        for lnum, layer in enumerate(activations[idx]):
            print('***********')
            activations[idx][layer] = torch.concatenate(activations[idx][layer], dim=0)
            print("min of act: {}, max: {}, mean: {}".format(torch.min(activations[idx][layer]), torch.max(activations[idx][layer]), torch.mean(activations[idx][layer])))
            # assert (activations[idx][layer] >= 0).all()

            if not args.prelu_acts and not lnum == (len(activations[idx])-1):
                # print("activation was ", activations[idx][layer])
                print("applying relu ---------------")
                activations[idx][layer] = relu(activations[idx][layer])
                # print("activation now ", activations[idx][layer])
                print("after RELU: min of act: {}, max: {}, mean: {}".format(torch.min(activations[idx][layer]),
                                                                 torch.max(activations[idx][layer]),

                                                                 torch.mean(activations[idx][layer])))
                
            elif args.model_name == 'vgg11_nobias' and args.pool_acts and len(activations[idx][layer].shape)>3:

                if args.pool_relu:
                    print("applying relu ---------------")
                    activations[idx][layer] = relu(activations[idx][layer])

                activations[idx][layer] = activations[idx][layer].squeeze(1)

                # # apply maxpool wherever the next thing in config list is 'M'
                # if (cfg_idx + 1) < len(model_cfg):
                #     if model_cfg[cfg_idx+1] == 'M':
                #         print("applying maxpool ---------------")
                #         activations[idx][layer] = maxpool(activations[idx][layer])
                #         cfg_idx += 2
                #     else:
                #         cfg_idx += 1

                # # apply avgpool only for the last layer
                # if cfg_idx == len(model_cfg):
                #     print("applying avgpool ---------------")
                #     activations[idx][layer] = avgpool(activations[idx][layer])

                # unsqueeze back at axis 1
                activations[idx][layer] = activations[idx][layer].unsqueeze(1)

                print("checking stats after pooling")
                print("min of act: {}, max: {}, mean: {}".format(torch.min(activations[idx][layer]),
                                                                 torch.max(activations[idx][layer]),
                                                                 torch.mean(activations[idx][layer])))

            if mode == 'mean':
                activations[idx][layer] = activations[idx][layer].mean(dim=0)
            elif mode == 'std':
                activations[idx][layer] = activations[idx][layer].std(dim=0)
            elif mode == 'meanstd':
                activations[idx][layer] = activations[idx][layer].mean(dim=0) * activations[idx][layer].std(dim=0)

            if args.standardize_acts:
                mean_acts = activations[idx][layer].mean(dim=0)
                std_acts = activations[idx][layer].std(dim=0)
                print("shape of mean, std, and usual acts are: ", mean_acts.shape, std_acts.shape, activations[idx][layer].shape)
                activations[idx][layer] = (activations[idx][layer] - mean_acts)/(std_acts + 1e-9)
            elif args.center_acts:
                mean_acts = activations[idx][layer].mean(dim=0)
                print("shape of mean and usual acts are: ", mean_acts.shape, activations[idx][layer].shape)
                activations[idx][layer] = (activations[idx][layer] - mean_acts)
            elif args.normalize_acts:
                print("normalizing the activation vectors")
                activations[idx][layer] = normalize_tensor(activations[idx][layer])
                print("min of act: {}, max: {}, mean: {}".format(torch.min(activations[idx][layer]),
                                                                 torch.max(activations[idx][layer]),
                                                                 torch.mean(activations[idx][layer])))

            print("activations for idx {} at layer {} have the following shape ".format(idx, layer), activations[idx][layer].shape)
            print('-----------')

    # Dump the activations for all models onto disk
    if dump_activations and dump_path is not None:
        for idx in range(len(models)):
            save_activations(idx, activations[idx], dump_path)

    # Remove the hooks (as this was intefering with prediction ensembling)
    for idx in range(len(forward_hooks)):
        for hook in forward_hooks[idx]:
            hook.remove()


    return activations


def process_activations(args, activations, layer0_name, layer1_name):
    activations_0 = activations[0][layer0_name.replace('.weight', '').replace('.bias', '')].squeeze()
    activations_1 = activations[1][layer1_name.replace('.weight', '').replace('.bias', '')].squeeze()

    # assert activations_0.shape == activations_1.shape
    _check_activation_sizes(args, activations_0, activations_1)

    if args.same_model != -1:
        # sanity check when averaging the same model (with value being the model index)
        assert (activations_0 == activations_1).all()
        print("Are the activations the same? ", (activations_0 == activations_1).all())

    if len(activations_0.shape) == 2:
        activations_0 = activations_0.t()
        activations_1 = activations_1.t()
    elif len(activations_0.shape) > 2:
        reorder_dim = [l for l in range(1, len(activations_0.shape))]
        reorder_dim.append(0)
        print("reorder_dim is ", reorder_dim)
        activations_0 = activations_0.permute(*reorder_dim).contiguous()
        activations_1 = activations_1.permute(*reorder_dim).contiguous()

    return activations_0, activations_1


def get_acts_wassersteinized_layers_modularized(
    args, 
    networks, 
    activations, 
    eps=1e-7, 
    train_loader=None, 
    test_loader=None,
    input_dim=17
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


    avg_aligned_layers, aligned_layers, T_vars = [], OrderedDict(), OrderedDict()
    T_var = None
    if args.handle_skips:
        skip_T_var = None
        skip_T_var_idx = -1
        residual_T_var = None
        residual_T_var_idx = -1

    marginals_beta = None
    # print(list(networks[0].parameters()))
    previous_layer_shape = None
    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    ground_metric_object = GroundMetric(args)

    if args.update_acts or args.eval_aligned:
        model0_aligned_layers = []

    if args.gpu_id==-1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id))

    networks_named_params = list(zip(networks[0].named_parameters(), networks[1].named_parameters()))
    idx = 0
    incoming_layer_aligned = True # for input
    while idx < num_layers:
        ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) = networks_named_params[idx]
    # for idx,  in \
    #         enumerate(zip(network0_named_params, network1_named_params)):
        print("\n--------------- At layer index {} ------------- \n ".format(idx))
        # layer shape is out x in
        # assert fc_layer0_weight.shape == fc_layer1_weight.shape
        # assert _check_layer_sizes(args, idx, fc_layer0_weight.shape, fc_layer1_weight.shape, num_layers)
        print("Previous layer shape is ", previous_layer_shape)
        previous_layer_shape = fc_layer1_weight.shape

        # will have shape layer_size x act_num_samples
        # layer0_name_reduced = _reduce_layer_name(layer0_name)
        # layer1_name_reduced = _reduce_layer_name(layer1_name)

        # print("let's see the difference in layer names", layer0_name.replace('.' + layer0_name.split('.')[-1], ''), layer0_name_reduced)
        # print(activations[0][layer0_name.replace('.' + layer0_name.split('.')[-1], '')].shape, 'shape of activations generally')
        # for conv layer I need to make the act_num_samples dimension the last one, but it has the intermediate dimensions for
        # height and width of channels, so that won't work.
        # So convert (num_samples, layer_size, ht, wt) -> (layer_size, ht, wt, num_samples)

        activations_0, activations_1 = process_activations(args, activations, layer0_name, layer1_name)

        # print("activations for 1st model are ", activations_0)
        # print("activations for 2nd model are ", activations_1)


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
        if len(layer_shape) > 2:
            is_conv = True
        else:
            is_conv = False

        fc_layer0_weight_data = _get_layer_weights(fc_layer0_weight, is_conv)
        fc_layer1_weight_data = _get_layer_weights(fc_layer1_weight, is_conv)

        if idx == 0 or incoming_layer_aligned:
            aligned_wt = fc_layer0_weight_data

        else:

            print("shape of layer: model 0", fc_layer0_weight_data.shape)
            print("shape of layer: model 1", fc_layer1_weight_data.shape)

            print("shape of activations: model 0", activations_0.shape)
            print("shape of activations: model 1", activations_1.shape)


            print("shape of previous transport map", T_var.shape)

            # aligned_wt = None, this caches the tensor and causes OOM
            if is_conv:
                if args.handle_skips:
                    assert len(layer0_shape) == 4
                    # save skip_level transport map if there is block ahead
                    if layer0_shape[1] != layer0_shape[0]:
                        if not (layer0_shape[2] == 1 and layer0_shape[3] == 1):
                            print(f'saved skip T_var at layer {idx} with shape {layer0_shape}')
                            skip_T_var = T_var.clone()
                            skip_T_var_idx = idx
                        else:
                            print(
                                f'utilizing skip T_var saved from layer layer {skip_T_var_idx} with shape {skip_T_var.shape}')
                            # if it's a shortcut (128, 64, 1, 1)
                            residual_T_var = T_var.clone()
                            residual_T_var_idx = idx  # use this after the skip
                            T_var = skip_T_var
                        print("shape of previous transport map now is", T_var.shape)
                    else:
                        if residual_T_var is not None and (residual_T_var_idx == (idx - 1)):
                            T_var = (T_var + residual_T_var) / 2
                            print("averaging multiple T_var's")
                        else:
                            print("doing nothing for skips")
                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)

            else:
                if 'bias' in layer0_name:
                    aligned_wt = torch.matmul(fc_layer0_weight_data, T_var).t()
                elif fc_layer0_weight_data.shape[1] != T_var.shape[0]:
                    print("shape of fc_layer0_weight_data", fc_layer0_weight_data.shape)
                    print("shape of T_var", T_var.shape)
                    if fc_layer0_weight_data.shape[1] == 2 * T_var.shape[0]:
                        # Handles cat connections of DynEdgeConv nn's first layer: same T_var twise
                        temp_T_var = [T_var, T_var]
                    elif fc_layer0_weight_data.shape[1] == 3 * T_var.shape[0]:
                        # Handles cat connections of DynEdgeConv readout for 3 global poolings
                        temp_T_var = [T_var, T_var, T_var]
                    else:
                        postproc_T_vars = \
                            [torch.eye(input_dim).cuda()] + \
                            [x for name, x in T_vars.items() if '2.weight' in name]
                        if fc_layer0_weight_data.shape[1] == sum(x.shape[0] for x in postproc_T_vars):
                            # Handles cat connections of DynEdge cat of conv layers: 
                            # input layer + 4 T_var's of conv layers' biases
                            temp_T_var = postproc_T_vars
                        else:
                            raise ValueError(
                                f'size mismatch for DynEdge: '
                                f'fc_layer0_weight_data.shape == {fc_layer0_weight_data.shape},'
                                f' T_var.shape == {T_var.shape}'
                            )
                    temp_T_var = torch.block_diag(*temp_T_var)
                    aligned_wt = torch.matmul(fc_layer0_weight_data, temp_T_var)
                else:
                    aligned_wt = torch.matmul(fc_layer0_weight_data, T_var)


        #### Refactored ####

            if args.update_acts:
                assert args.second_model_name is None
                activations_0, activations_1 = _get_updated_acts_v0(args, layer_shape, aligned_wt,
                                                                    model0_aligned_layers, networks,
                                                                    test_loader, [layer0_name, layer1_name])

        if args.importance is None or (idx == num_layers -1):
            mu = get_histogram(args, 0, mu_cardinality, layer0_name)
            nu = get_histogram(args, 1, nu_cardinality, layer1_name)
        else:
            # mu = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
            mu = _get_neuron_importance_histogram(args, fc_layer0_weight_data, is_conv)
            nu = _get_neuron_importance_histogram(args, fc_layer1_weight_data, is_conv)
            print(mu, nu)
            assert args.proper_marginals

        if args.act_bug:
            # bug from before (didn't change the activation part)
            # only for reproducing results from previous version
            M0 = ground_metric_object.process(
                aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
            )
        else:
            # debugged part
            print("Refactored ground metric calc")
            M0, M1 = _process_ground_metric_from_acts(args, is_conv, ground_metric_object,
                                                      [activations_0, activations_1])

            print("# of ground metric features in 0 is  ", (activations_0.view(activations_0.shape[0], -1)).shape[1])
            print("# of ground metric features in 1 is  ", (activations_1.view(activations_1.shape[0], -1)).shape[1])

        if args.debug and not args.gromov:
            # bug from before (didn't change the activation part)
            M_old = ground_metric_object.process(
                aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
            )
            print("Frobenius norm of old (i.e. bug involving wts) and new are ",
                  torch.norm(M_old, 'fro'), torch.norm(M0, 'fro'))
            print("Frobenius norm of difference between ground metric wrt old ",
                  torch.norm(M0 - M_old, 'fro') / torch.norm(M_old, 'fro'))

            print("ground metric old (i.e. bug involving wts) is ", M_old)
            print("ground metric new is ", M0)

        ####################

        if args.same_model!=-1:
            print("Checking ground metric matrix in case of same models")
            if not args.gromov:
                print(M0)
            else:
                print(M0, M1)

        if args.skip_last_layer and idx == (num_layers - 1):

            if args.skip_last_layer_type == 'average':
                print("Simple averaging of last layer weights. NO transport map needs to be computed")
                if args.ensemble_step != 0.5:
                    print("taking baby steps (even in skip) ! ")
                    avg_aligned_layers.append((1-args.ensemble_step) * aligned_wt +
                                              args.ensemble_step * fc_layer1_weight)
                else:
                    avg_aligned_layers.append(((aligned_wt + fc_layer1_weight)/2))
                aligned_layers[layer0_name] = aligned_wt
            elif args.skip_last_layer_type == 'second':
                print("Just giving the weights of the second model. NO transport map needs to be computed")
                avg_aligned_layers.append(fc_layer1_weight)
                aligned_layers[layer0_name] = fc_layer1_weight

            return avg_aligned_layers, aligned_layers, T_vars

        print("ground metric (m0) is ", M0)

        T_var = _get_current_layer_transport_map(args, mu, nu, M0, M1, idx=idx, layer_shape=layer_shape, eps=eps, layer_name=layer0_name)

        T_var, marginals = _compute_marginals(args, T_var, device, eps=eps)

        scale = fc_layer0_weight_data.shape[1] / fc_layer1_weight_data.shape[1]
        if not (idx == 0 or idx == (num_layers - 1)):
            scale = scale ** 0.5
        print('scale is ', scale)
        T_var = T_var * scale

        T_vars[layer0_name] = T_var

        if args.debug:
            if idx == (num_layers - 1):
                print("there goes the last transport map: \n ", T_var)
                print("and before marginals it is ", T_var/marginals)
            else:
                print("there goes the transport map at layer {}: \n ".format(idx), T_var)

        print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))
        print("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))
        setattr(args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

        if args.past_correction:
            print("Shape of aligned wt is ", aligned_wt.shape)
            print("Shape of fc_layer0_weight_data is ", fc_layer0_weight_data.shape)

            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))

        # Average the weights of aligned first layers
        if args.ensemble_step != 0.5:
            print("taking baby steps! ")
            geometric_fc = (1 - args.ensemble_step) * t_fc0_model + \
                           args.ensemble_step * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
        else:
            geometric_fc = (t_fc0_model + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)) / 2
        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)
        avg_aligned_layers.append(geometric_fc)
        aligned_layers[layer0_name] = t_fc0_model


        # print("The averaged parameters are :", geometric_fc)
        # print("The model0 and model1 parameters were :", fc_layer0_weight.data, fc_layer1_weight.data)

        if args.update_acts or args.eval_aligned:
            assert args.second_model_name is None
            # the thing is that there might be conv layers or other more intricate layers
            # hence there is no point in having them here
            # so instead call the compute_activations script and pass it the model0 aligned layers
            # and also the aligned weight computed (which has been aligned via the prev T map, i.e. incoming edges).
            if is_conv and layer_shape != t_fc0_model.shape:
                t_fc0_model = t_fc0_model.view(layer_shape)
            model0_aligned_layers.append(t_fc0_model)
            _, acc = update_model(args, networks[0], model0_aligned_layers, test=True,
                                  test_loader=test_loader, idx=0)
            print("For layer idx {}, accuracy of the updated model is {}".format(idx, acc))
            setattr(args, 'model0_aligned_acc_layer_{}'.format(str(idx)), acc)
            if idx == (num_layers - 1):
                setattr(args, 'model0_aligned_acc', acc)

        incoming_layer_aligned = False
        next_aligned_wt_reshaped = None

        # remove cached variables to prevent out of memory
        activations_0 = None
        activations_1 = None
        mu = None
        nu = None
        fc_layer0_weight_data = None
        fc_layer1_weight_data = None
        M0 = None
        M1 = None
        cpuM = None

        idx += 1
    return avg_aligned_layers, aligned_layers, T_vars


layer_names = {
    '_gnn._conv_layers.0.nn.0',
    '_gnn._conv_layers.0.nn.2',
    '_gnn._conv_layers.1.nn.0',
    '_gnn._conv_layers.1.nn.2',
    '_gnn._conv_layers.2.nn.0',
    '_gnn._conv_layers.2.nn.2',
    '_gnn._conv_layers.3.nn.0',
    '_gnn._conv_layers.3.nn.2',
    '_gnn._post_processing.0',
    '_gnn._post_processing.2',
    '_gnn._readout.0',
    '_tasks.0._affine',
}


def map_model(args, model_from, model_to, dataloader, n_samples):
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
        _, mapped_state_dict, _ = get_acts_wassersteinized_layers_modularized(
            args, models, activations, train_loader=dataloader, test_loader=None)
        
    mapped_state_dict = {
        k: v[:, 0] if 'bias' in k else v for k, v in mapped_state_dict.items()
    }

    model_mapped = deepcopy(model_to)
    model_mapped.load_state_dict(mapped_state_dict)

    return model_mapped


def convert_to_3d(df: pd.DataFrame) -> pd.DataFrame:
    """Converts zenith and azimuth to 3D direction vectors"""
    df['true_x'] = np.cos(df['azimuth']) * np.sin(df['zenith'])
    df['true_y'] = np.sin(df['azimuth']) * np.sin(df['zenith'])
    df['true_z'] = np.cos(df['zenith'])
    return df


def calculate_angular_error(df : pd.DataFrame) -> pd.DataFrame:
    """Calcualtes the opening angle (angular error) between true and reconstructed direction vectors"""
    df['angular_error'] = np.arccos(df['true_x']*df['direction_x'] + df['true_y']*df['direction_y'] + df['true_z']*df['direction_z'])
    return df

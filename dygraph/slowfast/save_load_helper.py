import os
import numpy as np
import paddle
import copy


def sub_to_normal_bn(sd):
    """
    When save, Convert the Sub-BN paprameters to normal BN parameters in a state dict.
    There are two copies of BN layers in a Sub-BN implementation: `bn.bn` and
    `bn.split_bn`. `bn.split_bn` is used during training and
    "compute_precise_bn". Before saving or evaluation, its stats are copied to
    `bn.bn`. We rename `bn.bn` to `bn` and store it to be consistent with normal
    BN layers.
    Args:
        sd (OrderedDict): a dict of parameters which might contain Sub-BN
        parameters.
    Returns:
        new_sd (OrderedDict): a dict with Sub-BN parameters reshaped to
        normal parameters.
    """
    modifications = [
        ("bn.bn._mean", "bn._mean"),
        ("bn.bn._variance", "bn._variance"),
    ]
    to_remove = ["bn.bn.", ".split_bn."]
    key_list = list(sd.keys())  #odict_keys to list
    for key in key_list:
        for before, after in modifications:
            if key.endswith(before):
                new_key = key.split(before)[0] + after
                sd[new_key] = sd.pop(key)

        for rm in to_remove:
            if rm in key and key in sd:
                del sd[key]


def normal_to_sub_bn(checkpoint_sd, model_sd):
    """
    When load, Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
    Args:
        checkpoint_sd (OrderedDict): source dict of parameters.
        model_sd (OrderedDict): target dict of parameters.
    Returns:
        new_sd (OrderedDict): converted dict of parameters.
    """
    for key in model_sd:
        if key not in checkpoint_sd:
            # not to replace bn.weight and bn.bias
            if "bn.split_bn." in key and "bn.weight" not in key and "bn.bias" not in key:
                load_key = key.replace("bn.split_bn.", "bn.")
                bn_key = key.replace("bn.split_bn.", "bn.bn.")
                checkpoint_sd[key] = checkpoint_sd.pop(load_key)
                checkpoint_sd[bn_key] = checkpoint_sd[key]

    # match the shape of bn.split_bn._xx
    # model_sd: split_bn.rm.shape = num_feature*num_split
    # checkpoint_sd: split_bn.rm.shape = bn.rm.shape = num_feature
    for key in model_sd:
        if key in checkpoint_sd:
            model_blob_shape = model_sd[key].shape  #bn.split_bn
            c2_blob_shape = checkpoint_sd[key].shape  #bn.bn

            if (len(model_blob_shape) == 1 and len(c2_blob_shape) == 1 and
                    model_blob_shape[0] > c2_blob_shape[0] and
                    model_blob_shape[0] % c2_blob_shape[0] == 0):
                before_shape = checkpoint_sd[key].shape
                checkpoint_sd[key] = np.concatenate(
                    [checkpoint_sd[key]] *
                    (model_blob_shape[0] // c2_blob_shape[0]))
                if 'split_bn' not in key:  #split_bn is excepted
                    print("{} {} -> {}".format(key, before_shape, checkpoint_sd[
                        key].shape))
    return checkpoint_sd


def mapping_opt_dict(opt_dict, model_key_list):
    """
    Paddle Name schedule: conv_1.w -> conv_2.w
    Sometimes: sub_bn -> bn
    when re-build model, we desire the parameter name to be coincident,
    but the parameters name index will be added, as conv_1 to conv_2, not conv_1.
    It will raise error if we set old saved parameters to new created optimizer.
    as conv_2 cannot find in state_dict(only conv_1).
    Args:
        opt_dict: optimizer state dict, including the name and value of parameters gradient.
        model_key_list: the parameters name list of re-build model.
    Return: optimizer state dict with modified keys
    """

    def get_name_info(PNAME, PN_key_list, key_list):
        min_index = float('inf')
        max_index = 0
        for name in PN_key_list[1:]:
            for key in key_list:
                if name in key:
                    index = int(key.split('.')[0].split(name)[-1])
                    if index < min_index:
                        min_index = index
                    if index > max_index:
                        max_index = index
            num_name = max_index - min_index + 1
            PNAME[name].append((min_index, max_index, num_name))
            min_index = float('inf')
            max_index = 0

    PNAME = {
        "LR_Scheduler": [],
        "conv3d_": [],
        "linear_": [],
        "sub_batch_norm3d_": [],
        "batch_norm3d_": [],
    }

    pd_key_list = list(opt_dict.keys())
    print("The number of parameters in saved optimizer state dict = {}".format(
        len(pd_key_list)))
    print("The number of parameters in re-build model list = {}".format(
        len(model_key_list)))
    # 1 may be LR_Scheduler
    PN_key_list = list(PNAME.keys())

    # get the number of each PNAME
    get_name_info(PNAME, PN_key_list, pd_key_list)
    get_name_info(PNAME, PN_key_list, model_key_list)
    print("[Parameters info] prefix: min_index, max_index, number_params: \n",
          PNAME)

    # whether to change name of bn layer
    change_name = False
    if PNAME["sub_batch_norm3d_"][0][-1] == -float('inf'):
        PN_key_list.remove("sub_batch_norm3d_")
        if PNAME["sub_batch_norm3d_"][1][-1] != -float('inf'):
            print(
                "Optimizer state dict saved bn, but Re-build model use sub_bn, changed name!"
            )
            change_name = True
        else:
            print("Optimizer state dict saved bn, and Re-build model use bn")
    else:
        PN_key_list.remove("batch_norm3d_")
        if PNAME["sub_batch_norm3d_"][1][-1] == -float('inf'):
            print(
                "Optimizer state dict saved sub_bn, but Re-build model use bn, changed name!"
            )
            change_name = True
        else:
            print(
                "Optimizer state dict saved sub_bn, Re-build model use sub_bn")

    #update key name
    # sub_bn -> bn name mapping, pre-define dict
    change_dict = {
        "sub_batch_norm3d_": "batch_norm3d_",
        "batch_norm3d_": "sub_batch_norm3d_"
    }
    for key in pd_key_list:
        for name in PN_key_list[1:]:
            if key.startswith(name):
                start = change_dict[name] if (change_name and
                                              "batch_norm" in name) else name
                str_index = key.split('.')[0].split(name)[-1]
                index = int(str_index)
                new_index = str(index + (PNAME[start][1][0] - PNAME[name][0][0]
                                         ))
                end = key.split('.')[-1]
                update_key = start + new_index + '.' + end
                opt_dict[update_key] = opt_dict.pop(key)

    return opt_dict


def subn_save(save_dir, name_prefix, epoch, video_model, optimizer):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, name_prefix + "{:05d}".format(epoch))
    model_dict = video_model.state_dict()
    sub_to_normal_bn(model_dict)
    opti_dict = optimizer.state_dict()
    paddle.save(model_dict, model_path + '.pdparams')
    paddle.save(opti_dict, model_path + '.pdopt')
    print('[Saved Epoch {} parameters and optimizer state ]'.format(epoch))


def subn_load(model, ck_path, optimizer=None):
    """
    Load the checkpoint from the given file.
    Args:
        model (model): model to load the weights from the checkpoint.
        optimizer (optim, optional): optimizer to load the historical state.
        ck_path (str): checkpoint path
    Returns:
        (int): the number of training epoch of the checkpoint.
    """

    assert os.path.exists(ck_path + ".pdparams"), \
        "Given dir {}.pdparams not exist.".format(ck_path)
    print("load checkpint from {}.pdparams".format(ck_path))

    model_dict = model.state_dict()
    checkpoint_dict = paddle.load(ck_path + ".pdparams")
    #    checkpoint_dict = copy.deepcopy(checkpoint_dict_orig)  #not modify when multi card
    pre_train_dict = normal_to_sub_bn(checkpoint_dict, model_dict)

    # Match pre-trained weights that have same shape as current model.
    pre_train_dict_match = {
        k: v
        for k, v in pre_train_dict.items()
        if k in model_dict and tuple(v.shape) == tuple(model_dict[k].shape)
    }

    # Weights that do not have match from the pre-trained model.
    not_load_layers = [
        k for k in model_dict.keys() if k not in pre_train_dict_match.keys()
    ]
    # Log weights that are not loaded with the pre-trained weights.
    if not_load_layers:
        for k in not_load_layers:
            if 'bn.weight' not in k and 'bn.bias' not in k:
                print("Network weights {} not loaded.".format(k))

    # Load pre-trained weights.
    model.set_state_dict(pre_train_dict_match)

    if optimizer:
        assert os.path.exists(ck_path + ".pdopt"), \
            "Given dir {}.pdopt not exist.".format(ck_path)
        print("load checkpint from {}.pdopt".format(ck_path))
        opt_dict = paddle.load(ck_path + ".pdopt")
        # get parameters that required gradient from re-build model
        model_key_list = []
        for param in model.parameters():
            if param.stop_gradient == False:
                model_key_list.append(param.name)

        new_opt_dict = mapping_opt_dict(opt_dict, model_key_list)
        optimizer.set_state_dict(new_opt_dict)

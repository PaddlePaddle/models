import os
import pickle
import sys

import torch

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..', '..'))

from ltr.admin.loading import torch_load_legacy
import paddle.fluid as fluid
import numpy as np

from ltr_pp.models.bbreg.atom import atom_resnet50, atom_resnet18


def load_torch_pth_save_to_pkl(params_path=None, save_split=False):
    if not os.path.exists(params_path):
        raise Exception("{} does not exists!!!".format(params_path))
    if not save_split and os.path.exists(params_path + '.pkl'):
        print("{}.pkl already exists!!!".format(params_path))
        return
    params_pkl = {}

    params = torch_load_legacy(params_path)['net']
    opt_params = torch_load_legacy(params_path)['optimizer']
    keys = [k for k in params.keys() if 'num_batches_tracked' not in k]

    torch_resnet = {}
    torch_iounet = {}

    for key in keys:
        params_data = params[key].detach().cpu().numpy()
        params_pkl[key] = params_data
        # "bb_regressor  feature_extractor"
        if "feature_extractor" in key:
            torch_resnet[key] = params_data
            print(key, '\t', list(params[key].cpu().numpy().shape))
        if "bb_regressor" in key:
            torch_iounet[key] = params_data
            print(key, '\t', list(params[key].cpu().numpy().shape))

    pickle.dump(params_pkl, open(params_path + '.pkl', 'wb'))
    print(params_path + ".pkl save done!!!")

    if save_split:
        pickle.dump(torch_resnet, open(params_path + "-resnet.pkl", 'wb'))
        print(params_path + "-resnet.pkl save done!!!")

        pickle.dump(torch_iounet, open(params_path + "-iounet.pkl", 'wb'))
        print(params_path + "-iounet.pkl save done!!!")


def pkl2paddle(atom_pkl_path, network_name, save_split=True):
    ## extract torch ATOM params from pkl file
    ## and set value to paddle ATOM params
    if not save_split and os.path.exists(atom_pkl_path + '.pdparams'):
        print(atom_pkl_path + ".pdparams already exists!!!")
        return

    Atom_pkl = pickle.load(open(atom_pkl_path, 'rb'))
    params_keys = Atom_pkl.keys()
    pkl_resnet_keys = [key for key in params_keys if "feature_extractor" in key]
    pkl_iounet_keys = [key for key in params_keys if "bb_regressor" in key]

    ## build ATOM model
    with fluid.dygraph.guard():
        if network_name == 'atom_resnet50':
            model = atom_resnet50(backbone_pretrained=False)
        elif network_name == 'atom_resnet18':
            model = atom_resnet18(backbone_pretrained=False)

        # extract backbone from Atom model
        backbone = model.feature_extractor
        backbone_params = backbone.state_dict()
        backbone_keys = backbone_params.keys()

        # set backbone params
        for k1, k2 in zip(pkl_resnet_keys, backbone_keys):
            print(k1, '\t', Atom_pkl[k1].shape, '\t', k2, '\t', backbone_params[k2].shape)
            if list(Atom_pkl[k1].shape) == list(backbone_params[k2].shape):
                param_array = Atom_pkl[k1]
                backbone_params[k2].set_value(param_array)
            elif list(Atom_pkl[k1].shape)[::-1] == list(backbone_params[k2].shape):
                param_array = np.transpose(Atom_pkl[k1], [1, 0])
                backbone_params[k2].set_value(param_array)
            else:
                raise Exception("the shape of params from pkl:{} and paddle \
                                                    ResNet:{} is not match".format(k1, k2))
        save_dir = 'pretrained_models/atom'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if save_split:
            fluid.save_dygraph(backbone_params, '{}/{}-ResNet'.format(save_dir, network_name))

        # set iounet params
        iounet = model.bb_regressor
        iounet_params = iounet.state_dict()
        iounet_keys = iounet_params.keys()
        for k1, k2 in zip(pkl_iounet_keys, iounet_keys):
            print(k1, '\t', Atom_pkl[k1].shape, '\t', k2, '\t', iounet_params[k2].shape)
            if list(Atom_pkl[k1].shape) == list(iounet_params[k2].shape):
                param_array = Atom_pkl[k1]
                iounet_params[k2].set_value(param_array)
            elif list(Atom_pkl[k1].shape)[::-1] == list(iounet_params[k2].shape):
                param_array = np.transpose(Atom_pkl[k1], [1, 0])
                iounet_params[k2].set_value(param_array)
            else:
                print(list(Atom_pkl[k1].shape), list(iounet_params[k2].shape))
                raise Exception("the shape of params from pkl:{} and paddle \
                                    IOUNet:{} is not match".format(k1, k2))
        if save_split:
            fluid.save_dygraph(iounet_params, '{}/{}-iounet'.format(save_dir, network_name))

        fluid.save_dygraph(model.state_dict(), '{}/{}'.format(save_dir, network_name))
        print("model params transfer done!!!")


if __name__ == '__main__':
    params_path = "pretrained_models/atom-torch/atom_res50.pth.tar"
    load_torch_pth_save_to_pkl(params_path)
    pkl2paddle(params_path + ".pkl", "atom_resnet50")

    params_path = "pretrained_models/atom-torch/atom_default.pth"
    load_torch_pth_save_to_pkl(params_path)
    pkl2paddle(params_path + ".pkl", "atom_resnet18")

import os
import os.path as osp
import sys

import numpy as np
from paddle import fluid

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..', '..'))

from ltr_pp.models.backbone.resnet import ResNet, Bottleneck, resnet18, resnet50


def test_resnet50_paddle():
    with fluid.dygraph.guard():
        data = np.random.random((2, 3, 288, 288)).astype(np.float32)
        data = fluid.dygraph.to_variable(data)
        model = ResNet("test", Block=Bottleneck, layers=50)
        model.eval()

        res = model(data, ['block1', 'block2', 'block3', 'fc'])
        for key in res:
            print(key)
        print("layer2", res['block1'].shape,
              "layer3", res['block2'].shape,
              "layer4", res['block3'].shape)
        state_dict = model.state_dict()
        for k in state_dict:
            print(k, state_dict[k].shape)


def torch_to_paddle(network_name):
    import torchvision
    import torch

    if network_name == 'ResNet50':
        resnet_th = torchvision.models.resnet50(pretrained=True)
    elif network_name == 'ResNet18':
        resnet_th = torchvision.models.resnet18(pretrained=True)
    resnet_th.eval()
    th_statedict = resnet_th.state_dict()
    params_keys = [k for k in th_statedict.keys() if 'num_batches_tracked' not in k]

    with fluid.dygraph.guard():
        if network_name == 'ResNet50':
            resnet_pd = resnet50("ResNet50")
        elif network_name == 'ResNet18':
            resnet_pd = resnet18("ResNet18")
        else:
            raise NotImplementedError

        pd_statedict = resnet_pd.state_dict()

        for kth, kpd in zip(params_keys, pd_statedict.keys()):
            print(kth, th_statedict[kth].shape, kpd, pd_statedict[kpd].shape)
            if list(th_statedict[kth].shape) == list(pd_statedict[kpd].shape):
                params_array = th_statedict[kth].numpy()
                pd_statedict[kpd].set_value(params_array)
            elif list(th_statedict[kth].shape) == list(pd_statedict[kpd].shape)[::-1]:
                params_array = np.transpose(th_statedict[kth].numpy(), [1, 0])
                pd_statedict[kpd].set_value(params_array)
            else:
                raise Exception("the shape of params from pkl:{} and paddle \
                                            IOUNet:{} is not match".format(kth, kpd))

        if not os.path.exists('pretrained_models/backbone'):
            os.makedirs('pretrained_models/backbone')
        fluid.save_dygraph(pd_statedict, './pretrained_models/backbone/{}'.format(network_name))
        print("model params transfer done!!!")
        resnet_pd.load_dict(pd_statedict)
        resnet_pd.eval()

        data = np.random.uniform(-1, 1, [2, 3, 224, 224]).astype(np.float32)
        data_th = torch.from_numpy(data)
        data_pd = fluid.dygraph.to_variable(data)

        res_th = resnet_th(data_th)
        res_pd = resnet_pd(data_pd, ['fc'])
    np.testing.assert_allclose(res_th.detach().numpy(), res_pd['fc'].numpy(), atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    torch_to_paddle("ResNet18")
    torch_to_paddle("ResNet50")

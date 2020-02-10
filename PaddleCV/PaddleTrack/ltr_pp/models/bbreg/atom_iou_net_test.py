import torch
import unittest
import numpy as np
import pickle

from paddle import fluid
from paddle.fluid import layers
from bilib import crash_on_ipy

import os.path as osp
import sys

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..', '..', '..'))

from pytracking_pp.libs.paddle_utils import n2p, n2t, p2n, t2n
from ltr_pp.models.bbreg.atom import atom_resnet50 as atom_p
from ltr.models.bbreg.atom import atom_resnet50 as atom_t

from ltr.admin import loading


def test_pproi_pooling():
    pass


def test_iounet_forward(mode='eval'):
    import numpy as np
    from paddle.fluid import dygraph, layers

    # prepare data
    batch_size = 10
    num_proposals = 16
    rng = np.random.RandomState(0)
    input_dim = (512, 1024)

    train_feat1_np = rng.uniform(-1, 1, (1, batch_size, input_dim[0], 36, 36)).astype('float32')
    train_feat2_np = rng.uniform(-1, 1, (1, batch_size, input_dim[1], 18, 18)).astype('float32')

    test_feat1_np = rng.uniform(-1, 1, (1, batch_size, input_dim[0], 36, 36)).astype('float32')
    test_feat2_np = rng.uniform(-1, 1, (1, batch_size, input_dim[1], 18, 18)).astype('float32')

    rois_x1y1 = rng.uniform(0, 288, (1, batch_size, num_proposals, 2))
    rois_wh = rng.uniform(2, 144, (1, batch_size, num_proposals, 2))
    test_proposals = np.concatenate([rois_x1y1, rois_wh], axis=3).astype(np.float32)

    rois_x1y1 = rng.uniform(0, 288, (1, batch_size, 2))
    rois_wh = rng.uniform(2, 144, (1, batch_size, 2))
    train_bb_np = np.concatenate([rois_x1y1, rois_wh], axis=2).astype(np.float32)

    # compute torch forward
    net_t = atom_t(backbone_pretrained=False)
    state_dict = loading.torch_load_legacy('ATOMnet_ep0040.pth.tar')
    net_t.load_state_dict(state_dict['net'])
    net_t = net_t.cuda()
    if mode == 'eval':
        net_t = net_t.eval()
    else:
        net_t = net_t.train()

    train_feat1_t = n2t(train_feat1_np).cuda()
    train_feat2_t = n2t(train_feat2_np).cuda()
    test_feat1_t = n2t(test_feat1_np).cuda()
    test_feat2_t = n2t(test_feat2_np).cuda()
    test_proposals_t = n2t(test_proposals).cuda()
    train_bb_t = n2t(train_bb_np).cuda()

    iou_pred_t = net_t.bb_regressor([train_feat1_t, train_feat2_t],
                                    [test_feat1_t, test_feat2_t],
                                    train_bb_t,
                                    test_proposals_t)
    iou_pred_t_np = t2n(iou_pred_t)

    with fluid.dygraph.guard():
        train_feat1_p = n2p(train_feat1_np)
        train_feat2_p = n2p(train_feat2_np)
        test_feat1_p = n2p(test_feat1_np)
        test_feat2_p = n2p(test_feat2_np)
        test_proposals_p = n2p(test_proposals)
        train_bb_p = n2p(train_bb_np)

        net_p = atom_p(backbone_pretrained=False, iounet_is_test=True if mode == 'eval' else False)
        # net = build_once(net, backbone_freeze=True, iounet_freeze=False)
        state_dictsm, _ = fluid.load_dygraph('paddle_ATOMnet-ep0040')
        net_p.load_dict(state_dictsm)

        if mode == 'eval':
            net_p.eval()
        else:
            net_p.train()

        iou_pred_p = net_p.bb_regressor([train_feat1_p, train_feat2_p],
                                        [test_feat1_p, test_feat2_p],
                                        train_bb_p,
                                        test_proposals_p)
        iou_pred_p_np = p2n(iou_pred_p)

    np.testing.assert_allclose(iou_pred_p_np, iou_pred_t_np, rtol=1e-07, atol=1e-05)


def test_iounet_train_backward():
    import numpy as np
    from paddle.fluid import dygraph, layers

    # prepare data
    batch_size = 10
    num_proposals = 16
    rng = np.random.RandomState(0)
    input_dim = (512, 1024)

    train_feat1_np = rng.uniform(-1, 1, (1, batch_size, input_dim[0], 36, 36)).astype('float32')
    train_feat2_np = rng.uniform(-1, 1, (1, batch_size, input_dim[1], 18, 18)).astype('float32')

    test_feat1_np = rng.uniform(-1, 1, (1, batch_size, input_dim[0], 36, 36)).astype('float32')
    test_feat2_np = rng.uniform(-1, 1, (1, batch_size, input_dim[1], 18, 18)).astype('float32')

    rois_x1y1 = rng.uniform(0, 288, (1, batch_size, num_proposals, 2))
    rois_wh = rng.uniform(2, 144, (1, batch_size, num_proposals, 2))
    test_proposals = np.concatenate([rois_x1y1, rois_wh], axis=3).astype(np.float32)

    rois_x1y1 = rng.uniform(0, 288, (1, batch_size, 2))
    rois_wh = rng.uniform(2, 144, (1, batch_size, 2))
    train_bb_np = np.concatenate([rois_x1y1, rois_wh], axis=2).astype(np.float32)

    # compute torch forward
    net_t = atom_t(backbone_pretrained=False)
    state_dict = loading.torch_load_legacy('ATOMnet_ep0040.pth.tar')
    net_t.load_state_dict(state_dict['net'])
    net_t = net_t.cuda()
    net_t = net_t.train()

    train_feat1_t = n2t(train_feat1_np).cuda()
    train_feat2_t = n2t(train_feat2_np).cuda()
    test_feat1_t = n2t(test_feat1_np).cuda()
    test_feat2_t = n2t(test_feat2_np).cuda()
    test_proposals_t = n2t(test_proposals).cuda()
    train_bb_t = n2t(train_bb_np).cuda()

    train_feat1_t.requires_grad = True
    train_feat2_t.requires_grad = True
    test_feat1_t.requires_grad = True
    test_feat2_t.requires_grad = True
    test_proposals_t.requires_grad = True
    train_bb_t.requires_grad = True

    iou_pred_t = net_t.bb_regressor([train_feat1_t, train_feat2_t],
                                    [test_feat1_t, test_feat2_t],
                                    train_bb_t,
                                    test_proposals_t)
    iou_pred_t.mean().backward()

    train_feat1_grad_t_np = t2n(train_feat1_t.grad)
    train_feat2_grad_t_np = t2n(train_feat2_t.grad)
    test_feat1_grad_t_np = t2n(test_feat1_t.grad)
    test_feat2_grad_t_np = t2n(test_feat2_t.grad)
    test_proposals_grad_t_np = t2n(test_proposals_t.grad)
    train_bb_grad_t_np = t2n(train_bb_t.grad)

    with fluid.dygraph.guard():
        train_feat1_p = n2p(train_feat1_np)
        train_feat2_p = n2p(train_feat2_np)
        test_feat1_p = n2p(test_feat1_np)
        test_feat2_p = n2p(test_feat2_np)
        test_proposals_p = n2p(test_proposals)
        train_bb_p = n2p(train_bb_np)

        train_feat1_p.stop_gradient = False
        train_feat2_p.stop_gradient = False
        test_feat1_p.stop_gradient = False
        test_feat2_p.stop_gradient = False
        test_proposals_p.stop_gradient = False
        train_bb_p.stop_gradient = False

        net_p = atom_p(backbone_pretrained=False, iounet_is_test=False)
        # net = build_once(net, backbone_freeze=True, iounet_freeze=False)
        state_dictsm, _ = fluid.load_dygraph('paddle_ATOMnet-ep0040')
        net_p.load_dict(state_dictsm)
        net_p.train()

        iou_pred_p = net_p.bb_regressor([train_feat1_p, train_feat2_p],
                                        [test_feat1_p, test_feat2_p],
                                        train_bb_p,
                                        test_proposals_p)
        layers.reduce_mean(iou_pred_p).backward()

        train_feat1_grad_p_np = train_feat1_p.gradient()
        train_feat2_grad_p_np = train_feat2_p.gradient()
        test_feat1_grad_p_np = test_feat1_p.gradient()
        test_feat2_grad_p_np = test_feat2_p.gradient()
        test_proposals_grad_p_np = test_proposals_p.gradient()
        train_bb_grad_p_np = train_bb_p.gradient()

    np.testing.assert_allclose(train_feat1_grad_p_np, train_feat1_grad_t_np, rtol=1e-07, atol=1e-05)
    np.testing.assert_allclose(train_feat2_grad_p_np, train_feat2_grad_t_np, rtol=1e-07, atol=1e-05)
    np.testing.assert_allclose(test_feat1_grad_p_np, test_feat1_grad_t_np, rtol=1e-07, atol=1e-05)
    np.testing.assert_allclose(test_feat2_grad_p_np, test_feat2_grad_t_np, rtol=1e-07, atol=1e-05)
    np.testing.assert_allclose(test_proposals_grad_p_np, test_proposals_grad_t_np, rtol=1e-07, atol=1e-05)
    np.testing.assert_allclose(train_bb_grad_p_np, train_bb_grad_t_np, rtol=1e-07, atol=1e-05)


def test_iounet_eval_backward():
    import numpy as np
    from paddle.fluid import dygraph, layers

    # prepare data
    batch_size = 10
    num_proposals = 16
    rng = np.random.RandomState(0)
    input_dim = (512, 1024)

    train_feat1_np = rng.uniform(-1, 1, (1, batch_size, input_dim[0], 36, 36)).astype('float32')
    train_feat2_np = rng.uniform(-1, 1, (1, batch_size, input_dim[1], 18, 18)).astype('float32')

    test_feat1_np = rng.uniform(-1, 1, (1, batch_size, input_dim[0], 36, 36)).astype('float32')
    test_feat2_np = rng.uniform(-1, 1, (1, batch_size, input_dim[1], 18, 18)).astype('float32')

    rois_x1y1 = rng.uniform(0, 288, (1, batch_size, num_proposals, 2))
    rois_wh = rng.uniform(2, 144, (1, batch_size, num_proposals, 2))
    test_proposals = np.concatenate([rois_x1y1, rois_wh], axis=3).astype(np.float32)

    rois_x1y1 = rng.uniform(0, 288, (1, batch_size, 2))
    rois_wh = rng.uniform(2, 144, (1, batch_size, 2))
    train_bb_np = np.concatenate([rois_x1y1, rois_wh], axis=2).astype(np.float32)

    # compute torch forward
    net_t = atom_t(backbone_pretrained=False)
    state_dict = loading.torch_load_legacy('ATOMnet_ep0040.pth.tar')
    net_t.load_state_dict(state_dict['net'])
    net_t = net_t.cuda()
    net_t = net_t.eval()

    train_feat1_t = n2t(train_feat1_np).cuda()
    train_feat2_t = n2t(train_feat2_np).cuda()
    test_feat1_t = n2t(test_feat1_np).cuda()
    test_feat2_t = n2t(test_feat2_np).cuda()
    test_proposals_t = n2t(test_proposals).cuda()
    train_bb_t = n2t(train_bb_np).cuda()

    train_feat1_t.requires_grad = True
    train_feat2_t.requires_grad = True
    test_feat1_t.requires_grad = True
    test_feat2_t.requires_grad = True
    test_proposals_t.requires_grad = True
    train_bb_t.requires_grad = True

    iou_pred_t = net_t.bb_regressor([train_feat1_t, train_feat2_t],
                                    [test_feat1_t, test_feat2_t],
                                    train_bb_t,
                                    test_proposals_t)
    iou_pred_t.mean().backward()

    train_feat1_grad_t_np = t2n(train_feat1_t.grad)
    train_feat2_grad_t_np = t2n(train_feat2_t.grad)
    test_feat1_grad_t_np = t2n(test_feat1_t.grad)
    test_feat2_grad_t_np = t2n(test_feat2_t.grad)
    test_proposals_grad_t_np = t2n(test_proposals_t.grad)
    train_bb_grad_t_np = t2n(train_bb_t.grad)

    with fluid.dygraph.guard():
        train_feat1_p = n2p(train_feat1_np)
        train_feat2_p = n2p(train_feat2_np)
        test_feat1_p = n2p(test_feat1_np)
        test_feat2_p = n2p(test_feat2_np)
        test_proposals_p = n2p(test_proposals)
        train_bb_p = n2p(train_bb_np)

        train_feat1_p.stop_gradient = False
        train_feat2_p.stop_gradient = False
        test_feat1_p.stop_gradient = False
        test_feat2_p.stop_gradient = False
        test_proposals_p.stop_gradient = False
        train_bb_p.stop_gradient = False

        net_p = atom_p(backbone_pretrained=False, iounet_is_test=True)
        # net = build_once(net, backbone_freeze=True, iounet_freeze=False)
        state_dictsm, _ = fluid.load_dygraph('paddle_ATOMnet-ep0040')
        net_p.load_dict(state_dictsm)
        net_p.train()

        iou_pred_p = net_p.bb_regressor([train_feat1_p, train_feat2_p],
                                        [test_feat1_p, test_feat2_p],
                                        train_bb_p,
                                        test_proposals_p)
        layers.reduce_mean(iou_pred_p).backward()

        train_feat1_grad_p_np = train_feat1_p.gradient()
        train_feat2_grad_p_np = train_feat2_p.gradient()
        test_feat1_grad_p_np = test_feat1_p.gradient()
        test_feat2_grad_p_np = test_feat2_p.gradient()
        test_proposals_grad_p_np = test_proposals_p.gradient()
        train_bb_grad_p_np = train_bb_p.gradient()

    np.testing.assert_allclose(train_feat1_grad_p_np, train_feat1_grad_t_np, rtol=1e-07, atol=1e-05)
    np.testing.assert_allclose(train_feat2_grad_p_np, train_feat2_grad_t_np, rtol=1e-07, atol=1e-05)
    np.testing.assert_allclose(test_feat1_grad_p_np, test_feat1_grad_t_np, rtol=1e-07, atol=1e-05)
    np.testing.assert_allclose(test_feat2_grad_p_np, test_feat2_grad_t_np, rtol=1e-07, atol=1e-05)
    np.testing.assert_allclose(test_proposals_grad_p_np, test_proposals_grad_t_np, rtol=1e-07, atol=1e-05)
    np.testing.assert_allclose(train_bb_grad_p_np, train_bb_grad_t_np, rtol=1e-07, atol=1e-05)


def test_optimize():
    # load test data
    with open('test_data.pickle', 'rb') as f:
        data = pickle.load(f)

    def torch_optimize(data):
        net = atom_t(backbone_pretrained=False)
        state_dict = loading.torch_load_legacy('ATOMnet_ep0040.pth.tar')
        net.load_state_dict(state_dict['net'])
        net = net.cuda()
        net = net.eval()

        for k, v in data.items():
            data[k] = torch.from_numpy(v).cuda()

        train_imgs = data['train_images']
        test_imgs = data['test_images']
        train_bb = data['train_anno']
        test_proposals = data['test_proposals']

        num_sequences = train_imgs.shape[-4]
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1
        num_test_images = test_imgs.shape[0] if test_imgs.dim() == 5 else 1

        # Extract backbone features
        train_feat = net.extract_backbone_features(
            train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat = net.extract_backbone_features(
            test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        # For clarity, send the features to bb_regressor in sequenceform, i.e. [sequence, batch, feature, row, col]
        train_feat_iou = [feat.view(num_train_images, num_sequences, feat.shape[-3], feat.shape[-2], feat.shape[-1])
                          for feat in train_feat.values()]
        test_feat_iou = [feat.view(num_test_images, num_sequences, feat.shape[-3], feat.shape[-2], feat.shape[-1])
                         for feat in test_feat.values()]

        train_bb = train_bb.view(num_train_images, num_sequences, 4)
        test_proposals.view(num_train_images, num_sequences, -1, 4)

        # Get filter
        feat1 = train_feat_iou
        feat2 = test_feat_iou
        bb1 = train_bb
        proposals2 = test_proposals
        assert feat1[0].dim() == 5, 'Expect 5  dimensional feat1'

        num_test_images = feat2[0].shape[0]
        batch_size = feat2[0].shape[1]

        # Extract first train sample
        feat1 = [f[0, ...] for f in feat1]
        bb1 = bb1[0, ...]

        # Get modulation vector
        filter = net.bb_regressor.get_filter(feat1, bb1)
        feat2 = [f.view(batch_size * num_test_images, f.shape[2], f.shape[3], f.shape[4]) for f in feat2]
        iou_feat = net.bb_regressor.get_iou_feat(feat2)

        filter = [f.view(1, batch_size, -1).repeat(num_test_images, 1, 1).view(batch_size * num_test_images, -1) for f
                  in filter]
        proposals2 = proposals2.view(batch_size * num_test_images, -1, 4)

        output_boxes = proposals2
        step_length = 1
        decay = 1

        for idx in range(50):
            bb_init = output_boxes.clone().detach()
            bb_init.requires_grad = True

            outputs = net.bb_regressor.predict_iou(filter, iou_feat, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            outputs.backward(gradient=torch.ones_like(outputs), retain_graph=True)

            # Update proposal
            bb_init_np = bb_init.detach().cpu().numpy()
            bb_init_gd = bb_init.grad.cpu().numpy()
            output_boxes = bb_init_np + step_length * bb_init_gd * np.tile(bb_init_np[:, :, 2:], (1, 1, 2))
            diff = np.mean((bb_init_np - output_boxes) ** 2)
            output_boxes = torch.from_numpy(output_boxes).cuda()
            step_length *= decay
            print('Torch: {}'.format(outputs.mean()))

    def paddle_optimize(data):
        with fluid.dygraph.guard():
            net = atom_p(backbone_pretrained=False, backbone_is_test=True, iounet_is_test=True)
            state_dictsm, _ = fluid.load_dygraph('paddle_ATOMnet-ep0040')
            net.load_dict(state_dictsm)
            net.train()

        with fluid.dygraph.guard():
            train_imgs = n2p(data['train_images'])
            test_imgs = n2p(data['test_images'])
            train_bb = n2p(data['train_anno'])
            test_proposals = n2p(data['test_proposals'])

            num_sequences = train_imgs.shape[-4]
            num_train_images = train_imgs.shape[0] if len(train_imgs.shape) == 5 else 1
            num_test_images = test_imgs.shape[0] if len(test_imgs.shape) == 5 else 1

            # Extract backbone features
            if len(train_imgs.shape) == 5:
                train_imgs = fluid.layers.reshape(train_imgs, [-1, *list(train_imgs.shape)[-3:]])
                test_imgs = fluid.layers.reshape(test_imgs, [-1, *list(test_imgs.shape)[-3:]])

            train_feat = net.extract_backbone_features(train_imgs)
            test_feat = net.extract_backbone_features(test_imgs)

            # For clarity, send the features to bb_regressor in sequenceform, i.e. [sequence, batch, feature, row, col]
            train_feat_iou = [fluid.layers.reshape(feat, (num_train_images, num_sequences, *feat.shape[-3:])) for feat
                              in
                              train_feat.values()]
            test_feat_iou = [fluid.layers.reshape(feat, (num_test_images, num_sequences, *feat.shape[-3:])) for feat in
                             test_feat.values()]

            train_bb = layers.reshape(train_bb, (num_train_images, num_sequences, 4))
            test_proposals = layers.reshape(test_proposals, (num_train_images, num_sequences, -1, 4))

            # Get filter
            feat1 = train_feat_iou
            feat2 = test_feat_iou
            bb1 = train_bb
            proposals2 = test_proposals
            #     assert feat1[0].dim() == 5, 'Expect 5  dimensional feat1'

            num_test_images = feat2[0].shape[0]
            batch_size = feat2[0].shape[1]

            # Extract first train sample
            feat1 = [f[0] for f in feat1]
            bb1 = bb1[0]

            # Get modulation vector
            filter = net.bb_regressor.get_filter(feat1, bb1)
            feat2 = [layers.reshape(f, (batch_size * num_test_images, f.shape[2], f.shape[3], f.shape[4])) for f in
                     feat2]
            iou_feat = net.bb_regressor.get_iou_feat(feat2)

            new_modulation = []
            for i in range(0, len(filter)):
                tmp = filter[i]
                tmp = fluid.layers.reshape(tmp, [1, batch_size, -1])
                tmp = fluid.layers.expand(tmp, [num_test_images, 1, 1])
                tmp = fluid.layers.reshape(tmp, [batch_size * num_test_images, -1])
                new_modulation.append(tmp)

            proposals2 = fluid.layers.reshape(proposals2, [batch_size * num_test_images, -1, 4])

            output_boxes = proposals2
            step_length = 1
            decay = 1

            state_dicts = net.state_dict()
            for k in state_dicts.keys():
                if 'Resnet' in k and "running" not in k:
                    state_dicts[k].stop_gradient = True
                elif 'IOUnet' in k and "running" not in k:
                    state_dicts[k].stop_gradient = False

            for idx in range(50):
                bb_init = output_boxes
                bb_init.stop_gradient = False

                outputs = net.bb_regressor.predict_iou(filter, iou_feat, bb_init)

                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                outputs.backward()

                # Update proposal
                bb_init_np = bb_init.numpy()
                bb_init_gd = bb_init.gradient()
                output_boxes = bb_init_np + step_length * bb_init_gd * np.tile(bb_init_np[:, :, 2:], (1, 1, 2))
                step_length *= decay
                diff = np.mean((bb_init_np - output_boxes) ** 2)
                net.clear_gradients()
                output_boxes = n2p(output_boxes)
                print('Paddle: {}'.format(outputs.numpy().mean()))

    torch_optimize(data)
    paddle_optimize(data)


def test_optimize2():
    # load test data
    with open('test_data.pickle', 'rb') as f:
        data = pickle.load(f)

    def torch_optimize(data):
        net = atom_t(backbone_pretrained=False)
        state_dict = loading.torch_load_legacy('ATOMnet_ep0040.pth.tar')
        net.load_state_dict(state_dict['net'])
        net = net.cuda()
        net = net.eval()

        for k, v in data.items():
            data[k] = torch.from_numpy(v).cuda()

        train_imgs = data['train_images']
        test_imgs = data['test_images']
        train_bb = data['train_anno']
        test_proposals = data['test_proposals']

        num_sequences = train_imgs.shape[-4]
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1
        num_test_images = test_imgs.shape[0] if test_imgs.dim() == 5 else 1

        # Extract backbone features
        train_feat = net.extract_backbone_features(
            train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat = net.extract_backbone_features(
            test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        # For clarity, send the features to bb_regressor in sequenceform, i.e. [sequence, batch, feature, row, col]
        train_feat_iou = [feat.view(num_train_images, num_sequences, feat.shape[-3], feat.shape[-2], feat.shape[-1])
                          for feat in train_feat.values()]
        test_feat_iou = [feat.view(num_test_images, num_sequences, feat.shape[-3], feat.shape[-2], feat.shape[-1])
                         for feat in test_feat.values()]

        train_bb = train_bb.view(num_train_images, num_sequences, 4)
        test_proposals.view(num_train_images, num_sequences, -1, 4)

        # Get filter
        feat1 = train_feat_iou
        feat2 = test_feat_iou
        bb1 = train_bb
        proposals2 = test_proposals
        assert feat1[0].dim() == 5, 'Expect 5  dimensional feat1'

        num_test_images = feat2[0].shape[0]
        batch_size = feat2[0].shape[1]

        # Extract first train sample
        feat1 = [f[0, ...] for f in feat1]
        bb1 = bb1[0, ...]

        # Get modulation vector
        filter = net.bb_regressor.get_filter(feat1, bb1)
        feat2 = [f.view(batch_size * num_test_images, f.shape[2], f.shape[3], f.shape[4]) for f in feat2]
        iou_feat = net.bb_regressor.get_iou_feat(feat2)

        filter = [f.view(1, batch_size, -1).repeat(num_test_images, 1, 1).view(batch_size * num_test_images, -1) for f
                  in filter]
        proposals2 = proposals2.view(batch_size * num_test_images, -1, 4)

        output_boxes = proposals2
        step_length = 1
        decay = 1

        for idx in range(50):
            bb_init = output_boxes.clone().detach()
            bb_init.requires_grad = True

            outputs = net.bb_regressor.predict_iou(filter, iou_feat, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            outputs.backward(gradient=torch.ones_like(outputs), retain_graph=True)

            # Update proposal
            bb_init_np = bb_init.detach().cpu().numpy()
            bb_init_gd = bb_init.grad.cpu().numpy()
            output_boxes = bb_init_np + step_length * bb_init_gd * np.tile(bb_init_np[:, :, 2:], (1, 1, 2))
            diff = np.mean((bb_init_np - output_boxes) ** 2)
            output_boxes = torch.from_numpy(output_boxes).cuda()
            step_length *= decay
            print('Torch: {}'.format(outputs.mean()))

    def paddle_optimize(data):
        with fluid.dygraph.guard():
            net = atom_p(backbone_pretrained=False, backbone_is_test=True, iounet_is_test=True)
            state_dictsm, _ = fluid.load_dygraph('paddle_ATOMnet-ep0040')
            net.load_dict(state_dictsm)
            net.train()

        with fluid.dygraph.guard():
            train_imgs = n2p(data['train_images'])
            test_imgs = n2p(data['test_images'])
            train_bb = n2p(data['train_anno'])
            test_proposals = n2p(data['test_proposals'])

            num_sequences = train_imgs.shape[-4]
            num_train_images = train_imgs.shape[0] if len(train_imgs.shape) == 5 else 1
            num_test_images = test_imgs.shape[0] if len(test_imgs.shape) == 5 else 1

            # Extract backbone features
            if len(train_imgs.shape) == 5:
                train_imgs = fluid.layers.reshape(train_imgs, [-1, *list(train_imgs.shape)[-3:]])
                test_imgs = fluid.layers.reshape(test_imgs, [-1, *list(test_imgs.shape)[-3:]])

            train_feat = net.extract_backbone_features(train_imgs)
            test_feat = net.extract_backbone_features(test_imgs)

            # For clarity, send the features to bb_regressor in sequenceform, i.e. [sequence, batch, feature, row, col]
            train_feat_iou = [fluid.layers.reshape(feat, (num_train_images, num_sequences, *feat.shape[-3:])) for feat
                              in
                              train_feat.values()]
            test_feat_iou = [fluid.layers.reshape(feat, (num_test_images, num_sequences, *feat.shape[-3:])) for feat in
                             test_feat.values()]

            train_bb = layers.reshape(train_bb, (num_train_images, num_sequences, 4))
            test_proposals = layers.reshape(test_proposals, (num_train_images, num_sequences, -1, 4))

            # Get filter
            feat1 = train_feat_iou
            feat2 = test_feat_iou
            bb1 = train_bb
            proposals2 = test_proposals
            #     assert feat1[0].dim() == 5, 'Expect 5  dimensional feat1'

            num_test_images = feat2[0].shape[0]
            batch_size = feat2[0].shape[1]

            # Extract first train sample
            feat1 = [f[0] for f in feat1]
            bb1 = bb1[0]

            # Get modulation vector
            filter = net.bb_regressor.get_filter(feat1, bb1)

            feat2 = [layers.reshape(f, (batch_size * num_test_images, f.shape[2], f.shape[3], f.shape[4])) for f in
                     feat2]
            iou_feat = net.bb_regressor.get_iou_feat(feat2)

            filter = [p2n(f) for f in filter]
            iou_feat = [p2n(f) for f in iou_feat]
            proposals2 = p2n(proposals2)

        with fluid.dygraph.guard():
            filter = [n2p(f) for f in filter]
            iou_feat = [n2p(f) for f in iou_feat]
            proposals2 = n2p(proposals2)

            for f in iou_feat:
                f.stop_gradient = False

            new_modulation = []
            for i in range(0, len(filter)):
                tmp = filter[i]
                tmp = fluid.layers.reshape(tmp, [1, batch_size, -1])
                tmp = fluid.layers.expand(tmp, [num_test_images, 1, 1])
                tmp = fluid.layers.reshape(tmp, [batch_size * num_test_images, -1])
                new_modulation.append(tmp)

            proposals2 = fluid.layers.reshape(proposals2, [batch_size * num_test_images, -1, 4])

            output_boxes = proposals2
            step_length = 1
            decay = 1

            state_dicts = net.state_dict()
            for k in state_dicts.keys():
                if 'Resnet' in k and "running" not in k:
                    state_dicts[k].stop_gradient = True
                elif 'IOUnet' in k and "running" not in k:
                    state_dicts[k].stop_gradient = False

            for idx in range(50):
                bb_init = output_boxes
                bb_init.stop_gradient = False

                outputs = net.bb_regressor.predict_iou(filter, iou_feat, bb_init)

                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                outputs.backward()

                # Update proposal
                bb_init_np = bb_init.numpy()
                bb_init_gd = bb_init.gradient()
                output_boxes = bb_init_np + step_length * bb_init_gd * np.tile(bb_init_np[:, :, 2:], (1, 1, 2))
                step_length *= decay
                diff = np.mean((bb_init_np - output_boxes) ** 2)
                net.clear_gradients()
                output_boxes = n2p(output_boxes)
                print('Paddle: {}'.format(outputs.numpy().mean()))

    torch_optimize(data)
    paddle_optimize(data)


if __name__ == '__main__':
    # unittest.main()
    # test_iounet_forward(mode='eval')
    # test_iounet_forward(mode='train')
    # test_iounet_train_backward()
    # test_iounet_eval_backward()
    # test_optimize()
    test_optimize2()

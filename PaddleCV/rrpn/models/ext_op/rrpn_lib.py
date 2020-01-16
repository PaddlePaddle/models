#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import Variable
fluid.load_op_library('models/ext_op/src/rrpn_lib.so')


def rrpn_target_assign(bbox_pred,
                       cls_logits,
                       anchor_box,
                       gt_boxes,
                       im_info,
                       rpn_batch_size_per_im=256,
                       rpn_straddle_thresh=0.0,
                       rpn_fg_fraction=0.5,
                       rpn_positive_overlap=0.7,
                       rpn_negative_overlap=0.3,
                       use_random=True):
    """
    **Target Assign Layer for rotated region proposal network (RRPN).**
    This layer can be, for given the  Intersection-over-Union (IoU) overlap
    between anchors and ground truth boxes, to assign classification and
    regression targets to each each anchor, these target labels are used for
    train RPN. The classification targets is a binary class label (of being
    an object or not). Following the paper of RRPN, the positive labels
    are two kinds of anchors: (i) the anchor/anchors with the highest IoU
    overlap with a ground-truth box, or (ii) an anchor that has an IoU overlap
    higher than rpn_positive_overlap(0.7) with any ground-truth box. Note
    that a single ground-truth box may assign positive labels to multiple
    anchors. A non-positive anchor is when its IoU ratio is lower than
    rpn_negative_overlap (0.3) for all ground-truth boxes. Anchors that are
    neither positive nor negative do not contribute to the training objective.
    The regression targets are the encoded ground-truth boxes associated with
    the positive anchors.
    Args:
        bbox_pred(Variable): A 3-D Tensor with shape [N, M, 5] represents the
            predicted locations of M bounding bboxes. N is the batch size,
            and each bounding box has five coordinate values and the layout
            is [x, y, w, h, angle]. The data type can be float32 or float64.
        cls_logits(Variable): A 3-D Tensor with shape [N, M, 1] represents the
            predicted confidence predictions. N is the batch size, 1 is the
            frontground and background sigmoid, M is number of bounding boxes.
            The data type can be float32 or float64.
        anchor_box(Variable): A 2-D Tensor with shape [M, 5] holds M boxes,
            each box is represented as [x, y, w, h, angle],
            [x, y] is the left top coordinate of the anchor box,
            if the input is image feature map, they are close to the origin
            of the coordinate system. [w, h] is the right bottom
            coordinate of the anchor box, angle is the rotation angle of box.
            The data type can be float32 or float64.
        gt_boxes (Variable): The ground-truth bounding boxes (bboxes) are a 2D
            LoDTensor with shape [Ng, 5], Ng is the total number of ground-truth
            bboxes of mini-batch input. The data type can be float32 or float64.
        im_info (Variable): A 2-D LoDTensor with shape [N, 3]. N is the batch size,
        3 is the height, width and scale.
        rpn_batch_size_per_im(int): Total number of RPN examples per image.
                                    The data type must be int32.
        rpn_straddle_thresh(float): Remove RPN anchors that go outside the image
            by straddle_thresh pixels. The data type must be float32.
        rpn_fg_fraction(float): Target fraction of RoI minibatch that is labeled
            foreground (i.e. class > 0), 0-th class is background. The data type must be float32.
        rpn_positive_overlap(float): Minimum overlap required between an anchor
            and ground-truth box for the (anchor, gt box) pair to be a positive
            example. The data type must be float32.
        rpn_negative_overlap(float): Maximum overlap allowed between an anchor
            and ground-truth box for the (anchor, gt box) pair to be a negative
            examples. The data type must be float32.
        use_random(bool): Whether to sample randomly when sampling.
    Returns:
        tuple:
        A tuple(predicted_scores, predicted_location, target_label,
        target_bbox) is returned. The predicted_scores 
        and predicted_location is the predicted result of the RPN.
        The target_label and target_bbox is the ground truth,
        respectively. The predicted_location is a 2D Tensor with shape
        [F, 5], and the shape of target_bbox is same as the shape of
        the predicted_location, F is the number of the foreground
        anchors. The predicted_scores is a 2D Tensor with shape
        [F + B, 1], and the shape of target_label is same as the shape
        of the predicted_scores, B is the number of the background
        anchors, the F and B is depends on the input of this operator.
        Bbox_inside_weight represents whether the predicted loc is fake_fg
        or not and the shape is [F, 5].
    Examples:
        .. code-block:: python
            import paddle.fluid as fluid
            bbox_pred = fluid.data(name='bbox_pred', shape=[None, 5], dtype='float32')
            cls_logits = fluid.data(name='cls_logits', shape=[None, 1], dtype='float32')
            anchor_box = fluid.data(name='anchor_box', shape=[None, 5], dtype='float32')
            gt_boxes = fluid.data(name='gt_boxes', shape=[None, 5], dtype='float32')
            im_info = fluid.data(name='im_infoss', shape=[None, 3], dtype='float32')
            loc, score, loc_target, score_target = rrpn_target_assign(
                bbox_pred, cls_logits, anchor_box, gt_boxes, im_info)
    """

    helper = LayerHelper('rrpn_target_assign', **locals())
    # Assign target label to anchors
    loc_index = helper.create_variable_for_type_inference(dtype='int32')
    score_index = helper.create_variable_for_type_inference(dtype='int32')
    target_label = helper.create_variable_for_type_inference(dtype='int32')
    target_bbox = helper.create_variable_for_type_inference(
        dtype=anchor_box.dtype)
    helper.append_op(
        type="rrpn_target_assign",
        inputs={'Anchor': anchor_box,
                'GtBoxes': gt_boxes,
                'ImInfo': im_info},
        outputs={
            'LocationIndex': loc_index,
            'ScoreIndex': score_index,
            'TargetLabel': target_label,
            'TargetBBox': target_bbox
        },
        attrs={
            'rpn_batch_size_per_im': rpn_batch_size_per_im,
            'rpn_straddle_thresh': rpn_straddle_thresh,
            'rpn_positive_overlap': rpn_positive_overlap,
            'rpn_negative_overlap': rpn_negative_overlap,
            'rpn_fg_fraction': rpn_fg_fraction,
            'use_random': use_random
        })

    loc_index.stop_gradient = True
    score_index.stop_gradient = True
    target_label.stop_gradient = True
    target_bbox.stop_gradient = True

    cls_logits = fluid.layers.reshape(x=cls_logits, shape=(-1, 1))
    bbox_pred = fluid.layers.reshape(x=bbox_pred, shape=(-1, 5))
    predicted_cls_logits = fluid.layers.gather(cls_logits, score_index)
    predicted_bbox_pred = fluid.layers.gather(bbox_pred, loc_index)

    return predicted_cls_logits, predicted_bbox_pred, target_label, target_bbox


def rotated_anchor_generator(input,
                             anchor_sizes=None,
                             aspect_ratios=None,
                             angles=None,
                             variance=[1.0, 1.0, 1.0, 1.0, 1.0],
                             stride=None,
                             offset=0.5,
                             name=None):
    """
    **Rotated Anchor generator operator**
    Generate anchors for RRPN algorithm.
    Each position of the input produce N anchors, N =
    size(anchor_sizes) * size(aspect_ratios) * size(angles).
    The order of generated anchors is firstly aspect_ratios
    loop then anchor_sizes loop.
    Args:
       input(Variable): 4-D Tensor with shape [N,C,H,W]. The input feature map.
       anchor_sizes(float32|list|tuple): The anchor sizes of generated
          anchors, given in absolute pixels e.g. [64., 128., 256., 512.].
          For instance, the anchor size of 64 means the area of this anchor 
          equals to 64**2. None by default.
       aspect_ratios(float32|list|tuple): The height / width ratios 
           of generated anchors, e.g. [0.5, 1.0, 2.0]. None by default.
       angle(list|tuple): Rotated angle of prior boxes. The data type is float32.
       variance(list|tuple): The variances to be used in box 
           regression deltas. The data type is float32, [1.0, 1.0, 1.0, 1.0, 1.0] by 
           default.
       stride(list|tuple): The anchors stride across width and height.
           The data type is float32. e.g. [16.0, 16.0]. None by default.
       offset(float32): Prior boxes center offset. 0.5 by default.
       name(str): Name of this layer. None by default. 
    Returns:
       Anchors(Variable): The output anchors with a layout of [H, W, num_anchors, 5].
                          H is the height of input, W is the width of input,
                          num_anchors is the box count of each position. Each anchor is
                          in (x, y, w, h, angle) format.
       Variances(Variable): The expanded variances of anchors with a layout of
                            [H, W, num_priors, 5]. H is the height of input,
                            W is the width of input num_anchors is the box count
                            of each position. Each variance is in (x, y, w, h, angle) format.
    Examples:
        .. code-block:: python
            import paddle.fluid as fluid
            conv1 = fluid.data(name='conv1', shape=[None, 48, 16, 16], dtype='float32')
            anchor, var = rotated_anchor_generator(
                input=conv1,
                anchor_sizes=[128, 256, 512],
                aspect_ratios=[0.2, 0.5, 1.0],
                variance=[1.0, 1.0, 1.0, 1.0, 1.0],
                stride=[16.0, 16.0],
                offset=0.5)
    """
    helper = LayerHelper("rotated_anchor_generator", **locals())
    dtype = helper.input_dtype()

    def _is_list_or_tuple_(data):
        return (isinstance(data, list) or isinstance(data, tuple))

    if not _is_list_or_tuple_(anchor_sizes):
        anchor_sizes = [anchor_sizes]
    if not _is_list_or_tuple_(aspect_ratios):
        aspect_ratios = [aspect_ratios]
    if not _is_list_or_tuple_(angles):
        angles = [angles]
    if not (_is_list_or_tuple_(stride) and len(stride) == 2):
        raise ValueError('stride should be a list or tuple ',
                         'with length 2, (stride_width, stride_height).')

    anchor_sizes = list(map(float, anchor_sizes))
    aspect_ratios = list(map(float, aspect_ratios))
    angles = list(map(float, angles))
    stride = list(map(float, stride))

    attrs = {
        'anchor_sizes': anchor_sizes,
        'aspect_ratios': aspect_ratios,
        'angles': angles,
        'variances': variance,
        'stride': stride,
        'offset': offset
    }

    anchor = helper.create_variable_for_type_inference(dtype)
    var = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="rotated_anchor_generator",
        inputs={"Input": input},
        outputs={"Anchors": anchor,
                 "Variances": var},
        attrs=attrs, )
    anchor.stop_gradient = True
    var.stop_gradient = True
    return anchor, var


def rrpn_box_coder(prior_box, prior_box_var, target_box, name=None):
    """
    Args:
        prior_box(Variable): Box list prior_box is a 2-D Tensor with shape 
            [M, 5] holds M boxes and data type is float32 or float64. Each box
            is represented as [x, y, w, h, angle], [x, y] is the 
            center coordinate of the anchor box, [w, h] is the width and height
            of the anchor box, angle is rotated angle of prior_box.
        prior_box_var(List|Variable|None): "prior_box_var is a 2-D Tensor with
             shape [M, 5] holds M group of variance."
        target_box(Variable): This input can be a 2-D LoDTensor with shape 
            [M, 5]. Each box is represented as [x, y, w, h, angle]. The data
            type is float32 or float64.
        name(str): Name of this layer. None by default. 
    Returns:
        Variable:
        output_box(Variable): The output tensor of rrpn_box_coder_op with shape [N, 5] representing the 
        result of N target boxes encoded with N Prior boxes and variances. 
        N represents the number of box and 5 represents [x, y, w, h ,angle].
    Examples:
 
        .. code-block:: python
 
            import paddle.fluid as fluid
            prior_box_decode = fluid.data(name='prior_box_decode',
                                          shape=[512, 5],
                                          dtype='float32')
            target_box_decode = fluid.data(name='target_box_decode',
                                           shape=[512, 5],
                                           dtype='float32')
            output_decode = rrpn_box_coder(prior_box=prior_box_decode,
                                           prior_box_var=[10, 10, 5, 5, 1],
                                           target_box=target_box_decode)
    """

    helper = LayerHelper("rrpn_box_coder", **locals())

    if name is None:
        output_box = helper.create_variable_for_type_inference(
            dtype=prior_box.dtype)
    else:
        output_box = helper.create_variable(
            name=name, dtype=prior_box.dtype, persistable=False)

    inputs = {"PriorBox": prior_box, "TargetBox": target_box}
    attrs = {}
    if isinstance(prior_box_var, Variable):
        inputs['PriorBoxVar'] = prior_box_var
    elif isinstance(prior_box_var, list):
        attrs['variance'] = prior_box_var
    else:
        raise TypeError(
            "Input variance of rrpn_box_coder must be Variable or list")
    helper.append_op(
        type="rrpn_box_coder",
        inputs=inputs,
        attrs=attrs,
        outputs={"OutputBox": output_box})
    return output_box


def rotated_roi_align(input,
                      rois,
                      pooled_height=1,
                      pooled_width=1,
                      spatial_scale=1.0,
                      name=None):
    """
    **RotatedRoIAlign Operator**

    Rotated Region of interest align (also known as Rotated RoI align) is to perform
    bilinear interpolation on inputs of nonuniform sizes to obtain 
    fixed-size feature maps (e.g. 7*7)
    
    Dividing each region proposal into equal-sized sections with
    the pooled_width and pooled_height. Location remains the origin
    result.
    
    Each ROI bin are transformed to become horizontal by perspective transformation and
    values in each ROI bin are computed directly through bilinear interpolation. The output is
    the mean of all values.
    Thus avoid the misaligned problem.  
    """
    helper = LayerHelper('rrpn_rotated_roi_align', **locals())
    dtype = helper.input_dtype()
    align_out = helper.create_variable_for_type_inference(dtype)
    cx = helper.create_variable_for_type_inference('float32')
    cy = helper.create_variable_for_type_inference('float32')
    helper.append_op(
        type="rrpn_rotated_roi_align",
        inputs={"X": input,
                "ROIs": rois},
        outputs={"Out": align_out,
                 "ConIdX": cx,
                 "ConIdY": cy},
        attrs={
            "pooled_height": pooled_height,
            "pooled_width": pooled_width,
            "spatial_scale": spatial_scale,
        })
    return align_out


def rotated_generate_proposal_labels(rpn_rois,
                                     gt_classes,
                                     is_crowd,
                                     gt_boxes,
                                     im_info,
                                     batch_size_per_im=256,
                                     fg_fraction=0.25,
                                     fg_thresh=0.25,
                                     bg_thresh_hi=0.5,
                                     bg_thresh_lo=0.0,
                                     bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
                                     class_nums=None,
                                     use_random=True,
                                     is_cls_agnostic=False):
    """
    **Rotated Generate Proposal Labels**
    This operator can be, for given the RotatedGenerateProposalOp output bounding boxes and groundtruth,
    to sample foreground boxes and background boxes, and compute loss target.
    RpnRois is the output boxes of RPN and was processed by rotated_generate_proposal_op, these boxes
    were combined with groundtruth boxes and sampled according to batch_size_per_im and fg_fraction,
    If an instance with a groundtruth overlap greater than fg_thresh, then it was considered as a foreground sample.
    If an instance with a groundtruth overlap greater than bg_thresh_lo and lower than bg_thresh_hi,
    then it was considered as a background sample.
    After all foreground and background boxes are chosen (so called Rois),
    then we apply random sampling to make sure
    the number of foreground boxes is no more than batch_size_per_im * fg_fraction.
    For each box in Rois, we assign the classification (class label) and regression targets (box label) to it.
    Finally BboxInsideWeights and BboxOutsideWeights are used to specify whether it would contribute to training loss.
    Args:
        rpn_rois(Variable): A 2-D LoDTensor with shape [N, 5]. N is the number of the RotatedGenerateProposalOp's output, each element is a bounding box with [x, y, w, h, angle] format. The data type can be float32 or float64.
        gt_classes(Variable): A 2-D LoDTensor with shape [M, 1]. M is the number of groundtruth, each element is a class label of groundtruth. The data type must be int32.
        is_crowd(Variable): A 2-D LoDTensor with shape [M, 1]. M is the number of groundtruth, each element is a flag indicates whether a groundtruth is crowd. The data type must be int32.
        gt_boxes(Variable): A 2-D LoDTensor with shape [M, 5]. M is the number of groundtruth, each element is a bounding box with [x, y, w, h, angle] format.
        im_info(Variable): A 2-D LoDTensor with shape [B, 3]. B is the number of input images, each element consists of im_height, im_width, im_scale.
        batch_size_per_im(int): Batch size of rois per images. The data type must be int32.
        fg_fraction(float): Foreground fraction in total batch_size_per_im. The data type must be float32.
        fg_thresh(float): Overlap threshold which is used to chose foreground sample. The data type must be float32.
        bg_thresh_hi(float): Overlap threshold upper bound which is used to chose background sample. The data type must be float32.
        bg_thresh_lo(float): Overlap threshold lower bound which is used to chose background sample. The data type must be float32.
        bbox_reg_weights(list|tuple): Box regression weights. The data type must be float32.
        class_nums(int): Class number. The data type must be int32.
        use_random(bool): Use random sampling to choose foreground and background boxes.
        is_cls_agnostic(bool): bbox regression use class agnostic simply which only represent fg and bg boxes.
    Returns:
        tuple:
        A tuple with format``(rois, labels_int32, bbox_targets, bbox_inside_weights, bbox_outside_weights)``.
        - **rois**: 2-D LoDTensor with shape ``[batch_size_per_im * batch_size, 5]``. The data type is the same as ``rpn_rois``.
        - **labels_int32**: 2-D LoDTensor with shape ``[batch_size_per_im * batch_size, 1]``. The data type must be int32.
        - **bbox_targets**: 2-D LoDTensor with shape ``[batch_size_per_im * batch_size, 5 * class_num]``. The regression targets of all RoIs. The data type is the same as ``rpn_rois``.
        - **bbox_inside_weights**: 2-D LoDTensor with shape ``[batch_size_per_im * batch_size, 5 * class_num]``. The weights of foreground boxes' regression loss. The data type is the same as ``rpn_rois``.
        - **bbox_outside_weights**: 2-D LoDTensor with shape ``[batch_size_per_im * batch_size, 5 * class_num]``. The weights of regression loss. The data type is the same as ``rpn_rois``.
    Examples:
        .. code-block:: python
            import paddle.fluid as fluid
            rpn_rois = fluid.data(name='rpn_rois', shape=[None, 5], dtype='float32')
            gt_classes = fluid.data(name='gt_classes', shape=[None, 1], dtype='float32')
            is_crowd = fluid.data(name='is_crowd', shape=[None, 1], dtype='float32')
            gt_boxes = fluid.data(name='gt_boxes', shape=[None, 5], dtype='float32')
            im_info = fluid.data(name='im_info', shape=[None, 3], dtype='float32')
            rois, labels, bbox, inside_weights, outside_weights = rotated_generate_proposal_labels(
                           rpn_rois, gt_classes, is_crowd, gt_boxes, im_info,
                           class_nums=10)
    """
    helper = LayerHelper('rrpn_generate_proposal_labels', **locals())
    rois = helper.create_variable_for_type_inference(dtype=rpn_rois.dtype)
    labels_int32 = helper.create_variable_for_type_inference(
        dtype=gt_classes.dtype)
    bbox_targets = helper.create_variable_for_type_inference(
        dtype=rpn_rois.dtype)
    bbox_inside_weights = helper.create_variable_for_type_inference(
        dtype=rpn_rois.dtype)
    bbox_outside_weights = helper.create_variable_for_type_inference(
        dtype=rpn_rois.dtype)

    helper.append_op(
        type="rrpn_generate_proposal_labels",
        inputs={
            'RpnRois': rpn_rois,
            'GtClasses': gt_classes,
            'IsCrowd': is_crowd,
            'GtBoxes': gt_boxes,
            'ImInfo': im_info
        },
        outputs={
            'Rois': rois,
            'LabelsInt32': labels_int32,
            'BboxTargets': bbox_targets,
            'BboxInsideWeights': bbox_inside_weights,
            'BboxOutsideWeights': bbox_outside_weights
        },
        attrs={
            'batch_size_per_im': batch_size_per_im,
            'fg_fraction': fg_fraction,
            'fg_thresh': fg_thresh,
            'bg_thresh_hi': bg_thresh_hi,
            'bg_thresh_lo': bg_thresh_lo,
            'bbox_reg_weights': bbox_reg_weights,
            'class_nums': class_nums,
            'use_random': use_random,
            'is_cls_agnostic': is_cls_agnostic
        })

    rois.stop_gradient = True
    labels_int32.stop_gradient = True
    bbox_targets.stop_gradient = True
    bbox_inside_weights.stop_gradient = True
    bbox_outside_weights.stop_gradient = True

    return rois, labels_int32, bbox_targets, bbox_inside_weights, bbox_outside_weights


def rotated_generate_proposals(scores,
                               bbox_deltas,
                               im_info,
                               anchors,
                               variances,
                               pre_nms_top_n=6000,
                               post_nms_top_n=1000,
                               nms_thresh=0.5,
                               min_size=0.1,
                               name=None):
    """
    **Rotated Generate proposal**
    This operation proposes Rotated RoIs according to each box with their
    probability to be a foreground object and the box can be calculated by anchors.
    bbox_deltas and scores are the output of RPN. Final proposals could be used to
    train detection net. For generating proposals, this operation performs following steps:
    1. Transposes and resizes scores and bbox_deltas in size of
       (H*W*A, 1) and (H*W*A, 5)
    2. Calculate box locations as proposals candidates. 
    3. Remove predicted boxes with small area. 
    4. Apply NMS to get final proposals as output.
    Args:
        scores(Variable): A 4-D Tensor with shape [N, A, H, W] represents
            the probability for each box to be an object.
            N is batch size, A is number of anchors, H and W are height and
            width of the feature map. The data type must be float32.
        bbox_deltas(Variable): A 4-D Tensor with shape [N, 5*A, H, W]
            represents the differece between predicted box locatoin and
            anchor location. The data type must be float32.
        im_info(Variable): A 2-D Tensor with shape [N, 3] represents origin
            image information for N batch. Info contains height, width and scale
            between origin image size and the size of feature map.
            The data type must be int32.
        anchors(Variable):   A 4-D Tensor represents the anchors with a layout
            of [H, W, A, 5]. H and W are height and width of the feature map,
            num_anchors is the box count of each position. Each anchor is
            in (x, y, w, h, angle) format. The data type must be float32.
        variances(Variable): A 4-D Tensor. The expanded variances of anchors with a layout of
            [H, W, num_priors, 5]. Each variance is in
            (xcenter, ycenter, w, h) format. The data type must be float32.
        pre_nms_top_n(float): Number of total bboxes to be kept per
            image before NMS. The data type must be float32. `6000` by default.
        post_nms_top_n(float): Number of total bboxes to be kept per
            image after NMS. The data type must be float32. `1000` by default.
        nms_thresh(float): Threshold in NMS. The data type must be float32. `0.5` by default.
        min_size(float): Remove predicted boxes with either height or
            width < min_size. The data type must be float32. `0.1` by default.
    Returns:
        tuple:
        A tuple with format ``(rrpn_rois, rrpn_roi_probs)``.
        - **rpn_rois**: The generated RoIs. 2-D Tensor with shape ``[N, 5]`` while ``N`` is the number of RoIs. The data type is the same as ``scores``.
        - **rpn_roi_probs**: The scores of generated RoIs. 2-D Tensor with shape ``[N, 1]`` while ``N`` is the number of RoIs. The data type is the same as ``scores``.
    Examples:
        .. code-block:: python
        
            import paddle.fluid as fluid
            scores = fluid.data(name='scores', shape=[None, 4, 5, 5], dtype='float32')
            bbox_deltas = fluid.data(name='bbox_deltas', shape=[None, 20, 5, 5], dtype='float32')
            im_info = fluid.data(name='im_info', shape=[None, 3], dtype='float32')
            anchors = fluid.data(name='anchors', shape=[None, 5, 4, 5], dtype='float32')
            variances = fluid.data(name='variances', shape=[None, 5, 10, 5], dtype='float32')
            rrois, rroi_probs = fluid.layers.rotated_generate_proposals(scores, bbox_deltas,
                         im_info, anchors, variances)
    """

    helper = LayerHelper('rrpn_generate_proposals', **locals())

    rpn_rois = helper.create_variable_for_type_inference(
        dtype=bbox_deltas.dtype)
    rpn_roi_probs = helper.create_variable_for_type_inference(
        dtype=scores.dtype)
    helper.append_op(
        type="rrpn_generate_proposals",
        inputs={
            'Scores': scores,
            'BboxDeltas': bbox_deltas,
            'ImInfo': im_info,
            'Anchors': anchors,
            'Variances': variances
        },
        attrs={
            'pre_nms_topN': pre_nms_top_n,
            'post_nms_topN': post_nms_top_n,
            'nms_thresh': nms_thresh,
            'min_size': min_size
        },
        outputs={'RpnRois': rpn_rois,
                 'RpnRoiProbs': rpn_roi_probs})
    rpn_rois.stop_gradient = True
    rpn_roi_probs.stop_gradient = True

    return rpn_rois, rpn_roi_probs

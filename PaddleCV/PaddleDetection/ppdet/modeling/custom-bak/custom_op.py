import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import Variable
fluid.load_op_library('/paddle/PaddleDetection/ppdet/modeling/lib.so')

#fluid.load_op_library('./rrpn_box_coder_op.so')


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


def rrpn_box_coder(prior_box,
                   prior_box_var,
                   target_box,
                   code_type="encode_center_size",
                   name=None):
    #box_normalized=True,
    #name=None,
    #axis=0):

    helper = LayerHelper("rrpn_box_coder", **locals())

    if name is None:
        output_box = helper.create_variable_for_type_inference(
            dtype=prior_box.dtype)
    else:
        output_box = helper.create_variable(
            name=name, dtype=prior_box.dtype, persistable=False)

    inputs = {"PriorBox": prior_box, "TargetBox": target_box}
    attrs = {
        "code_type": code_type  #,
        #"box_normalized": box_normalized,
        #"axis": axis
    }
    if isinstance(prior_box_var, Variable):
        inputs['PriorBoxVar'] = prior_box_var
    elif isinstance(prior_box_var, list):
        attrs['variance'] = prior_box_var
    else:
        raise TypeError("Input variance of box_coder must be Variable or lisz")
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
                                     is_cls_agnostic=False,
                                     is_cascade_rcnn=False):

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
            'is_cls_agnostic': is_cls_agnostic,
            'is_cascade_rcnn': is_cascade_rcnn
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
                               eta=1.0,
                               name=None):

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
            'min_size': min_size,
            'eta': eta
        },
        outputs={'RpnRois': rpn_rois,
                 'RpnRoiProbs': rpn_roi_probs})
    rpn_rois.stop_gradient = True
    rpn_roi_probs.stop_gradient = True

    return rpn_rois, rpn_roi_probs

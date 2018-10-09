"""
This Module provide different kinds of Match layers
"""

import paddle.fluid as fluid
import paddle.v2 as paddle


def MultiPerspectiveMatching(vec1, vec2, perspective_num):
    """
    MultiPerspectiveMatching
    """
    sim_res = None
    for i in range(perspective_num):
        vec1_res = fluid.layers.elementwise_add_with_weight(
                       vec1,
                       param_attr="elementwise_add_with_weight." + str(i))
        vec2_res = fluid.layers.elementwise_add_with_weight(
                       vec2,
                       param_attr="elementwise_add_with_weight." + str(i))
        m = fluid.layers.cos_sim(vec1_res, vec2_res)
        if sim_res is None:
            sim_res = m
        else:
            sim_res = fluid.layers.concat(input=[sim_res, m], axis=1)
    return sim_res


def ConcateMatching(vec1, vec2):
    """
    ConcateMatching
    """
    #TODO: assert shape
    return fluid.layers.concat(input=[vec1, vec2], axis=1)


def ElementwiseMatching(vec1, vec2):
    """
    reference: [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364)
    """
    elementwise_mul = fluid.layers.elementwise_mul(x=vec1, y=vec2)
    elementwise_sub = fluid.layers.elementwise_sub(x=vec1, y=vec2) 
    elementwise_abs_sub = fluid.layers.abs(elementwise_sub)
    return fluid.layers.concat(input=[vec1, vec2, elementwise_mul, elementwise_abs_sub], axis=1)


def MultiPerspectiveFullMatching(seq1, seq2, perspective_num):
    """
    seq1: Lod tensor with shape [-1, feature_dim] (lod level == 1) is a representation of a sentence.
    seq2: Another Lod tensor with shape [-1, feature_dim] (lod level == 1) is a representation of a sentence.
    use seq1 to match seq2
    return match seq with same shape as seq1.
    """
    print seq2
    seq2_last = fluid.layers.sequence_pool(input=seq2, pool_type="last")
    print seq2_last
    seq2 = fluid.layers.sequence_expand(seq2_last, seq1)
    print seq2
    #seq2 = fluid.layers.lod_reset(x=seq2, y=seq1)
    seq2.set_lod(seq1)
    print seq2

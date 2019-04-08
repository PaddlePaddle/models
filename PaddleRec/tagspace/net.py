import paddle.fluid as fluid
import paddle.fluid.layers.nn as nn
import paddle.fluid.layers.tensor as tensor
import paddle.fluid.layers.control_flow as cf
import paddle.fluid.layers.io as io

def network(vocab_text_size, vocab_tag_size, emb_dim=10, hid_dim=1000, win_size=5, margin=0.1, neg_size=5):
    """ network definition """
    text = io.data(name="text", shape=[1], lod_level=1, dtype='int64')
    pos_tag = io.data(name="pos_tag", shape=[1], lod_level=1, dtype='int64')
    neg_tag = io.data(name="neg_tag", shape=[1], lod_level=1, dtype='int64')
    text_emb = nn.embedding(
            input=text, size=[vocab_text_size, emb_dim], param_attr="text_emb")
    pos_tag_emb = nn.embedding(
            input=pos_tag, size=[vocab_tag_size, emb_dim], param_attr="tag_emb")
    neg_tag_emb = nn.embedding(
            input=neg_tag, size=[vocab_tag_size, emb_dim], param_attr="tag_emb")

    conv_1d = fluid.nets.sequence_conv_pool(
            input=text_emb,
            num_filters=hid_dim,
            filter_size=win_size,
            act="tanh",
            pool_type="max",
            param_attr="cnn")
    text_hid = fluid.layers.fc(input=conv_1d, size=emb_dim, param_attr="text_hid")
    cos_pos = nn.cos_sim(pos_tag_emb, text_hid)
    mul_text_hid = fluid.layers.sequence_expand_as(x=text_hid, y=neg_tag_emb)
    mul_cos_neg = nn.cos_sim(neg_tag_emb, mul_text_hid)
    cos_neg_all = fluid.layers.sequence_reshape(input=mul_cos_neg, new_dim=neg_size)
    #choose max negtive cosine
    cos_neg = nn.reduce_max(cos_neg_all, dim=1, keep_dim=True)
    #calculate hinge loss
    loss_part1 = nn.elementwise_sub(
            tensor.fill_constant_batch_size_like(
                input=cos_pos,
                shape=[-1, 1],
                value=margin,
                dtype='float32'),
            cos_pos)
    loss_part2 = nn.elementwise_add(loss_part1, cos_neg)
    loss_part3 = nn.elementwise_max(
        tensor.fill_constant_batch_size_like(
            input=loss_part2, shape=[-1, 1], value=0.0, dtype='float32'),
        loss_part2)
    avg_cost = nn.mean(loss_part3)
    less = tensor.cast(cf.less_than(cos_neg, cos_pos), dtype='float32')
    correct = nn.reduce_sum(less)
    return avg_cost, correct, cos_pos

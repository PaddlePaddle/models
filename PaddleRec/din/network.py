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

def din_attention(hist, target_expand, max_len, mask):
    """activation weight"""

    hidden_size = hist.shape[-1]

    concat = fluid.layers.concat(
        [hist, target_expand, hist - target_expand, hist * target_expand],
        axis=2)
    atten_fc1 = fluid.layers.fc(name="atten_fc1",
                                input=concat,
                                size=80,
                                act="sigmoid",
                                num_flatten_dims=2)
    atten_fc2 = fluid.layers.fc(name="atten_fc2",
                                input=atten_fc1,
                                size=40,
                                act="sigmoid",
                                num_flatten_dims=2)
    atten_fc3 = fluid.layers.fc(name="atten_fc3",
                                input=atten_fc2,
                                size=1,
                                num_flatten_dims=2)
    atten_fc3 += mask
    atten_fc3 = fluid.layers.transpose(x=atten_fc3, perm=[0, 2, 1])
    atten_fc3 = fluid.layers.scale(x=atten_fc3, scale=hidden_size**-0.5)
    weight = fluid.layers.softmax(atten_fc3)
    out = fluid.layers.matmul(weight, hist)
    out = fluid.layers.reshape(x=out, shape=[0, hidden_size])
    return out


def network(item_count, cat_count, max_len):
    """network definition"""

    item_emb_size = 64
    cat_emb_size = 64
    is_sparse = False
    #significant for speeding up the training process

    item_emb_attr = fluid.ParamAttr(name="item_emb")
    cat_emb_attr = fluid.ParamAttr(name="cat_emb")

    hist_item_seq = fluid.layers.data(
        name="hist_item_seq", shape=[max_len, 1], dtype="int64")
    hist_cat_seq = fluid.layers.data(
        name="hist_cat_seq", shape=[max_len, 1], dtype="int64")
    target_item = fluid.layers.data(
        name="target_item", shape=[1], dtype="int64")
    target_cat = fluid.layers.data(
        name="target_cat", shape=[1], dtype="int64")
    label = fluid.layers.data(
        name="label", shape=[1], dtype="float32")
    mask = fluid.layers.data(
        name="mask", shape=[max_len, 1], dtype="float32")
    target_item_seq = fluid.layers.data(
        name="target_item_seq", shape=[max_len, 1], dtype="int64")
    target_cat_seq = fluid.layers.data(
        name="target_cat_seq", shape=[max_len, 1], dtype="int64", lod_level=0)

    hist_item_emb = fluid.layers.embedding(
        input=hist_item_seq,
        size=[item_count, item_emb_size],
        param_attr=item_emb_attr,
        is_sparse=is_sparse)

    hist_cat_emb = fluid.layers.embedding(
        input=hist_cat_seq,
        size=[cat_count, cat_emb_size],
        param_attr=cat_emb_attr,
        is_sparse=is_sparse)

    target_item_emb = fluid.layers.embedding(
        input=target_item,
        size=[item_count, item_emb_size],
        param_attr=item_emb_attr,
        is_sparse=is_sparse)

    target_cat_emb = fluid.layers.embedding(
        input=target_cat,
        size=[cat_count, cat_emb_size],
        param_attr=cat_emb_attr,
        is_sparse=is_sparse)

    target_item_seq_emb = fluid.layers.embedding(
        input=target_item_seq,
        size=[item_count, item_emb_size],
        param_attr=item_emb_attr,
        is_sparse=is_sparse)

    target_cat_seq_emb = fluid.layers.embedding(
        input=target_cat_seq,
        size=[cat_count, cat_emb_size],
        param_attr=cat_emb_attr,
        is_sparse=is_sparse)

    item_b = fluid.layers.embedding(
        input=target_item,
        size=[item_count, 1],
        param_attr=fluid.initializer.Constant(value=0.0))

    hist_seq_concat = fluid.layers.concat([hist_item_emb, hist_cat_emb], axis=2)
    target_seq_concat = fluid.layers.concat(
        [target_item_seq_emb, target_cat_seq_emb], axis=2)
    target_concat = fluid.layers.concat(
        [target_item_emb, target_cat_emb], axis=1)

    out = din_attention(hist_seq_concat, target_seq_concat, max_len, mask)
    out_fc = fluid.layers.fc(name="out_fc",
                             input=out,
                             size=item_emb_size + cat_emb_size,
                             num_flatten_dims=1)
    embedding_concat = fluid.layers.concat([out_fc, target_concat], axis=1)

    fc1 = fluid.layers.fc(name="fc1",
                          input=embedding_concat,
                          size=80,
                          act="sigmoid")
    fc2 = fluid.layers.fc(name="fc2", input=fc1, size=40, act="sigmoid")
    fc3 = fluid.layers.fc(name="fc3", input=fc2, size=1)
    logit = fc3 + item_b

    loss = fluid.layers.sigmoid_cross_entropy_with_logits(x=logit, label=label)
    avg_loss = fluid.layers.mean(loss)
    return avg_loss, fluid.layers.sigmoid(logit)

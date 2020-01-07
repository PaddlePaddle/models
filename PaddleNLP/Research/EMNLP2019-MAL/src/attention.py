#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.layer_helper import LayerHelper as LayerHelper

def generate_relative_positions_matrix(length, max_relative_position, cache=False):
    if not cache:
        range_vec = layers.range(0, length, 1, 'int32')
        range_vec.stop_gradient = True
        shapes = layers.shape(range_vec)
        range_vec = layers.reshape(range_vec, shape=[1, shapes[0]])
        range_mat = layers.expand(range_vec, [shapes[0], 1])
        distance_mat = range_mat - layers.transpose(range_mat, [1, 0])
    else:
        distance_mat = layers.range(-1 * length+1, 1, 1, 'int32')
        distance_mat.stop_gradient = True
        shapes = layers.shape(distance_mat)
        distance_mat = layers.reshape(distance_mat, [1, shapes[0]])

    distance_mat_clipped = layers.clip(layers.cast(distance_mat, dtype="float32"), float(-max_relative_position), float(max_relative_position))
    final_mat = layers.cast(distance_mat_clipped, dtype = 'int32') + max_relative_position
    return final_mat


def generate_relative_positions_embeddings(length, depth, max_relative_position, name, cache=False):
    relative_positions_matrix = generate_relative_positions_matrix(
                                    length, max_relative_position, cache=cache)

    y = layers.reshape(relative_positions_matrix, [-1])
    y.stop_gradient = True
    vocab_size = max_relative_position * 2 + 1
    #embeddings_table = layers.create_parameter(shape=[vocab_size, depth], dtype='float32', default_initializer=fluid.initializer.Constant(1.2345), name=name)
    embeddings_table = layers.create_parameter(shape=[vocab_size, depth], dtype='float32', name=name)
    #layers.Print(embeddings_table, message = "embeddings_table=====")
    embeddings_1 = layers.gather(embeddings_table, y)
    embeddings = layers.reshape(embeddings_1, [-1, length, depth])
    return embeddings


def _relative_attention_inner(q, k, v, transpose):
    batch_size = layers.shape(q)[0]
    heads = layers.shape(q)[1]
    length = layers.shape(q)[2]

    xy_matmul = layers.matmul(q, k, transpose_y=transpose)
    x_t = layers.transpose(q, [2, 0, 1, 3])
    x_t_r = layers.reshape(x_t, [length, batch_size * heads, -1])
    x_tz_matmul = layers.matmul(x_t_r, v, transpose_y = transpose)
    x_tz_matmul_r = layers.reshape(x_tz_matmul, [length, batch_size, heads, -1])
    x_tz_matmul_r_t = layers.transpose(x_tz_matmul_r, [1, 2, 0, 3])
    return xy_matmul + x_tz_matmul_r_t

def _dot_product_relative(q, k, v, bias, dropout=0.1, cache=None, params_type="normal"):
    depth_constant = int(k.shape[3])
    heads = layers.shape(k)[1]
    length = layers.shape(k)[2]

    max_relative_position = 4
    pre_name = "relative_positions_"
    if params_type == "fixed":
        pre_name = "fixed_relative_positions_"
    elif params_type == "new":
        pre_name = "new_relative_positions_"
    relations_keys = generate_relative_positions_embeddings(
                        length, depth_constant, max_relative_position, name=pre_name + "keys",
                        cache=cache is not None)

    relations_values = generate_relative_positions_embeddings(
                            length, depth_constant, max_relative_position,
                            name = pre_name + "values",
                            cache=cache is not None)

    logits = _relative_attention_inner(q, k, relations_keys, True)

    if bias is not None: logits += bias
    weights = layers.softmax(logits, name = "attention_weights")
    weights = layers.dropout(weights, dropout_prob=float(dropout))
    output =  _relative_attention_inner(weights, v, relations_values, False)
    return output

def scaled_dot_product_attention(q, k, v, attn_bias, d_key, dropout_rate):
    """
    Scaled Dot-Product Attention
    """
    scaled_q = layers.scale(x=q, scale=d_key ** -0.5)
    product = layers.matmul(x=scaled_q, y=k, transpose_y=True)
    if attn_bias:
        product += attn_bias
    weights = layers.softmax(product)
    if dropout_rate:
        weights = layers.dropout(
            weights,
            dropout_prob=dropout_rate,
            seed=ModelHyperParams.dropout_seed,
            is_test=False, dropout_implementation='upscale_in_train')
    out = layers.matmul(weights, v)
    return out

if __name__ == "__main__":
    batch_size = 2
    heads = 8
    length = 5
    depth = 3
    cpu = fluid.core.CPUPlace()
    exe = fluid.Executor(cpu)
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard("forward"):
            x = layers.reshape(layers.cast(layers.range(0, 18, 1, "int32"), dtype = "float32"), shape =[-1, 3, 3])
            y = layers.reshape(layers.cast(layers.range(0, 2, 1, "int32"), dtype = "float32"), shape =[-1, 1])
            z = x * y

    exe.run(startup_prog)
    outs = exe.run(train_prog, fetch_list=[x, y, z])
    print outs[0]
    print outs[1]
    print outs[2]



import tensorflow as tf
import paddle

tf_paddle_encoder_dict = {
    "LayerNorm/beta:0": "norm.bias",
    "LayerNorm/gamma:0": "norm.weight",
    "attention/self/query/kernel:0": "self_attn.q_proj.weight",
    "attention/self/query/bias:0": "self_attn.q_proj.bias",
    "attention/self/key/kernel:0": "self_attn.k_proj.weight",
    "attention/self/key/bias:0": "self_attn.k_proj.bias",
    "attention/self/value/kernel:0": "self_attn.v_proj.weight",
    "attention/self/value/bias:0": "self_attn.v_proj.bias",
    "attention/output/dense/kernel:0": "self_attn.out_proj.weight",
    "attention/output/dense/bias:0": "self_attn.out_proj.bias",
    "intermediate/dense/kernel:0": "linear1.weight",
    "intermediate/dense/bias:0": "linear1.bias",
    "output/dense/kernel:0": "linear2.weight",
    "output/dense/bias:0": "linear2.bias",
    "attention/output/LayerNorm/beta:0": "norm1.bias",
    "attention/output/LayerNorm/gamma:0": "norm1.weight",
    "output/LayerNorm/beta:0": "norm2.bias",
    "output/LayerNorm/gamma:0": "norm2.weight",
}


def get_paddle_weight_name(tf_weight_name):
    paddle_name = None
    name = tf_weight_name
    # 去掉第一个encoder前缀
    name = name.split("/", 1)[1]
    if name in tf_paddle_encoder_dict:
        paddle_name = tf_paddle_encoder_dict[name]
    elif name.startswith('layer_'):
        name_arr = name.split("/", 1)
        layer_num = name_arr[0].split("_", 1)[1]
        name = name_arr[1]
        paddle_name = "layers." + layer_num + "." + tf_paddle_encoder_dict[name]
    return paddle_name


def load_dict_from_tf(paddle_model, tf_model):
    for v in tf_model.trainable_variables:
        paddle_weight_name = get_paddle_weight_name(v.name)
        #print("{} -> {}".format(v.name, paddle_weight_name))
        paddle_model.state_dict()[paddle_weight_name].set_value(v.numpy())

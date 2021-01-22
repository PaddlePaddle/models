import tensorflow as tf
from bigbird.core import encoder
import numpy as np
import paddle
from paddle import fluid
from paddlenlp.efficient_transformers import TransformerEncoderLayer, TransformerEncoder

from tf_to_paddle import load_dict_from_tf
config = {
    # transformer basic configs
    "attention_probs_dropout_prob": 0.,
    "hidden_act": "relu",
    "hidden_dropout_prob": 0.,
    "hidden_size": 128,
    "initializer_range": 0.02,
    "intermediate_size": 512,
    "num_attention_heads": 1,
    "num_hidden_layers": 1,
    "use_bias": True,
    # sparse mask configs
    "attention_type": "simulated_sparse",
    "norm_type": "postnorm",
    "block_size": 1,
    "num_rand_blocks": 3
}
seed = 100


def test_equal(batch_size=2, input_length=50, hidden_size=128):
    np_input = np.random.rand(batch_size, input_length,
                              hidden_size).astype("float32")
    ########################## tf #################################

    enc_tf_input = tf.convert_to_tensor(np_input, dtype=tf.dtypes.float32)
    enc_tf_mask = tf.ones(
        shape=[batch_size, input_length], dtype=tf.dtypes.int32)

    bigbird_sparse_encoder = encoder.EncoderStack(config)
    rand_attn, attn_mask, enc_sparse_output = bigbird_sparse_encoder(
        enc_tf_input, enc_tf_mask, True)
    # print("attn_mask")
    # print(attn_mask)
    # print("trainable_variables")
    # for i,v in enumerate(bigbird_sparse_encoder.trainable_variables):
    #     print("{} {}".format(i, v.name))
    #    print(v.numpy())

    #print(enc_sparse_output)
    ########################## paddle #################################
    # paddle_input = np.concatenate(
    #     [np_input[:,0:config["block_size"],:],
    #      np_input[:, -config["block_size"]:, :],
    #      np_input[:,config["block_size"] : -config["block_size"],:]], 1)

    # paddle_attn_mask = attn_mask.numpy()
    # paddle_attn_mask = np.concatenate(
    #     [paddle_attn_mask[:,:,0:config["block_size"],:],
    #      paddle_attn_mask[:,:, -config["block_size"]:, :],
    #      paddle_attn_mask[:,:,config["block_size"] : -config["block_size"],:]], 2)
    # paddle_attn_mask = np.concatenate(
    #     [paddle_attn_mask[:,:,:,0:config["block_size"]],
    #      paddle_attn_mask[:,:,:,-config["block_size"]:,],
    #      paddle_attn_mask[:,:,:,config["block_size"]:-config["block_size"]]], 3)
    paddle_input = np_input
    paddle_attn_mask = attn_mask.numpy().astype("float32")
    paddle_attn_mask[paddle_attn_mask == 0] = -np.inf
    paddle_attn_mask[paddle_attn_mask == 1.] = 0

    paddle_attn_mask = paddle.to_tensor(paddle_attn_mask)

    enc_input = paddle.to_tensor(paddle_input)

    encoder_layer = TransformerEncoderLayer(
        config["hidden_size"],
        config["num_attention_heads"],
        config["intermediate_size"],
        num_global_blocks=2,
        attention_type="bigbird_simulated",
        seed=None,
        num_rand_blocks=config["num_rand_blocks"],
        dropout=0,
        block_size=config["block_size"])
    paddle_encoder = TransformerEncoder(encoder_layer, 1)
    load_dict_from_tf(paddle_encoder, bigbird_sparse_encoder)
    enc_output = paddle_encoder(enc_input, paddle_attn_mask)
    #print(enc_output)
    ret = np.allclose(enc_sparse_output.numpy(), enc_output.numpy(), atol=1e-5)
    if not ret:
        print("diff")
        print(enc_sparse_output.numpy() - enc_output.numpy())
        print()
    return ret


wrong_times = 0
for i in range(1):
    config["initializer_range"] = np.random.rand()  #* 0.1
    if not test_equal():
        print("initializer_range: {}".format(config["initializer_range"]))
        print("wrong at step {}".format(i))
        wrong_times += 1
print("wrong times:{}".format(wrong_times))

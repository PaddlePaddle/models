import paddle
paddle.set_device("cpu")


def trans_nanme(key):
    k = key
    k = k.replace("transformer", "gpt2.decoder")
    k = k.replace("mlp.dense_h_to_4h", "linear1")
    k = k.replace("mlp.dense_4h_to_h", "linear2")
    k = k.replace("attention.dense", "self_attn.out_proj")
    k = k.replace("input_layernorm", "norm1")
    k = k.replace("post_attention_layernorm", "norm2")
    k = k.replace("final_layernorm", "norm")
    k = k.replace("word_embeddings", "gpt2.embeddings.word_embeddings")
    k = k.replace("position_embeddings", "gpt2.embeddings.position_embeddings")
    return k


state = paddle.load("gpt2.pdparams")

new_state_dict = {}
for key in sorted(list(state.keys())):
    print(key, state[key].shape)
    new_key = trans_nanme(key)
    if "query_key_value" in key:
        if "weight" in key:
            q = state[key][:, :2560]
            k = state[key][:, 2560:5120]
            v = state[key][:, 5120:7680]
        else:
            q = state[key][:2560]
            k = state[key][2560:5120]
            v = state[key][5120:7680]
        q_name = new_key.replace("attention.query_key_value",
                                 "self_attn.q_proj")
        k_name = new_key.replace("attention.query_key_value",
                                 "self_attn.k_proj")
        v_name = new_key.replace("attention.query_key_value",
                                 "self_attn.v_proj")
        new_state_dict[q_name] = q
        new_state_dict[k_name] = k
        new_state_dict[v_name] = v
        continue

    new_state_dict[new_key] = state[key]

for k in sorted(list(new_state_dict.keys())):
    print(k)

paddle.save(new_state_dict, 'new_gpt2.pdparams')
# with open("short_a", 'r') as f:
#     lines = f.readlines()
#     for  line in lines:
#         print(trans_nanme(line), end="")

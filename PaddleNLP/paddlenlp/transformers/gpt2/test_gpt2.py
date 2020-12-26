# Pass
import sys
import paddle
import numpy as np
sys.path.append("../../../")
from paddlenlp.transformers import GPT2ForPretraining, GPT2Model


def main():
    config = GPT2ForPretraining.pretrained_init_configuration["gpt2-base"]
    gpt = GPT2ForPretraining(GPT2Model(**config))

    state_dict = paddle.load("new_gpt2.pdparams")
    for param in state_dict:
        state_dict[param] = state_dict[param].astype('float32')
    gpt.set_dict(state_dict)
    gpt.eval()
    model_dict = gpt.state_dict()
    for k in sorted(list(model_dict.keys())):
        print(k, model_dict[k].shape)
    np.random.seed(2020)
    cache = np.random.randn(32, 1, 2, 32, 9, 80).astype("float32")
    out, cached_kvs = gpt(
        input_ids=paddle.ones([1, 1], 'int64'),
        kv_cache=paddle.to_tensor(cache),
        #kv_cache=paddle.randn([32, 2, 32, 9, 80], 'float32'),
        use_cache=True)
    print(out.shape)
    print(out.reshape([30, -1]))


if __name__ == '__main__':
    main()

# Pass
import sys
import paddle
import numpy as np
sys.path.append("../../../")
from paddlenlp.transformers import GPT2ForPretraining, GPT2Model


def main():
    config = GPT2ForPretraining.pretrained_init_configuration["gpt2-base-cn"]
    gpt = GPT2ForPretraining(GPT2Model(**config))

    state_dict = paddle.load("new_gpt2.pdparams")
    for param in state_dict:
        state_dict[param] = state_dict[param].astype('float32')
    print("the dict has convert done.")
    paddle.save(state_dict, "./just_download")
    #gpt.set_dict(state_dict)
    #gpt.eval()
    #model_dict = gpt.state_dict()
    #np.random.seed(2020)
    #cache = np.random.randn(32, 1, 2, 32, 9, 80).astype("float32")
    #out, cached_kvs = gpt(
    #    input_ids=paddle.ones([1, 1], 'int64'),
    #    kv_cache=paddle.to_tensor(cache),
    #    use_cache=True)
    #print(out.mean())
    #print(cached_kvs[0].mean())


if __name__ == '__main__':
    main()

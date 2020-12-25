# Pass
import sys
import paddle

sys.path.append("../../../")
from paddlenlp.transformers import GPT2ForPretraining, GPT2Model

if __name__ == '__main__':
    # gpt = GPT2ForPretraining(
    #     vocab_size=30000,
    #     layer_size=2,
    #     block_size=1024,
    #     embedding_dropout=0.0,
    #     embedding_size=2560,
    #     num_attention_heads=32,
    #     attention_dropout=0.0,
    #     residual_dropout=0.0)
    config = GPT2ForPretraining.pretrained_init_configuration["gpt2-base"]
    gpt = GPT2ForPretraining(GPT2Model(**config))

    gpt.eval()
    model_dict = gpt.state_dict()
    for k in sorted(list(model_dict.keys())):
        print(k, model_dict[k].shape)
    out, cached_kvs = gpt(
        input_ids=paddle.ones([1, 1], 'int64'),
        kv_cache=paddle.randn([32, 1, 2, 32, 9, 80], 'float32'),
        #kv_cache=paddle.randn([32, 2, 32, 9, 80], 'float32'),
        use_cache=True)
    cached_kvs = paddle.stack(cached_kvs)
    print(out.shape, cached_kvs.shape)

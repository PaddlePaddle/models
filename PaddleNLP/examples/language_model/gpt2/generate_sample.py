import paddle
import argparse
import numpy as np
from paddlenlp.transformers import GPT2Model, GPT2ForPretraining
from paddlenlp.transformers import GPT2ChineseTokenizer

tokenizer = GPT2ChineseTokenizer.from_pretrained("gpt2-base-cn")
token_ids = tokenizer.convert_tokens_to_ids("中国")
print(token_ids)

print('正在加载模型，耗时需要几分钟，请稍后...')
model = GPT2ForPretraining.from_pretrained("gpt2-base-cn")
model.eval()
print('模型加载完成.')


# 基础预测函数
def predict(text, max_len=10):
    ids = tokenizer.convert_tokens_to_ids(text)
    input_id = paddle.to_tensor(np.array(ids).reshape(1, -1).astype('int64'))
    output, cached_kvs = model(input_id, use_cache=True, cache=None)
    nid = int(np.argmax(output[0, -1].numpy()))
    ids += [nid]
    out = [nid]
    print(np.mean(output.numpy()))
    for i in range(max_len):
        input_id = paddle.to_tensor(
            np.array([nid]).reshape(1, -1).astype('int64'))
        output, cached_kvs = model(input_id, use_cache=True, cache=cached_kvs)
        print(np.mean(output.numpy()))
        nid = int(np.argmax(output[0, -1].numpy()))
        ids += [nid]
        # 若遇到'\n'则结束预测
        if nid == 3:
            break
        out.append(nid)
    print(tokenizer.convert_ids_to_tokens(out))


# 问答
def ask_question(question, max_len=10):
    predict('''问题：中国的首都是哪里？
    答案：北京。
    问题：李白在哪个朝代？
    答案：唐朝。
    问题：%s
    答案：''' % question, max_len)


def ask_question(question, max_len=10):
    predict("问题：中国的首都是哪里？答案：北京。\n问题：%s 答案：" % question, max_len)


# 古诗默写
def dictation_poetry(front, max_len=10):
    predict('''默写古诗:
    白日依山尽，黄河入海流。
    %s，''' % front, max_len)


dictation_poetry("黄河之水天上来")

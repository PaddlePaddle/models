import paddle
import argparse
import numpy as np
from paddlenlp.utils.tools import loadz
from paddlenlp.transformers import GPT2Model, GPT2ForPretraining
from paddlenlp.transformers import GPT2ChineseTokenizer
from paddlenlp.utils.log import logger


class Demo:
    def __init__(self):
        self.tokenizer = GPT2ChineseTokenizer.from_pretrained("gpt2-base-cn")
        logger.info('Loading the model parameters, please wait...')
        self.model = GPT2ForPretraining.from_pretrained("gpt2-base-cn")
        self.model.eval()
        logger.info('Model loaded.')

# prediction function

    def predict(self, text, max_len=10):
        ids = self.tokenizer.convert_tokens_to_ids(text)
        input_id = paddle.to_tensor(
            np.array(ids).reshape(1, -1).astype('int64'))
        output, cached_kvs = self.model(input_id, use_cache=True, cache=None)
        nid = int(np.argmax(output[0, -1].numpy()))
        ids += [nid]
        out = [nid]
        for i in range(max_len):
            input_id = paddle.to_tensor(
                np.array([nid]).reshape(1, -1).astype('int64'))
            output, cached_kvs = self.model(
                input_id, use_cache=True, cache=cached_kvs)
            nid = int(np.argmax(output[0, -1].numpy()))
            ids += [nid]
            # if nid is '\n', the predicion is over.
            if nid == 3:
                break
            out.append(nid)
        logger.info(text)
        logger.info(self.tokenizer.convert_ids_to_tokens(out))

    # One shot example
    def ask_question(self, question, max_len=10):
        self.predict("问题：中国的首都是哪里？答案：北京。\n问题：%s 答案：" % question, max_len)

    # dictation poetry
    def dictation_poetry(self, front, max_len=10):
        self.predict('''默写古诗: 大漠孤烟直，长河落日圆。\n%s''' % front, max_len)

if __name__ == "__main__":
    demo = Demo()
    demo.ask_question("百度的厂长是谁?")
    demo.dictation_poetry("举杯邀明月，")

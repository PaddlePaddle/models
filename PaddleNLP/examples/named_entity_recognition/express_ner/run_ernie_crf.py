# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License.
import paddle
import paddle.nn as nn
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieTokenizer, ErniePretrainedModel
from paddlenlp.layers.crf import LinearChainCrf, LinearChainCrfLoss, ViterbiDecoder
from paddlenlp.metrics import ChunkEvaluator
from paddle.static import InputSpec 
from functools import partial


def parse_decodes(ds, decodes, lens):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]
    id_label = dict(zip(ds.label_vocab.values(), ds.label_vocab.keys()))

    outputs = []
    for idx, end in enumerate(lens):
        sent = ds.word_ids[idx][:end]
        tags = [id_label[x] for x in decodes[idx][1:end]]
        sent_out = []
        tags_out = []
        words = ""
        for s, t in zip(sent, tags):
            if t.endswith('-B') or t == 'O':
                if len(words):
                    sent_out.append(words)
                tags_out.append(t.split('-')[0])
                words = s
            else:
                words += s
        if len(sent_out) < len(tags_out):
            sent_out.append(words)
        outputs.append(''.join([str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return outputs

 
def convert_example(example, tokenizer, label_vocab):
    tokens, labels = example
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(tokens)
    lens = len(input_ids)
    labels = ['O'] + labels + ['O']
    labels = [label_vocab[x] for x in labels]
    return input_ids, segment_ids, lens, labels
 
 
def load_dict(dict_path):
    vocab = {}
    for line in open(dict_path, 'r', encoding='utf-8'):
        value, key = line.strip('\n').split('\t')
        vocab[key] = int(value)
    return vocab
 
 
class ExpressDataset(paddle.io.Dataset):
    def __init__(self, data_path):
        self.word_vocab = load_dict('./conf/word.dic')
        self.label_vocab = load_dict('./conf/tag.dic')
        self.word_ids = []
        self.label_ids = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                self.word_ids.append(words)
                self.label_ids.append(labels)
        self.word_num = max(self.word_vocab.values()) + 1
        self.label_num = max(self.label_vocab.values()) + 1
    def __len__(self):
        return len(self.word_ids)
    def __getitem__(self, index):
        return self.word_ids[index], self.label_ids[index]
 
 
class ErnieForTokenClassification(ErniePretrainedModel):
    def __init__(self, ernie, num_classes=2, dropout=None):
        super(ErnieForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie = ernie
        self.dropout = nn.Dropout(self.ernie.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"], num_classes)
        self.apply(self.init_weights)
 
    def forward(self,
                input_ids,
                token_type_ids=None,
                lens=None,
                position_ids=None,
                attention_mask=None):
        sequence_output, _ = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)
 
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits, lens, paddle.argmax(logits, axis=-1) 


class ErnieCRF(ErnieForTokenClassification):
    def __init__(self, ernie, num_classes=2, crf_lr=1.0, dropout=None):
        super(ErnieCRF, self).__init__(ernie, num_classes, dropout)
        self.crf = LinearChainCrf(num_classes, crf_lr, False)


if __name__ == '__main__':
    paddle.set_device('gpu')
 
    train_ds = ExpressDataset('./data/train.txt')
    dev_ds = ExpressDataset('./data/dev.txt')
    test_ds = ExpressDataset('./data/test.txt')
 
    tokenizer = ErnieTokenizer.from_pretrained('ernie')
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_vocab=train_ds.label_vocab)
 
    ignore_label = -100
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),
        Stack(),
        Pad(axis=0, pad_val=ignore_label)
    ): fn(list(map(trans_func, samples)))
 
    train_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_size=200,
        shuffle=True,
        return_list=True,
        collate_fn=batchify_fn)
    dev_loader = paddle.io.DataLoader(
        dataset=dev_ds,
        batch_size=200,
        return_list=True,
        collate_fn=batchify_fn)
    test_loader = paddle.io.DataLoader(
        dataset=test_ds,
        batch_size=200,
        return_list=True,
        collate_fn=batchify_fn)
        
    model = ErnieCRF.from_pretrained(
        'ernie', num_classes=train_ds.label_num)
    loss = LinearChainCrfLoss(transitions=model.crf.transitions)
    decoder = ViterbiDecoder(transitions=model.crf.transitions) 
    metric = ChunkEvaluator((train_ds.label_num + 2) // 2, "IOB")
    inputs = [InputSpec([None, None], dtype='int64', name='input_ids'), 
              InputSpec([None, None], dtype='int64', name='token_type_ids'), 
              InputSpec([None, None], dtype='int64', name='lens')]
 
    model = paddle.Model(model, inputs)
 
    optimizer = paddle.optimizer.AdamW(learning_rate=2e-5,parameters=model.parameters())
 
    model.prepare(optimizer, loss, metric) 
    model.fit(train_data=train_loader,
              eval_data=dev_loader,
              epochs=10,
              save_dir='./results',
              log_freq=1,
              save_freq=10000)

    model.evaluate(eval_data=test_loader)
    outputs, lens, decodes = model.predict(test_data=test_loader)
    pred = parse_decodes(test_ds, decodes, lens)
    print('\n'.join(pred[:10]))

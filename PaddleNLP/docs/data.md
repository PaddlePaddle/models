# paddlenlp.data

该模块提供了在NLP任务中构建有效的数据pipeline的一些常用API。

## API汇总

| API                             | 简介                                       |
| ------------------------------- | :----------------------------------------- |
| `paddlenlp.data.Stack`          | 堆叠N个具有相同shape的输入数据来构建一个batch |
| `paddlenlp.data.Pad`            | 堆叠N个输入数据来构建一个batch，每个输入数据将会被padding到N个输入数据中最大的长度 |
| `paddlenlp.data.Tuple`          | 将多个batchify函数包装在一起 |
| `paddlenlp.data.SamplerHelper`  | 构建用于`Dataloader`的可迭代sampler |
| `paddlenlp.data.Vocab`          | 用于文本token和ID之间的映射 |
| `paddlenlp.data.JiebaTokenizer` | Jieba分词 |

## API使用示例

```python
from paddlenlp.data import Vocab, JiebaTokenizer, Stack, Pad, Tuple, SamplerHelper
from paddlenlp.datasets import GlueCoLA
from paddlenlp.datasets import MapDatasetWrapper
from paddle.io import DataLoader

# 词表文件路径
vocab_file_path = './vocab.txt'
# 构建词表
vocab = Vocab.load_vocabulary(
    vocab_file_path,
    unk_token='[UNK]',
    pad_token='[PAD]',
    bos_token='[CLS]',
    eos_token='[SEP]')
# 初始化分词器
tokenizer = JiebaTokenizer(vocab)

pad_id = vocab.token_to_idx[vocab.pad_token]
bos_id = vocab.token_to_idx[vocab.bos_token]
eos_id = vocab.token_to_idx[vocab.eos_token]

def convert_example(example):
    text, label = example
    ids = [bos_id] + tokenizer.encode(text) + [eos_id]
    label = [label]
    return ids, label

dataset = GlueCoLA('train')
dataset = MapDatasetWrapper(dataset).apply(convert_example, lazy=True)

batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=pad_id),  # ids
    Stack(dtype='int64')  # label
): fn(samples)

batch_size = 16
batch_sampler = SamplerHelper(dataset).shuffle().batch(
    batch_size=batch_size,
    drop_last=True)
data_loader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    collate_fn=batchify_fn,
    return_list=True)

# 测试数据集
for batch in data_loader:
    ids, label = batch
    print(ids.shape, label.shape)
    print(ids)
    print(label)
    break
```

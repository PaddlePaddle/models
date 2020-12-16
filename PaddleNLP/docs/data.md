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

## API使用方法

### `paddlenlp.data.Stack`

```python
from paddle.data import Stack
a = [1, 2, 3, 4]
b = [3, 4, 5, 6]
c = [5, 6, 7, 8]
result = Stack()([a, b, c])
'''
[[1, 2, 3, 4],
 [3, 4, 5, 6],
 [5, 6, 7, 8]]
'''
```

### `paddlenlp.data.Pad`

```python
from paddle.data import Pad
a = [1, 2, 3, 4]
b = [5, 6, 7]
c = [8, 9]
result = Pad(axis=0, pad_val=0)([a, b, c])
'''
[[1, 2, 3, 4],
 [5, 6, 7, 0],
 [8, 9, 0, 0]]
'''
```

### `paddlenlp.data.Tuple`

```python
from paddle.data import Stack, Pad, Tuple
batchify_fn = Tuple(Pad(axis=0, pad_val=0), Stack())
```

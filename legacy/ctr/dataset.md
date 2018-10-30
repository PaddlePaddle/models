# 数据及处理
## 数据集介绍

本教程演示使用Kaggle上CTR任务的数据集\[[3](#参考文献)\]的预处理方法，最终产生本模型需要的格式，详细的数据格式参考[README.md](./README.md)。

Wide && Deep Model\[[2](#参考文献)\]的优势是融合稠密特征和大规模稀疏特征，
因此特征处理方面也针对稠密和稀疏两种特征作处理，
其中Deep部分的稠密值全部转化为ID类特征，
通过embedding 来转化为稠密的向量输入；Wide部分主要通过ID的叉乘提升维度。

数据集使用 `csv` 格式存储，其中各个字段内容如下：

-   `id` : ad identifier
-   `click` : 0/1 for non-click/click
-   `hour` : format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
-   `C1` : anonymized categorical variable
-   `banner_pos`
-   `site_id`
-   `site_domain`
-   `site_category`
-   `app_id`
-   `app_domain`
-   `app_category`
-   `device_id`
-   `device_ip`
-   `device_model`
-   `device_type`
-   `device_conn_type`
-   `C14-C21` : anonymized categorical variables


## 特征提取

下面我们会简单演示几种特征的提取方式。

原始数据中的特征可以分为以下几类：

1.  ID 类特征（稀疏，数量多）
-   `id`
-   `site_id`
-   `app_id`
-   `device_id`

2.  类别类特征（稀疏，但数量有限）

-   `C1`
-   `site_category`
-   `device_type`
-   `C14-C21`

3.  数值型特征转化为类别型特征

-   hour (可以转化成数值，也可以按小时为单位转化为类别）

### 类别类特征

类别类特征的提取方法有以下两种：

1.  One-hot 表示作为特征
2.  类似词向量，用一个 Embedding 将每个类别映射到对应的向量


### ID 类特征

ID 类特征的特点是稀疏数据，但量比较大，直接使用 One-hot 表示时维度过大。

一般会作如下处理：

1.  确定表示的最大维度 N
2.  newid = id % N
3.  用 newid 作为类别类特征使用

上面的方法尽管存在一定的碰撞概率，但能够处理任意数量的 ID 特征，并保留一定的效果\[[2](#参考文献)\]。

### 数值型特征

一般会做如下处理：

-   归一化，直接作为特征输入模型
-   用区间分割处理成类别类特征，稀疏化表示，模糊细微上的差别

## 特征处理


### 类别型特征

类别型特征有有限多种值，在模型中，我们一般使用 Embedding将每种值映射为连续值的向量。

这种特征在输入到模型时，一般使用 One-hot 表示，相关处理方法如下：

```python
class CategoryFeatureGenerator(object):
    '''
    Generator category features.

    Register all records by calling ~register~ first, then call ~gen~ to generate
    one-hot representation for a record.
    '''

    def __init__(self):
        self.dic = {'unk': 0}
        self.counter = 1

    def register(self, key):
        '''
        Register record.
        '''
        if key not in self.dic:
            self.dic[key] = self.counter
            self.counter += 1

    def size(self):
        return len(self.dic)

    def gen(self, key):
        '''
        Generate one-hot representation for a record.
        '''
        if key not in self.dic:
            res = self.dic['unk']
        else:
            res = self.dic[key]
        return [res]

    def __repr__(self):
        return '<CategoryFeatureGenerator %d>' % len(self.dic)
```

`CategoryFeatureGenerator` 需要先扫描数据集，得到该类别对应的项集合，之后才能开始生成特征。

我们的实验数据集\[[3](https://www.kaggle.com/c/avazu-ctr-prediction/data)\]已经经过shuffle，可以扫描前面一定数目的记录来近似总的类别项集合（等价于随机抽样），
对于没有抽样上的低频类别项，可以用一个 UNK 的特殊值表示。

```python
fields = {}
for key in categorial_features:
    fields[key] = CategoryFeatureGenerator()

def detect_dataset(path, topn, id_fea_space=10000):
    '''
    Parse the first `topn` records to collect meta information of this dataset.

    NOTE the records should be randomly shuffled first.
    '''
    # create categorical statis objects.

    with open(path, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for row_id, row in enumerate(reader):
            if row_id > topn:
                break

            for key in categorial_features:
                fields[key].register(row[key])
```

`CategoryFeatureGenerator` 在注册得到数据集中对应类别信息后，可以对相应记录生成对应的特征表示：

```python
record = []
for key in categorial_features:
    record.append(fields[key].gen(row[key]))
```

本任务中，类别类特征会输入到 DNN 中使用。

### ID 类特征

ID 类特征代稀疏值，且值的空间很大的情况，一般用模操作规约到一个有限空间，
之后可以当成类别类特征使用，这里我们会将 ID 类特征输入到 LR 模型中使用。

```python
class IDfeatureGenerator(object):
    def __init__(self, max_dim):
        '''
        @max_dim: int
            Size of the id elements' space
        '''
        self.max_dim = max_dim

    def gen(self, key):
        '''
        Generate one-hot representation for records
        '''
        return [hash(key) % self.max_dim]

    def size(self):
        return self.max_dim
```

`IDfeatureGenerator` 不需要预先初始化，可以直接生成特征，比如

```python
record = []
for key in id_features:
    if 'cross' not in key:
        record.append(fields[key].gen(row[key]))
```

### 交叉类特征

LR 模型作为 Wide & Deep model 的 `wide` 部分，可以输入很 wide 的数据（特征空间的维度很大），
为了充分利用这个优势，我们将演示交叉组合特征构建成更大维度特征的情况，之后塞入到模型中训练。

这里我们依旧使用模操作来约束最终组合出的特征空间的大小，具体实现是直接在 `IDfeatureGenerator` 中添加一个 `gen_cross_feature` 的方法：

```python
def gen_cross_fea(self, fea1, fea2):
    key = str(fea1) + str(fea2)
    return self.gen(key)
```

比如，我们觉得原始数据中， `device_id` 和 `site_id` 有一些关联（比如某个 device 倾向于浏览特定 site)，
我们通过组合出两者组合来捕捉这类信息。

```python
fea0 = fields[key].cross_fea0
fea1 = fields[key].cross_fea1
record.append(
    fields[key].gen_cross_fea(row[fea0], row[fea1]))
```

### 特征维度
#### Deep submodel(DNN)特征
| feature          | dimention |
|------------------|-----------|
| app_category     |        21 |
| site_category    |        22 |
| device_conn_type |         5 |
| hour             |        24 |
| banner_pos       |         7 |
| **Total**        | 79        |

#### Wide submodel(LR)特征
| Feature             | Dimention |
|---------------------|-----------|
| id                  |     10000 |
| site_id             |     10000 |
| app_id              |     10000 |
| device_id           |     10000 |
| device_id X site_id |   1000000 |
| **Total**           | 1,040,000 |

## 输入到 PaddlePaddle 中

Deep 和 Wide 两部分均以 `sparse_binary_vector` 的格式 \[[1](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/api/v1/data_provider/pydataprovider2_en.rst)\] 输入，输入前需要将相关特征拼合，模型最终只接受 3 个 input，
分别是

1.  `dnn input` ，DNN 的输入
2.  `lr input` , LR 的输入
3.  `click`  ， 标签

拼合特征的方法：

```python
def concat_sparse_vectors(inputs, dims):
    '''
    concaterate sparse vectors into one

    @inputs: list
        list of sparse vector
    @dims: list of int
        dimention of each sparse vector
    '''
    res = []
    assert len(inputs) == len(dims)
    start = 0
    for no, vec in enumerate(inputs):
        for v in vec:
            res.append(v + start)
        start += dims[no]
    return res
```

生成最终特征的代码如下：

```python
# dimentions of the features
categorial_dims = [
    feature_dims[key] for key in categorial_features + ['hour']
]
id_dims = [feature_dims[key] for key in id_features]

dense_input = concat_sparse_vectors(record, categorial_dims)
sparse_input = concat_sparse_vectors(record, id_dims)

record = [dense_input, sparse_input]
record.append(list((int(row['click']), )))
yield record
```

## 参考文献

1. <https://github.com/PaddlePaddle/Paddle/blob/develop/doc/api/v1/data_provider/pydataprovider2_en.rst>
2. Mikolov T, Deoras A, Povey D, et al. [Strategies for training large scale neural network language models](https://www.researchgate.net/profile/Lukas_Burget/publication/241637478_Strategies_for_training_large_scale_neural_network_language_models/links/542c14960cf27e39fa922ed3.pdf)[C]//Automatic Speech Recognition and Understanding (ASRU), 2011 IEEE Workshop on. IEEE, 2011: 196-201.
3. <https://www.kaggle.com/c/avazu-ctr-prediction/data>

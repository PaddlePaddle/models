<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
<ul>
<li><a href="#orga96c5e8">1. 数据集介绍</a></li>
<li><a href="#orge73ddcc">2. 特征提取</a>
<ul>
<li><a href="#orgbe379b1">2.1. 类别类特征</a></li>
<li><a href="#org811ca7c">2.2. ID 类特征</a></li>
<li><a href="#orgc1d7d23">2.3. 数值型特征</a></li>
</ul>
</li>
<li><a href="#org609b660">3. 特征处理</a>
<ul>
<li><a href="#org5fdd532">3.1. 类别型特征</a></li>
<li><a href="#orgad85d3e">3.2. ID 类特征</a></li>
<li><a href="#org0cbe90e">3.3. 交叉类特征</a></li>
<li><a href="#org4bdb372">3.4. 特征维度</a>
<ul>
<li><a href="#org0530a25">3.4.1. Deep submodel(DNN)特征</a></li>
<li><a href="#orged20ff2">3.4.2. Wide submodel(LR)特征</a></li>
</ul>
</li>
</ul>
</li>
<li><a href="#org70ad7d8">4. 输入到 PaddlePaddle 中</a></li>
</ul>
</div>
</div>


<a id="orga96c5e8"></a>

# 数据集介绍

数据集使用 `csv` 格式存储，其中各个字段内容如下：

-   id: ad identifier
-   click: 0/1 for non-click/click
-   hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
-   C1 &#x2013; anonymized categorical variable
-   banner<sub>pos</sub>
-   site<sub>id</sub>
-   site<sub>domain</sub>
-   site<sub>category</sub>
-   app<sub>id</sub>
-   app<sub>domain</sub>
-   app<sub>category</sub>
-   device<sub>id</sub>
-   device<sub>ip</sub>
-   device<sub>model</sub>
-   device<sub>type</sub>
-   device<sub>conn</sub><sub>type</sub>
-   C14-C21 &#x2013; anonymized categorical variables


<a id="orge73ddcc"></a>

# 特征提取

下面我们会简单演示几种特征的提取方式。

原始数据中的特征可以分为以下几类：

1.  ID 类特征（稀疏，数量多）
```python
    -   id
    -   site<sub>id</sub>
    -   app<sub>id</sub>
    -   device<sub>id</sub>

2.  类别类特征（稀疏，但数量有限）
```python
    -   C1
    -   site<sub>category</sub>
    -   device<sub>type</sub>
    -   C14-C21

3.  数值型特征转化为类别型特征
```python
    -   hour (可以转化成数值，也可以按小时为单位转化为类别）


<a id="orgbe379b1"></a>

## 类别类特征

类别类特征的提取方法有以下两种：

1.  One-hot 表示作为特征
2.  类似词向量，用一个 Embedding Table 将每个类别映射到对应的向量


<a id="org811ca7c"></a>

## ID 类特征

ID 类特征的特点是稀疏数据，但量比较大，直接使用 One-hot 表示时维度过大。

一般会作如下处理：

1.  确定表示的最大维度 N
2.  newid = id % N
3.  用 newid 作为类别类特征使用

上面的方法尽管存在一定的碰撞概率，但能够处理任意数量的 ID 特征，并保留一定的效果[2]。


<a id="orgc1d7d23"></a>

## 数值型特征

一般会做如下处理：

-   归一化，直接作为特征输入模型
-   用区间分割处理成类别类特征，稀疏化表示，模糊细微上的差别


<a id="org609b660"></a>

# 特征处理


<a id="org5fdd532"></a>

## 类别型特征

类别型特征有有限多种值，在模型中，我们一般使用 embedding table 将每种值映射为连续值的向量。

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

本任务中，类别类特征会输入到 DNN 中使用。


<a id="orgad85d3e"></a>

## ID 类特征

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


<a id="org0cbe90e"></a>

## 交叉类特征

LR 模型作为 Wide & Deep model 的 `wide` 部分，可以输入很 wide 的数据（特征空间的维度很大），
为了充分利用这个优势，我们将演示交叉组合特征构建成更大维度特征的情况，之后塞入到模型中训练。

这里我们依旧使用模操作来约束最终组合出的特征空间的大小，具体实现是直接在 `IDfeatureGenerator` 中添加一个~gen<sub>cross</sub><sub>feature</sub>~ 的方法：

```python
    def gen_cross_fea(self, fea1, fea2):
        key = str(fea1) + str(fea2)
        return self.gen(key)

比如，我们觉得原始数据中， `device_id` 和 `site_id` 有一些关联（比如某个 device 倾向于浏览特定 site)，
我们通过组合出两者组合来捕捉这类信息。


<a id="org4bdb372"></a>

## 特征维度


<a id="org0530a25"></a>

### Deep submodel(DNN)特征

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">feature</th>
<th scope="col" class="org-right">dimention</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">app<sub>category</sub></td>
<td class="org-right">21</td>
</tr>


<tr>
<td class="org-left">site<sub>category</sub></td>
<td class="org-right">22</td>
</tr>


<tr>
<td class="org-left">device<sub>conn</sub><sub>type</sub></td>
<td class="org-right">5</td>
</tr>


<tr>
<td class="org-left">hour</td>
<td class="org-right">24</td>
</tr>


<tr>
<td class="org-left">banner<sub>pos</sub></td>
<td class="org-right">7</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">Total</td>
<td class="org-right">79</td>
</tr>
</tbody>
</table>


<a id="orged20ff2"></a>

### Wide submodel(LR)特征

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Feature</th>
<th scope="col" class="org-right">Dimention</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">id</td>
<td class="org-right">10000</td>
</tr>


<tr>
<td class="org-left">site<sub>id</sub></td>
<td class="org-right">10000</td>
</tr>


<tr>
<td class="org-left">app<sub>id</sub></td>
<td class="org-right">10000</td>
</tr>


<tr>
<td class="org-left">device<sub>id</sub></td>
<td class="org-right">10000</td>
</tr>


<tr>
<td class="org-left">device<sub>id</sub> X site<sub>id</sub></td>
<td class="org-right">1000000</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">Total</td>
<td class="org-right">1,040,000</td>
</tr>
</tbody>
</table>


<a id="org70ad7d8"></a>

# 输入到 PaddlePaddle 中

Deep 和 Wide 两部分均以 `sparse_binary_vector` 的格式[1]输入，输入前需要将相关特征拼合，模型最终只接受 3 个 input，
分别是

1.  ~dnn input~，DNN 的输入
2.  `lr input`, LR 的输入
3.  ~click~， 标签

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

[1] <https://github.com/PaddlePaddle/Paddle/blob/develop/doc/api/v1/data_provider/pydataprovider2_en.rst>


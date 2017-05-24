<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
<ul>
<li><a href="#org5589824">1. 背景介绍</a>
<ul>
<li><a href="#org39b4382">1.1. LR vs DNN</a></li>
</ul>
</li>
<li><a href="#org5b2aafc">2. 数据和任务抽象</a></li>
<li><a href="#org17c7e05">3. 特征提取</a>
<ul>
<li><a href="#orga3db7a6">3.1. ID 类特征</a></li>
</ul>
</li>
<li><a href="#org5ce3921">4. 模型实现</a>
<ul>
<li><a href="#orgbb98635">4.1. DNN 简单模型</a></li>
<li><a href="#org41335f7">4.2. long wide 复杂模型</a></li>
</ul>
</li>
<li><a href="#orgb8efca9">5. 写在最后</a></li>
</ul>
</div>
</div>

<a id="org5589824"></a>

# 背景介绍

CTR(Click-through rate) 是用来表示用户点击一个特定链接的概率， 
通常被用来衡量一个在线广告系统的有效性。

当有多个广告位时，CTR 预估一般会作为排序的基准。
比如在百度的搜索广告系统，当用户输入一个带商业价值的搜索词（query）时，系统大体上会执行下列步骤：

1.  召回满足 query 的广告集合
2.  业务规则和相关性过滤
3.  根据拍卖机制和 CTR 排序
4.  展出

可以看到，CTR 在最终排序中起到了很重要的作用。

在业内，CTR 模型经历了如下的发展阶段：

-   Logistic Regression(LR) + 特征工程
-   LR + DNN 特征
-   DNN + 特征工程

在发展早期是 LR 一统天下，但最近 DNN 模型由于其强大的学习能力和逐渐成熟的性能优化，
逐渐地接过 CTR 预估任务的大旗。


<a id="org39b4382"></a>

## LR vs DNN

下图展示了 LR 和一个 \(3x2\) 的 NN 模型的结构：

![img](背景介绍/LR vs DNN_2017-05-22_10-09-02.jpg)

LR 部分和蓝色箭头部分可以直接类比到 NN 中的结构，可以看到 LR 和 NN 有一些共通之处（比如权重累加），
但前者的模型复杂度在相同输入维度下比后者可能第很多（从某方面讲，模型越复杂，越有潜力学习到更复杂的信息）。

如果 LR 要达到匹敌 NN 的学习能力，必须增加输入的维度，也就是增加特征的数量（作为输入），
这也就是为何 LR 和大规模的特征工程必须绑定在一起的原因。

而 NN 模型具有自己学习新特征的能力，一定程度上能够提升特征使用的效率，
这使得 NN 模型在同样规模特征的情况下，更有可能达到更好的学习效果。

本文会演示，如何使用 NN 模型来完成 CTR 预估的任务。


<a id="org5b2aafc"></a>

# 数据和任务抽象

我们可以将 \`click\` 作为学习目标，具体任务可以有以下几种方案：

1.  直接学习 click，0,1 作二元分类，或 pairwise rank（标签 1>0）
2.  统计每个广告的点击率，将同一个 query 下的广告两两组合，点击率高的>点击率低的

这里，我们直接使用第一种方法做分类任务。

我们使用 Kaggle 上 \`Click-through rate prediction\` 任务的数据集来演示模型。

各个字段内容如下：

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


<a id="org17c7e05"></a>

# 特征提取

下面我们会简单演示几种特征的提取方式。 

原始数据中的特征可以分为以下几类：

1.  ID 类特征（稀疏，数量多）
    -   id
    -   site<sub>id</sub>
    -   app<sub>id</sub>
    -   device<sub>id</sub>

2.  类别类特征（稀疏，但数量有限）
    -   C1
    -   site<sub>category</sub>
    -   device<sub>type</sub>
    -   C14-C21

3.  数值型特征
    -   hour (可以转化成数值，也可以按小时为单位转化为类别）


<a id="orga3db7a6"></a>

## ID 类特征

ID 类特征的特点是稀疏数据，但量比较大，直接使用 One-hot 表示时维度过大。

一般会作如下处理：


<a id="org5ce3921"></a>

# 模型实现


<a id="orgbb98635"></a>

## DNN 简单模型


<a id="org41335f7"></a>

## long wide 复杂模型


<a id="orgb8efca9"></a>

# 写在最后

<https://en.wikipedia.org/wiki/Click-through_rate>


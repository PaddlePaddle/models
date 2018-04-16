此目录中代码示例所需PaddlePaddle版本至少为v0.10.0。 如果您使用的是早于v0.10.0的PaddlePadddle版本，[请更新](http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html)。
*****
# <center>基于序列标注的事实型自动问答模型</center>

这个模型在下面的论文中实现了这个工作：

李鹏，李伟，何正言，徐旭光，曹颖，周杰，徐伟。事实型自动问答数据集和序列标注模型。[arXiv：1607.06275](https://arxiv.org/abs/1607.06275)。

如果您在研究中使用数据集/代码，请引用上述论文：

    @article{li:2016:arxiv,
    author  = {Li, Peng and Li, Wei and He, Zhengyan and Wang, Xuguang and Cao, Ying and Zhou, Jie and Xu, Wei},
    title   = {Dataset and Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question Answering},
    journal = {arXiv:1607.06275v2},
    year    = {2016},
    url     = {https://arxiv.org/abs/1607.06275v2},
    }


## 安装
1、利用下面的命令安装PaddlePaddle v0.10.5。提示：v0.10.0不支持。

    # either one is OK
    # CPU
    pip install paddlepaddle
    # GPU
    pip install paddlepaddle-gpu
2、通过运行下面的命令，下载[WebQA](http://research.baidu.com/)数据集。

    cd data && ./download.sh && cd ..

## 超参数
`connfig.py`定义了所有的超参数。其默认值和论文中一致。

## 训练
可以使用以下命令启动训练:

    PYTHONPATH=data/evaluation:<span class="markdown-equation" id="equation-0"></span>PYTHONPATH python val_and_test.py models [ann|ir]

其中
+ `models`：存储模型的文件夹。如果`config.py`没有更改，则可以直接使用模型。
+ `ann`：使用具有注释证据的验证集和测试集。
+ `ir`：使用带有检索证据的验证集和测试集。
请注意，验证和测试可以与训练同时运行。 `val_and_test.py`将处理同步的问题。

中间结果保存在`tmp`目录下。 在验证和测试后，你可以安全地删除它们。

结果应该与论文中表3所示的结果大致相同。

## 使用训练模型进行预测
通过下面的命令，来使用训练好的模型进行预测：

    PYTHONPATH=data/evaluation:$PYTHONPATH python infer.py \
    MODEL_FILE \
    INPUT_DATA \
    OUTPUT_FILE \
    2>&1 | tee infer.log

其中
+ `模型`：由`train.py`生成的训练模型。
+ `INPUT_DATA`：与WebQA数据集的验证/测试集相同的格式输入数据。
+ `OUTPUT_FILE`：利用评估脚本将结果格式化为WebQA数据集中为指定的格式。

## 预训练的模型
我们提供了两个预先训练的模型，一个用于带有注释证据的验证和测试集，另一个用于检索证据。这两个模型是根据相应版本验证集上的性能来选择的，与论文中一致。

上述模型可以通过下面命令下载：

    cd pre-trained-models && ./download-models.sh && cd ..

具有注释证据的测试集的评估结果可以利用下面的命令获取：

    PYTHONPATH=data/evaluation:$PYTHONPATH python infer.py \
    pre-trained-models/params_pass_00010.tar.gz \
    data/data/test.ann.json.gz \
    test.ann.output.txt.gz

    PYTHONPATH=data/evaluation:$PYTHONPATH \
    python data/evaluation/evaluate-tagging-result.py \
    test.ann.output.txt.gz \
    data/data/test.ann.json.gz \
    --fuzzy --schema BIO2
    # The result should be
    # chunk_f1=0.739091 chunk_precision=0.686119 chunk_recall=0.800926 true_chunks=3024 result_chunks=3530 correct_chunks=2422

对于检索证据的测试集的评估结果可以利用下面的命令获取：

    PYTHONPATH=data/evaluation:$PYTHONPATH python infer.py \
    pre-trained-models/params_pass_00021.tar.gz \
    data/data/test.ir.json.gz \
    test.ir.output.txt.gz

    PYTHONPATH=data/evaluation:$PYTHONPATH \
    python data/evaluation/evaluate-voting-result.py \
    test.ir.output.txt.gz \
    data/data/test.ir.json.gz \
    --fuzzy --schema BIO2
    # The result should be
    # chunk_f1=0.749358 chunk_precision=0.727868 chunk_recall=0.772156 true_chunks=3024 result_chunks=3208 correct_chunks=2335

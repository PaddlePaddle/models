此目录中代码示例PaddlePaddle所需版本至少为v0.11.0。如果您使用的PaddlePaddle版本早于v0.11.0， [请更新](http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html).

---

# 全球标准化阅读器

该模型实现以下功能：

Jonathan Raiman and John Miller. Globally Normalized Reader. Empirical Methods in Natural Language Processing (EMNLP), 2017

如果您在研究中使用数据集/代码，请引用上述论文：

```text
@inproceedings{raiman2015gnr,
    author={Raiman, Jonathan and Miller, John},
    booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
    title={Globally Normalized Reader},
    year={2017},
}
```

您也可以访问 https://github.com/baidu-research/GloballyNormalizedReader 以获取更多信息。


# 安装

1. 请使用 [docker image](http://doc.paddlepaddle.org/develop/doc/getstarted/build_and_install/docker_install_en.html) 安装最新的PaddlePaddle，运行方法：
    ```bash
    docker pull paddledev/paddle
    ```
2. 下载所有必要的数据，运行方法：
    ```bash
    cd data && ./download.sh && cd ..
    ```
3. 预处理并特征化数据：
    ```bash
    python featurize.py --datadir data --outdir data/featurized  --glove-path data/glove.840B.300d.txt
    ```

# 模型训练

- 根据需要修改config.py来配置模型，然后运行：

    ```bash
    python train.py 2>&1 | tee train.log
    ```

# 使用训练过的模型推断

- 运行以下训练模型来推断：
   ```bash
   python infer.py \
     --model_path models/pass_00000.tar.gz \
     --data_dir data/featurized/ \
     --batch_size 2 \
     --use_gpu 0 \
     --trainer_count 1 \
     2>&1 | tee infer.log
   ```

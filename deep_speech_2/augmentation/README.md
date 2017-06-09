# Augmentation: Design Doc
```python
代码结构：
| -- generate_data.py            exp：启动脚本，数据生成器
| -- audio_featurizer.py         exp：数据特征化处理脚本
| -- char_map.py                 exp：字符映射
| -- job_config_parser.py
| -- data                        exp：配置文件目录
|    | -- chars.txt
|    | -- config.json
|    | -- list
|    | -- num_files
| -- libspeech                   exp：语音处理库
|    | -- addio.py               exp：语音类定义及处理函数
|    | -- augmentation.py        exp：语音augmentation脚本
|    | -- error.py
|    | -- features.py            exp：语音特征化基类
|    | -- utils.py
|    | -- datasets  
|    | -- net
|    | -- augmentation_impl      exp：语音数据augmentation基类及派生
|    |    | -- base.py
|    |    | -- audio_database.py
|    |    | -- {augmentation}.py
```

# 执行过程说明：
## 1.执行 python generate_data.py

    参数： data_dir:  配置文件及清单所在目录
        max_minibatch: 数据最大分块数

    执行顺序：
        1）minibatches, opts, audio_pool = _prepare(data_dir, max_minibatch)

            读取data_dir中的配置，生成feature特征化处理所需的对象
            minibatches: List of lists. Each index of the outer list
                corresponds to a list of StrippedUtterances corresponding to
                that minibatch.
            opts (dict): Configuration parameters for this folder.
            audio_pool (pool-like): A pool-like object from the
                concurrent.futures module that has a `submit` method
                to be invoked by new tasks. Also has a `shutdown` method
                that needs to be invoked when appropriate.
            执行内容：
                读取配置脚本${data_dir}/config.json,生成opts(dict)
                读取数据文件清单${data_dir}/list， 生成minibatches
                生成audio_pool对象

        2）featurizer = _instantiate_featurizer(opts, audio_pool, data_dir, False)
            生成AudioFeaturizer对象
            初始化AudioFeaturizer对象，如果aug_pipeline存在，则生成augmenttation对象

        3）_featurize(featurizer, minibatches)
            对minibatches中的数据进行相关特征化处理
            执行 featurizer.featurize_minibatch(i, minibatch)
                对每个utterance执行get_audio_files(),提取feature，若存在aug_pipeline，则进行相对应的augmentation
                然后对每个utterance执行process_utterance，音频谱特征化处理，文本转录映射
                sorted(minibatch)
                write_minibatch()


# 接入DataGenerator
## 1.数据准备：

    audio数据：
        生成list文件清单
            "minibatch_id \t fname \t text \t duration \t add_noise"
        chars.txt
            字符映射
        config.json
            数据处理配置文件，参考config.md

    noise数据：
        生成index_file:
            "fname \t duration \t tags_dist"

## 2.执行 python generate_data.py
    对数据进行加噪、音频数据谱特征化及文本转录映射处理

## 3.接入 DataGenerator，加载reader()

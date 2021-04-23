# Audioset Tagging Example

本示例采用PANNs预训练模型，对输入音频实时打tag，并最终以文本形式输出对应时刻的topk类别和对应的得分。

PANNs预训练模型的详情，请参考论文[PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/pdf/1912.10211.pdf)。

## Usage

```python
python audio_tag.py \
    --wav ./cat_meow.wav \
    --sr 32000 \
    --sample_duration 2 \
    --hop_duration 0.3 \
    --checkpoint ./assets/cnn14.pdparams \
    --use_gpu True \
    --output_dir ./output_dir
```

参数用法：
```
--wav  # 音频路径
--sr  # sample rate
--sample_duration  # tagging音频长度，单位为秒
--hop_duration  # tagging音频间步长，单位为秒
--checkpoint  # 预训练模型参数
--use_gpu  # 使用GPU加速
--output_dir  # 输出路径
```

执行结果：
```
[2021-04-06 21:10:36,438] [    INFO] - Loaded CNN14 pretrained parameters from: ./assets/cnn14.pdparams
[2021-04-06 21:10:38,193] [    INFO] - Saved tagging results to ./output_dir/audioset_tagging_sr_32000.npz  
```

执行后得分结果保存在`output_dir`的`.npz`文件中。


## Output
```python
python parse_result.py \
    --input_file ./output_dir/audioset_tagging_sr_32000.npz \
    --topk 10 \
    --smooth True \
    --smooth_size 5 \
    --output_dir ./output_dir
```

参数用法：
```
--input_file  # tagging得分文件
--topk  # 展示topk结果
--smooth  # 帧间得分平滑
--smooth_size  # 平滑窗口大小
--output_dir  # 输出路径
```

执行结果：
```
[2021-04-06 21:22:00,696] [    INFO] - Posterior smoothing...
[2021-04-06 21:22:00,699] [    INFO] - Saved tagging labels to ./output_dir/audioset_tagging_sr_32000.txt
```

执行后文本结果保存在`output_dir`的`.txt`文件中。


## Labels

最终输出的文本结果如下所示。  
不同tagging的topk结果用空行分隔。每一个结果中，第一行是时间信息，数字表示tagging结果的起始样本点；接下来的k行是对应的标签和得分。

```
0
Cat: 0.80844646692276
Animal: 0.6848719716072083
Meow: 0.6470851898193359
Domestic animals, pets: 0.6392854452133179
Inside, small room: 0.05361200496554375
Purr: 0.02675800956785679
Music: 0.021260583773255348
Speech: 0.0209784135222435
Caterwaul: 0.019929537549614906
Outside, urban or manmade: 0.010916451923549175

9600
Cat: 0.7778594493865967
Meow: 0.6465566158294678
Animal: 0.6342337131500244
Domestic animals, pets: 0.5945377349853516
Inside, small room: 0.04747435823082924
Purr: 0.027785276994109154
Music: 0.022447215393185616
Caterwaul: 0.020785318687558174
Speech: 0.01982543244957924
Vehicle: 0.014558425173163414

19200
Cat: 0.8243843913078308
Animal: 0.6799540519714355
Meow: 0.6794822812080383
Domestic animals, pets: 0.6637188792228699
Caterwaul: 0.09927166253328323
Inside, small room: 0.0378643162548542
Music: 0.02170632779598236
Purr: 0.02035444974899292
Speech: 0.02006830833852291
Vehicle: 0.01234798226505518

28800
Cat: 0.8329735398292542
Animal: 0.6937487125396729
Meow: 0.6766577959060669
Domestic animals, pets: 0.6669812798500061
Caterwaul: 0.08647485077381134
Inside, small room: 0.03593464195728302
Music: 0.022975120693445206
Speech: 0.01964726485311985
Purr: 0.017558127641677856
Vehicle: 0.010926523245871067

38400
Cat: 0.8097503781318665
Animal: 0.6702587604522705
Meow: 0.6487116813659668
Domestic animals, pets: 0.6369225382804871
Caterwaul: 0.07185821980237961
Inside, small room: 0.039198972284793854
Music: 0.02381189912557602
Speech: 0.018534155562520027
Purr: 0.0178740955889225
Outside, urban or manmade: 0.011107126250863075

...
...
```

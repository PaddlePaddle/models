# 模型列表

## 1. 文本检测模型

### 1.1 中文检测模型

|模型名称|模型简介|推理模型大小|下载地址|
| --- | --- | --- | --- |
|ch_PP-OCRv3_det_slim|slim量化+蒸馏版超轻量模型，支持中英文、多语种文本检测| 1.1M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_slim_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_slim_distill_train.tar) / [nb模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_slim_infer.nb)|
|ch_PP-OCRv3_det| 原始超轻量模型，支持中英文、多语种文本检测 | 3.80M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar)|


### 1.2 英文检测模型

|模型名称|模型简介|推理模型大小|下载地址|
| --- | --- | --- | --- |
|en_PP-OCRv3_det_slim |slim量化版超轻量模型，支持英文、数字检测 | 1.1M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_slim_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_slim_distill_train.tar) / [nb模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_slim_infer.nb) |
|en_PP-OCRv3_det |原始超轻量模型，支持英文、数字检测|3.8M | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar) |

### 1.3 多语言检测模型

|模型名称|模型简介|推理模型大小|下载地址|
| --- | --- | --- | --- |
| ml_PP-OCRv3_det_slim |slim量化版超轻量模型，支持多语言检测 | 1.1M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_slim_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_slim_distill_train.tar) / [nb模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_slim_infer.nb) |
| ml_PP-OCRv3_det |原始超轻量模型，支持多语言检测 | 3.8M | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_distill_train.tar) |


## 2. 文本识别模型

### 2.1 中文识别模型

|模型名称|模型简介|推理模型大小|下载地址|
| --- | --- | --- | --- |
|ch_PP-OCRv3_rec_slim |slim量化版超轻量模型，支持中英文、数字识别| 4.9M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_train.tar) / [nb模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.nb) |
|ch_PP-OCRv3_rec|原始超轻量模型，支持中英文、数字识别| 12.4M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar) |

**说明：** `训练模型`是基于预训练模型在真实数据与竖排合成文本数据上finetune得到的模型，在真实应用场景中有着更好的表现，`预训练模型`则是直接基于全量真实数据与合成数据训练得到，更适合用于在自己的数据集上finetune。

### 2.2 英文识别模型

|模型名称|模型简介|推理模型大小|下载地址|
| --- | --- | --- | --- |
|en_PP-OCRv3_rec_slim |slim量化版超轻量模型，支持英文、数字识别 | 3.2M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_slim_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_slim_train.tar) / [nb模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_slim_infer.nb) |
|en_PP-OCRv3_rec |原始超轻量模型，支持英文、数字识别|9.6M | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar) |


### 2.3 多语言识别模型

|模型名称|字典文件|模型简介|推理模型大小|下载地址|
| --- | --- | --- | --- |--- |
| korean_PP-OCRv3_rec | ppocr/utils/dict/korean_dict.txt |韩文识别|11.0M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_train.tar) |
| japan_PP-OCRv3_rec | ppocr/utils/dict/japan_dict.txt |日文识别|11.0M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_train.tar) |
| chinese_cht_PP-OCRv3_rec | ppocr/utils/dict/chinese_cht_dict.txt | 中文繁体识别|12.0M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_train.tar) |
| te_PP-OCRv3_rec | ppocr/utils/dict/te_dict.txt | 泰卢固文识别|9.6M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_train.tar) |
| ka_PP-OCRv3_rec | ppocr/utils/dict/ka_dict.txt |卡纳达文识别|9.9M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_train.tar) |
| ta_PP-OCRv3_rec | ppocr/utils/dict/ta_dict.txt |泰米尔文识别|9.6M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_train.tar) |
| latin_PP-OCRv3_rec |  ppocr/utils/dict/latin_dict.txt | 拉丁文识别  |9.7M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_train.tar) |
| arabic_PP-OCRv3_rec | ppocr/utils/dict/arabic_dict.txt | 阿拉伯字母|9.6M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_train.tar) |
| cyrillic_PP-OCRv3_rec | ppocr/utils/dict/cyrillic_dict.txt | 斯拉夫字母  |9.6M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_train.tar) |
| devanagari_PP-OCRv3_rec | ppocr/utils/dict/devanagari_dict.txt |梵文字母 |9.9M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_train.tar) |



<a name="文本方向分类模型"></a>
## 3. 文本方向分类模型

|模型名称|模型简介|推理模型大小|下载地址|
| --- | --- | --- | --- |
|ch_ppocr_mobile_slim_v2.0_cls|slim量化版模型，对检测到的文本行文字角度分类| 2.1M |[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_slim_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_infer.tar) / [nb模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/lite/ch_ppocr_mobile_v2.0_cls_infer_opt.nb) |
|ch_ppocr_mobile_v2.0_cls|原始分类器模型，对检测到的文本行文字角度分类|1.38M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) |

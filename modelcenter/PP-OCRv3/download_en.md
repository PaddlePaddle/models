#  model list

## 1. Text Detection Model

### 1.1 Chinese detection model

| model  | description | model_size  | download         |
| --- | --- | --- | --- |
|ch_PP-OCRv3_det_slim|Slim quantization + distilled ultra-light model, supports Chinese, English and multilingual text detection| 1.1M |[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_slim_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_slim_distill_train.tar) / [nb model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_slim_infer.nb)|
|ch_PP-OCRv3_det| The original ultra-lightweight model supports Chinese, English, and multilingual text detection| 3.80M |[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar)|


### 1.2 English detection model

| model  | description | model_size  | download         |
| --- | --- | --- | --- |
|en_PP-OCRv3_det_slim |Slim quantitative version ultra-light model, supports English and digital detection | 1.1M |[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_slim_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_slim_distill_train.tar) / [nb model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_slim_infer.nb) |
|en_PP-OCRv3_det |Original ultra-lightweight model, supports English and digital detection|3.8M | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar) |

### 1.3 Multilingual Detection Model

| model  | description | model_size  | download         |
| --- | --- | --- | --- |
| ml_PP-OCRv3_det_slim |The slim quantized version of the ultra-lightweight model supports multi-language detection | 1.1M |[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_slim_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_slim_distill_train.tar) / [nb model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_slim_infer.nb) |
| ml_PP-OCRv3_det |Original ultra-lightweight model that supports multilingual detection | 3.8M | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_distill_train.tar) |


## 2. Text Recognition Model

### 2.1 Chinese Recognition Model

| model  | description | model_size  | download         |
| --- | --- | --- | --- |
|ch_PP-OCRv3_rec_slim |Slim quantitative version ultra-light model, supports Chinese and English, digital recognition| 4.9M |[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_train.tar) / [nb model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.nb) |
|ch_PP-OCRv3_rec|Original ultra-lightweight model, support Chinese and English, number recognition| 12.4M |[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar) |

**Note**: The `trained model` is finetuned on the `pre-trained model` with real data and synthsized vertical text data, which achieved better performance in real scene. The `pre-trained model` is directly trained on the full amount of real data and synthsized data, which is more suitable for finetune on your own dataset.

### 2.2 English Recognition Model

| model  | description | model_size  | download         |
| --- | --- | --- | --- |
|en_PP-OCRv3_rec_slim |Slim pruned and quantized lightweight model, supporting English and number recognition | 3.2M |[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_slim_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_slim_train.tar) / [nb模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_slim_infer.nb) |
|en_PP-OCRv3_rec |Original lightweight model, supporting English and number recognition|9.6M | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar) |

### 2.3 Multilingual Recognition Model

| model  | dict file| description | model_size  | download         |
| --- | --- | --- | --- |--- |
| korean_PP-OCRv3_rec | ppocr/utils/dict/korean_dict.txt |Korean recognition|11.0M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_train.tar) |
| japan_PP-OCRv3_rec | ppocr/utils/dict/japan_dict.txt |Japanese recognition|11.0M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_train.tar) |
| chinese_cht_PP-OCRv3_rec | ppocr/utils/dict/chinese_cht_dict.txt | Traditional Chinese recognition|12.0M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_train.tar) |
| te_PP-OCRv3_rec | ppocr/utils/dict/te_dict.txt | Telugu recognition|9.6M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_train.tar) |
| ka_PP-OCRv3_rec | ppocr/utils/dict/ka_dict.txt |Kannada recognition|9.9M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_train.tar) |
| ta_PP-OCRv3_rec | ppocr/utils/dict/ta_dict.txt |Tamil recognition|9.6M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_train.tar) |
| latin_PP-OCRv3_rec |  ppocr/utils/dict/latin_dict.txt | Latin recognition  |9.7M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_train.tar) |
| arabic_PP-OCRv3_rec | ppocr/utils/dict/arabic_dict.txt | Arabic alphabet|9.6M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_train.tar) |
| cyrillic_PP-OCRv3_rec | ppocr/utils/dict/cyrillic_dict.txt | Cyrillic alphabet  |9.6M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_train.tar) |
| devanagari_PP-OCRv3_rec | ppocr/utils/dict/devanagari_dict.txt |Sanskrit alphabet |9.9M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_train.tar) |

## 3. Text Angle Classification Model

| model  | description | model_size  | download         |
| --- | --- | --- | --- |
|ch_ppocr_mobile_slim_v2.0_cls|Slim quantized model for text angle classification| 2.1M |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_slim_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_infer.tar) |
|ch_ppocr_mobile_v2.0_cls|Original model for text angle classification|1.38M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) |

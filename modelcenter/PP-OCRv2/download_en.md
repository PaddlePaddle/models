#  model list

## 1. Text Detection Model

| model  | description | model_size  | download         |
| --- | --- | --- | --- |
|ch_PP-OCRv2_det_slim|slim quantization with distillation lightweight model, supporting Chinese, English, multilingual text detection| 3M |[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_slim_quant_infer.tar)|
|ch_PP-OCRv2_det|Original lightweight model, supporting Chinese, English, multilingual text detection|3M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar)|


## 2. Text Recognition Model

### 2.1 Chinese Recognition Model

| model  | description | model_size  | download         |
| --- | --- | --- | --- |
|ch_PP-OCRv2_rec_slim|Slim qunatization with distillation lightweight model, supporting Chinese, English, multilingual text detection| 9M |[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_train.tar) |
|ch_PP-OCRv2_rec|Original lightweight model, supporting Chinese, English, multilingual text detection|8.5M|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_train.tar) |

**Note**: The `trained model` is finetuned on the `pre-trained model` with real data and synthsized vertical text data, which achieved better performance in real scene. The `pre-trained model` is directly trained on the full amount of real data and synthsized data, which is more suitable for finetune on your own dataset.

### 2.2 English Recognition Model

| model  | description | model_size  | download         |
| --- | --- | --- | --- |
|en_number_mobile_slim_v2.0_rec|Slim pruned and quantized lightweight model, supporting English and number recognition| 2.7M | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/en_number_mobile_v2.0_rec_slim_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/en_number_mobile_v2.0_rec_slim_train.tar) |
|en_number_mobile_v2.0_rec|Original lightweight model, supporting English and number recognition|2.6M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_train.tar) |

### 2.3 Multilingual Recognition Model

| model  | dict file| description | model_size  | download         |
| --- | --- | --- | --- |--- |
| french_mobile_v2.0_rec | ppocr/utils/dict/french_dict.txt |Lightweight model for French recognition|2.65M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_train.tar) |
| german_mobile_v2.0_rec | ppocr/utils/dict/german_dict.txt |Lightweight model for German recognition|2.65M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_train.tar) |
| korean_mobile_v2.0_rec | ppocr/utils/dict/korean_dict.txt |Lightweight model for Korean recognition|3.9M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_train.tar) |
| japan_mobile_v2.0_rec | ppocr/utils/dict/japan_dict.txt |Lightweight model for Japanese recognition|4.23M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_train.tar) |
| chinese_cht_mobile_v2.0_rec | ppocr/utils/dict/chinese_cht_dict.txt | Lightweight model for chinese cht recognition|5.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_train.tar) |
| te_mobile_v2.0_rec | ppocr/utils/dict/te_dict.txt | Lightweight model for Telugu recognition|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_train.tar) |
| ka_mobile_v2.0_rec | ppocr/utils/dict/ka_dict.txt |Lightweight model for Kannada recognition|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_train.tar) |
| ta_mobile_v2.0_rec | ppocr/utils/dict/ta_dict.txt |Lightweight model for Tamil recognition|2.63M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_train.tar) |
| latin_mobile_v2.0_rec |  ppocr/utils/dict/latin_dict.txt | Lightweight model for latin recognition |2.6M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/latin_ppocr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/latin_ppocr_mobile_v2.0_rec_train.tar) |
| arabic_mobile_v2.0_rec | ppocr/utils/dict/arabic_dict.txt | Lightweight model for arabic recognition |2.6M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/arabic_ppocr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/arabic_ppocr_mobile_v2.0_rec_train.tar) |
| cyrillic_mobile_v2.0_rec | ppocr/utils/dict/cyrillic_dict.txt | Lightweight model for cyrillic recognition  |2.6M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/cyrillic_ppocr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/cyrillic_ppocr_mobile_v2.0_rec_train.tar) |
| devanagari_mobile_v2.0_rec | ppocr/utils/dict/devanagari_dict.txt |Lightweight model for devanagari recognition  |2.6M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/devanagari_ppocr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/devanagari_ppocr_mobile_v2.0_rec_train.tar) |


## 3. Text Angle Classification Model

| model  | description | model_size  | download         |
| --- | --- | --- | --- |
|ch_ppocr_mobile_slim_v2.0_cls|Slim quantized model for text angle classification| 2.1M |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_slim_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_infer.tar) |
|ch_ppocr_mobile_v2.0_cls|Original model for text angle classification|1.38M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) |

# Model list

## 1. Layout Analysis

|model name|description | inference model size |download|dict path|
| --- |---| --- | --- | --- |
| picodet_lcnet_x1_0_fgd_layout | The layout analysis English model trained on the PubLayNet dataset based on PicoDet LCNet_x1_0 and FGD . the model can recognition 5 types of areas such as **Text, Title, Table, Picture and List** | 9.7M | [inference model](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout.pdparams) | [PubLayNet dict](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt) |
| ppyolov2_r50vd_dcn_365e_publaynet | The layout analysis English model trained on the PubLayNet dataset based on PP-YOLOv2 | 221.0M | [inference_moel](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet.tar) / [trained model](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet_pretrained.pdparams) | same as above |
| picodet_lcnet_x1_0_fgd_layout_cdla | The layout analysis Chinese model trained on the CDLA dataset, the model can recognition 10 types of areas such as **Table、Figure、Figure caption、Table、Table caption、Header、Footer、Reference、Equation** | 9.7M | [inference model](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla.pdparams) | [CDLA dict](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/dict/layout_dict/layout_cdla_dict.txt) |
| picodet_lcnet_x1_0_fgd_layout_table | The layout analysis model trained on the table dataset, the model can detect tables in Chinese and English documents                     | 9.7M                                                  | [inference model](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_table_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_table.pdparams) | [Table dict](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/dict/layout_dict/layout_table_dict.txt) |
| ppyolov2_r50vd_dcn_365e_tableBank_word | The layout analysis model trained on the TableBank Word dataset based on PP-YOLOv2, the model can detect  tables  in English documents | 221.0M | [inference model](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_tableBank_word.tar) | same as above |
| ppyolov2_r50vd_dcn_365e_tableBank_latex | The layout analysis model trained on the TableBank Latex dataset based on PP-YOLOv2, the model can detect  tables  in English documents | 221.0M                 | [inference model](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_tableBank_latex.tar) | same as above |

## 2. OCR and Table Recognition

### 2.1 OCR

|model name| description | inference model size |download|
| --- |---|---| --- |
|en_ppocr_mobile_v2.0_table_det| Text detection model of English table scenes trained on PubTabNet dataset | 4.7M                |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_det_train.tar) |
|en_ppocr_mobile_v2.0_table_rec| Text recognition model of English table scenes trained on PubTabNet dataset | 6.9M                |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_rec_train.tar) |

If you need to use other OCR models, you can download the model in [PP-OCR model_list](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md) or use the model you trained yourself to configure to `det_model_dir`, `rec_model_dir` field.

<a name="22"></a>
### 2.2 Table Recognition

|model| description                                                                 |inference model size|download|
| --- |-----------------------------------------------------------------------------| --- | --- |
|en_ppocr_mobile_v2.0_table_structure| English table recognition model trained on PubTabNet dataset based on TableRec-RARE |6.8M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_structure_train.tar) |
|en_ppstructure_mobile_v2.0_SLANet|English table recognition model trained on PubTabNet dataset based on SLANet|9.2M|[inference model](https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_train.tar) |
|ch_ppstructure_mobile_v2.0_SLANet|Chinese table recognition model based on SLANet|9.3M|[inference model](https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_train.tar) |

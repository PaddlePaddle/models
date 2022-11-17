# 模型列表

## 1. 版面分析模型

|模型名称|模型简介|推理模型大小|下载地址|dict path|
| --- | --- | --- | --- | --- |
| picodet_lcnet_x1_0_fgd_layout | 基于PicoDet LCNet_x1_0和FGD蒸馏在PubLayNet 数据集训练的英文版面分析模型，可以划分**文字、标题、表格、图片以及列表**5类区域 | 9.7M | [推理模型](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout.pdparams) | [PubLayNet dict](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt) |
| ppyolov2_r50vd_dcn_365e_publaynet | 基于PP-YOLOv2在PubLayNet数据集上训练的英文版面分析模型 | 221.0M | [推理模型](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet.tar) / [训练模型](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet_pretrained.pdparams) | 同上 |
| picodet_lcnet_x1_0_fgd_layout_cdla | CDLA数据集训练的中文版面分析模型，可以划分为**表格、图片、图片标题、表格、表格标题、页眉、脚本、引用、公式**10类区域 | 9.7M | [推理模型](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla.pdparams) | [CDLA dict](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/dict/layout_dict/layout_cdla_dict.txt) |
| picodet_lcnet_x1_0_fgd_layout_table | 表格数据集训练的版面分析模型，支持中英文文档表格区域的检测 | 9.7M | [推理模型](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_table_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_table.pdparams) | [Table dict](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/dict/layout_dict/layout_table_dict.txt) |
| ppyolov2_r50vd_dcn_365e_tableBank_word | 基于PP-YOLOv2在TableBank Word 数据集训练的版面分析模型，支持英文文档表格区域的检测 | 221.0M | [推理模型](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_tableBank_word.tar) | 同上 |
| ppyolov2_r50vd_dcn_365e_tableBank_latex | 基于PP-YOLOv2在TableBank Latex数据集训练的版面分析模型，支持英文文档表格区域的检测 | 221.0M | [推理模型](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_tableBank_latex.tar) | 同上 |


## 2. OCR和表格识别模型

### 2.1 OCR

|模型名称|模型简介|推理模型大小|下载地址|
| --- | --- | --- | --- |
|en_ppocr_mobile_v2.0_table_det|PubTabNet数据集训练的英文表格场景的文字检测|4.7M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_det_train.tar) |
|en_ppocr_mobile_v2.0_table_rec|PubTabNet数据集训练的英文表格场景的文字识别|6.9M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_rec_train.tar) |

如需要使用其他OCR模型，可以在 [PP-OCR model_list](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md) 下载模型或者使用自己训练好的模型配置到 `det_model_dir`, `rec_model_dir`两个字段即可。

### 2.2 表格识别模型

|模型名称|模型简介|推理模型大小|下载地址|
| --- | --- | --- | --- |
|en_ppocr_mobile_v2.0_table_structure|基于TableRec-RARE在PubTabNet数据集上训练的英文表格识别模型|6.8M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_structure_train.tar) |
|en_ppstructure_mobile_v2.0_SLANet|基于SLANet在PubTabNet数据集上训练的英文表格识别模型|9.2M|[推理模型](https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_train.tar) |
|ch_ppstructure_mobile_v2.0_SLANet|基于SLANet的中文表格识别模型|9.3M|[推理模型](https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_train.tar) |

<a name="3"></a>

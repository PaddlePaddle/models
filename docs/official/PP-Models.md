## 飞桨PP系列模型


针对用户产业实践中的痛点问题，飞桨打造了PP系列模型，实现模型精度与预测效率的最佳平衡，满足企业落地实际需求。

|开发套件|PP系列模型|模型简介|快速开始|
|---|---|---|---|
|PaddleClas|PP-LCNet|CPU轻量级骨干网络，提供8种不同尺度的模型，在ImageNet 1k分类数据集上，精度可达71.32%，相比MobileNetV3-Small x0.35模型，提高18个百分点，Intel CPU 硬件上预测速度超过400 FPS，相比MobileNetV3-Small x0.35模型提高22%。|[快速开始](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/PP-LCNet.md)|
|PaddleClas|PP-LCNetv2|基于PP-LCNet优化的轻量级SOTA骨干网络，在ImageNet 1k分类数据集上，精度可达77.04%，相较MobileNetV3-Large x1.25精度提高0.64个百分点，同时在  Intel CPU 硬件上，预测速度可达 230 FPS ，相比 MobileNetV3-Large x1.25 预测速度提高 20%|[快速开始](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/PP-LCNetV2.md)|
|PaddleClas|PP-HGNet|GPU高性能骨干网络，在ImageNet 1k分类数据集上，精度可达79.83%、81.51%，同等速度下，相较ResNet34-D提高3.8个百分点，相较ResNet50-D提高2.4个百分点，在使用百度自研 SSLD 蒸馏策略后，精度相较ResNet50-D提高4.7个百分点。|[快速开始](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/PP-HGNet.md)|
|PaddleClas|PP-ShiTu|轻量图像识别系统，集成了目标检测、特征学习、图像检索等模块，广泛适用于各类图像识别任务，CPU上0.2s即可完成在10w+库的图像识别。|[快速开始](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.3#pp-shitu%E5%9B%BE%E5%83%8F%E8%AF%86%E5%88%AB%E7%B3%BB%E7%BB%9F%E4%BB%8B%E7%BB%8D)|
|PaddleClas|PP-ShiTuV2|PP-ShiTuV2 是基于 PP-ShiTuV1 改进的一个实用轻量级通用图像识别系统，由主体检测、特征提取、向量检索三个模块构成，相比 PP-ShiTuV1 具有更高的识别精度、更强的泛化能力以及相近的推理速度。|[快速开始](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PP-ShiTu/README.md)|
|PaddleDetection|PP-YOLO|基于YOLOv3优化的高精度目标检测模型，精度达到45.9%，在单卡V100上FP32推理速度为72.9 FPS, V100上开启TensorRT下FP16推理速度为155.6 FPS。|[快速开始](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/ppyolo/README_cn.md)|
|PaddleDetection|PP-YOLOv2|高精度目标检测模型，对比PP-YOLO， 精度提升 3.6%，达到49.5%；在 640*640 的输入尺寸下，速度可实现68.9FPS，采用 TensorRT 加速，FPS 还可达到106.5FPS。|[快速开始](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/ppyolo/README_cn.md)|
|PaddleDetection|PP-YOLOE|高精度云边一体SOTA目标检测模型，提供s/m/l/x版本，l版本COCO test2017数据集精度51.4%，V100预测速度78.1 FPS，支持混合精度训练，训练较PP-YOLOv2加速33%，全系列多尺度模型满足不同硬件算力需求，可适配服务器、边缘端GPU及其他服务器端AI加速卡。|[快速开始](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/ppyoloe/README_cn.md)|
|PaddleDetection|PP-YOLOE+|PP-YOLOE升级版，最高精度提升2.4% mAP，达到54.9% mAP，模型训练收敛速度提升3.75倍，端到端预测速度最高提升2.3倍；多个下游任务泛化性提升。|[快速开始](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/ppyoloe)|
|PaddleDetection|PP-PicoDet|超轻量级目标检测模型，提供xs/s/m/l四种尺寸，其中s版本参数量仅1.18m，却可达到32.5%mAP，相较YOLOX-Nano精度高6.7%，速度快26%，同时优化量化部署方案，实现在移动端部署加速30%+。|[快速开始](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet)|
|PaddleDetection|PP-Tracking|实时多目标跟踪工具，融合目标检测、行人重识别、轨迹融合等核心能力，提供行人车辆跟踪、跨镜头跟踪、多类别跟踪、小目标跟踪及流量技术等能力与产业应用。|[快速开始](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/deploy/pptracking/README_cn.md)|
|PaddleDetection|PP-TinyPose|超轻量级人体关键点检测算法，单人场景FP16推理可达到122FPS、精度51.8%AP，具有精度高速度快、检测人数无限制、微小目标效果好的特点。|[快速开始](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/keypoint/tiny_pose)|
|PaddleDetection|PP-TinyPose+|PP-TinyPose升级版，在健身、舞蹈等场景的业务数据集端到端AP提升9.1；新增体育场景真实数据，复杂动作识别效果显著提升；覆盖侧身、卧躺、跳跃、高抬腿等非常规动作；检测模型升级为[PP-PicoDet增强版](https://github.com/PaddlePaddle/PaddleDetection/blob/ede22043927a944bb4cbea0e9455dd9c91b295f0/configs/picodet/README.md)，在COCO数据集上精度提升3.1%；关键点稳定性增强；新增滤波稳定方式，视频预测结果更加稳定平滑|[快速开始](https://github.com/PaddlePaddle/PaddleDetection/blob/ede22043927a944bb4cbea0e9455dd9c91b295f0/configs/keypoint/tiny_pose/README.md)|
|PaddleDetection|PP-Human|产业级实时行人分析工，支持属性分析、行为识别、流量计数/轨迹留存三大功能，覆盖目标检测、多目标跟踪、属性识别、关键点检测、行为识别和跨镜跟踪六大核心技术。|[快速开始](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/pphuman)|
|PaddleDetection|PP-HumanV2|新增打架、打电话、抽烟、闯入四大行为识别，底层算法性能升级，覆盖行人检测、跟踪、属性三类核心算法能力，提供保姆级全流程开发及模型优化策略。|[快速开始](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/deploy/pipeline)|
|PaddleDetection|PP-Vehicle|提供车牌识别、车辆属性分析（颜色、车型）、车流量统计以及违章检测四大功能，完善的文档教程支持高效完成二次开发与模型优化。|[快速开始](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/deploy/pipeline)|
|PaddleSeg|PP-HumanSeg|PP-HumanSeg是在大规模人像数据上训练的人像分割系列模型，提供了多种模型，满足在Web端、移动端、服务端多种使用场景的需求。其中PP-HumanSeg-Lite采用轻量级网络设计、连通性学习策略、非结构化稀疏技术，实现体积、速度和精度的SOTA平衡。（参数量137K，速度达95FPS，mIoU达93%）|[快速开始](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/README_cn.md)|
|PaddleSeg|PP-HumanSegV2|PP-HumanSegV2是PP-HumanSeg的改进版本，肖像分割模型的推理耗时减小45.5%、mIoU提升3.03%、可视化效果更佳，通用人像分割模型的推理速度和精度也有明显提升。|[快速开始](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/README_cn.md)|
|PaddleSeg|PP-HumanMatting|PP-HumanMatting通过低分辨粗预测和高分辨率Refine的两阶段设计，在增加小量计算量的情况下，有效保持了高分辨率(>2048)人像扣图中细节信息。|[快速开始](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/Matting/README_CN.md)|
|PaddleSeg|PP-LiteSeg|PP-LiteSeg是通用轻量级语义分割模型，使用灵活高效的解码模块、统一注意力融合模块、轻量的上下文模块，针对Nvidia GPU上的产业级分割任务，实现精度和速度的SOTA平衡。在1080ti上精度为mIoU 72.0（Cityscapes数据集）时，速度高达273.6 FPS|[快速开始](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/configs/pp_liteseg)|
|PaddleSeg|PP-Matting|PP-Matting 通过引导流设计，实现语义引导下的高分辨率细节预测，进而实现trimap-free高精度图像抠图。在公开数据集Composition-1k和Distinctions-646测试集取得了SOTA的效果 。|[快速开始](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/Matting/README_CN.md)|
|PaddleSeg|PP-MattingV2|PP-MattingV2是PaddleSeg自研的轻量级抠图SOTA模型，通过双层金字塔池化及空间注意力提取高级语义信息，并利用多级特征融合机制兼顾语义和细节的预测。 对比MODNet模型推理速度提升44.6%， 误差平均相对减小17.91%。追求更高速度，推荐使用该模型。|[快速开始](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/Matting/README_CN.md)|
|PaddleOCR|PP-OCR|PP-OCR是一个两阶段超轻量OCR系统，包括文本检测、方向分类器、文本识别三个部分，支持竖排文本识别。PP-OCR mobile中英文模型3.5M，英文数字模型2.8M。在通用场景下达到产业级SOTA标准|[快速开始](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/quickstart.md)|
|PaddleOCR|PP-OCRv2|PP-OCRv2在PP-OCR的基础上进行优化，平衡PP-OCR模型的精度和速度，效果相比PP-OCR mobile 提升7%；推理速度相比于PP-OCR server提升220%；支持80种多语言模型|[快速开始](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/quickstart.md)|
|PaddleOCR|PP-OCRv3|PP-OCRv3进一步在原先系统上优化，在中文场景效果相比于PP-OCRv2再提升5%，英文场景提升11%，80语种多语言模型平均识别准确率提升5%以上|[快速开始](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/quickstart.md)|
|PaddleOCR|PP-Structure|PP-Structure是一套智能文档分析系统，支持版面分析、表格识别（含Excel导出）、关键信息提取与DocVQA（含语义实体识别和关系抽取）|[快速开始](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/ppstructure/docs/quickstart.md)|
|PaddleOCR|PP-StructureV2|基于PP-Structure系统功能性能全面升级，适配中文场景，新增支持版面复原，支持一行命令完成PDF转Word|[快速开始](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/docs/quickstart.md)|
|PaddeleGAN|PP-MSVSR|高精度视频超分算法，提供1.45M和7.4M两种参数量大小的模型，峰值信噪比与结构相似度均高于其他开源算法，以PSNR 32.53、SSIM 0.9083达到业界SOTA，同时对输入视频的分辨率不限制，支持分辨率一次提升400%。|[快速开始](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md)|
|PaddleVideo|PP-TSM|高精度2D实用视频分类模型PP-TSM。在不增加参数量和计算量的情况下，在UCF-101、Kinetics-400等数据集上精度显著超过TSM原始模型|[快速开始](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/pp-tsm.md)|
|PaddleVideo|PP-TSMv2|PP-TSMv2沿用了部分PP-TSM的优化策略，从骨干网络与预训练模型选择、数据增强、tsm模块调优、输入帧数优化、解码速度优化、dml蒸馏、新增时序attention模块等7个方面进行模型调优，在中心采样评估方式下，精度达到75.16%，输入10s视频在CPU端的推理速度仅需456ms。|[快速开始](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/quick_start.md)|
|PaddleNLP|ERNIE-M|面向多语言建模的预训练模型，ERNIE-M 提出基于回译机制，从单语语料中学习语言间的语义对齐关系，在跨语言自然语言推断、语义检索、语义相似度、命名实体识别、阅读理解等各种跨语言下游任务中取得了 SOTA 效果。|[快速开始](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-m)|
|PaddleNLP|ERNIE-UIE|通用信息抽取模型，实现了实体抽取、关系抽取、事件抽取、情感分析等任务的统一建模，并使得不同任务间具备良好的迁移和泛化能力。支持文本、跨模态文档的信息抽取。支持中、英、中英混合文本抽取。零样本和小样本能力卓越。|[快速开始](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie)|
|PaddleNLP|ERNIE 3.0-Medium|文本领域预训练模型，在文心大模型 ERNIE 3.0 基础上通过在线蒸馏技术得到的轻量级模型，CLUE 评测验证其在同等规模模型(6-layer, 768-hidden, 12-heads)中效果SOTA。|[快速开始](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0)|
|PaddleNLP|ERNIE 3.0-Mini|文本领域预训练模型，在文心大模型 ERNIE 3.0 基础上通过在线蒸馏技术得到的轻量级模型，CLUE 评测验证其在同等规模模型(6-layer, 384-hidden, 12-heads)中效果SOTA。|[快速开始](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0)|
|PaddleNLP|ERNIE 3.0-Micro|文本领域预训练模型，在文心大模型 ERNIE 3.0 基础上通过在线蒸馏技术得到的轻量级模型，CLUE 评测验证其在同等规模模型(4-layer, 384-hidden, 12-heads)中效果SOTA。|[快速开始](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0)|
|PaddleNLP|ERNIE 3.0-Nano|文本领域预训练模型，在文心大模型 ERNIE 3.0 基础上通过在线蒸馏技术得到的轻量级模型，CLUE 评测验证其在同等规模模型(4-layer, 312-hidden, 12-heads)中效果SOTA。|[快速开始](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0)|
|PaddleNLP|ERNIE-Layout|多语言跨模态布局增强文档智能大模型，将布局知识增强技术融入跨模态文档预训练，在4项文档理解任务上刷新世界最好效果，登顶 DocVQA 榜首。|[快速开始](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-layout)|
|PaddleNLP|ERNIE-ViL|业界首个融合场景图知识的多模态预训练模型，在包括视觉常识推理、视觉问答、引用表达式理解、跨模态图像检索、跨模态文本检索等 5 项典型多模态任务中刷新了世界最好效果，并在多模态领域权威榜单视觉常识推理任务（VCR）上登顶榜首。|[快速开始](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/ernie_vil)|
|PaddleSpeech|PP-ASR|PP-ASR是一套基于端到端神经网络结构模型的流式语音识别系统，支持实时语音识别服务，支持Language Model解码与个性化识别|[快速开始](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/asr/PPASR_cn.md)|
|PaddleSpeech|PP-TTS|PP-TTS是一套基于基于端到端神经网络结构的流式语音合成系统，支持流式声学模型与流式声码器，开源快速部署流式合成服务方案|[快速开始](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/tts/PPTTS_cn.md)|
|PaddleSpeech|PP-VPR|PP-VPR是一套声纹提取与检索系统，使用ECAPA-TDNN模型提取声纹特征，识别等错误率（EER，Equal error rate）低至0.95%，并且通过串联Mysql和Milvus，搭建完整的音频检索系统，实现毫秒级声音检索。|[快速开始](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/vpr/PPVPR_cn.md)|
|PaddleSpeech|ERNIE-SAT|语音-语言跨模态大模型文心 ERNIE-SAT 在语音编辑、个性化语音合成以及跨语言的语音合成等多个任务取得了领先效果，可以应用于语音编辑、个性化合成、语音克隆、同传翻译等一系列场景|[快速开始](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3_vctk/ernie_sat)|

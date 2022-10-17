## 飞桨PP系列模型


针对用户产业实践中的痛点问题，飞桨打造了PP系列模型，实现模型精度与预测效率的最佳平衡，满足企业落地实际需求。

|开发套件|PP系列模型|模型简介|快速开始|
|---|---|---|---|
|PaddleClas|PP-LCNet|CPU轻量级骨干网络，提供8种不同尺度的模型，在ImageNet 1k分类数据集上，精度可达71.32%，相比MobileNetV3-Small x0.35模型，提高18个百分点，Intel CPU 硬件上预测速度超过400 FPS，相比MobileNetV3-Small x0.35模型提高22%。|[快速开始](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/PP-LCNet.md)|
|PaddleClas|PP-LCNetv2|基于PP-LCNet优化的轻量级SOTA骨干网络，在ImageNet 1k分类数据集上，精度可达77.04%，相较MobileNetV3-Large x1.25精度提高0.64个百分点，同时在  Intel CPU 硬件上，预测速度可达 230 FPS ，相比 MobileNetV3-Large x1.25 预测速度提高 20%|[快速开始](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/PP-LCNetV2.md)|
|PaddleClas|PP-HGNet|GPU高性能骨干网络，在ImageNet 1k分类数据集上，精度可达79.83%、81.51%，同等速度下，相较ResNet34-D提高3.8个百分点，相较ResNet50-D提高2.4个百分点，在使用百度自研 SSLD 蒸馏策略后，精度相较ResNet50-D提高4.7个百分点。|[快速开始](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/PP-HGNet.md)|
|PaddleClas|PP-ShiTu|轻量图像识别系统，集成了目标检测、特征学习、图像检索等模块，广泛适用于各类图像识别任务，CPU上0.2s即可完成在10w+库的图像识别。|[快速开始](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.3#pp-shitu%E5%9B%BE%E5%83%8F%E8%AF%86%E5%88%AB%E7%B3%BB%E7%BB%9F%E4%BB%8B%E7%BB%8D)|
|PaddleDetection|PP-YOLO|基于YOLOv3优化的高精度目标检测模型，精度达到45.9%，在单卡V100上FP32推理速度为72.9 FPS, V100上开启TensorRT下FP16推理速度为155.6 FPS。|[快速开始](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/ppyolo/README_cn.md)|
|PaddleDetection|PP-YOLOv2|高精度目标检测模型，对比PP-YOLO， 精度提升 3.6%，达到49.5%；在 640*640 的输入尺寸下，速度可实现68.9FPS，采用 TensorRT 加速，FPS 还可达到106.5FPS。|[快速开始](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/ppyolo/README_cn.md)|
|PaddleDetection|PP-YOLOE|高精度云边一体SOTA目标检测模型，提供s/m/l/x版本，l版本COCO test2017数据集精度51.4%，V100预测速度78.1 FPS，支持混合精度训练，训练较PP-YOLOv2加速33%，全系列多尺度模型满足不同硬件算力需求，可适配服务器、边缘端GPU及其他服务器端AI加速卡。|[快速开始](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/ppyoloe/README_cn.md)|
|PaddleDetection|PP-PicoDet|超轻量级目标检测模型，提供xs/s/m/l四种尺寸，其中s版本参数量仅1.18m，却可达到32.5%mAP，相较YOLOX-Nano精度高6.7%，速度快26%，同时优化量化部署方案，实现在移动端部署加速30%+。|[快速开始](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet)|
|PaddleDetection|PP-Tracking|实时多目标跟踪工具，融合目标检测、行人重识别、轨迹融合等核心能力，提供行人车辆跟踪、跨镜头跟踪、多类别跟踪、小目标跟踪及流量技术等能力与产业应用。|[快速开始](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/deploy/pptracking/README_cn.md)|
|PaddleDetection|PP-TinyPose|超轻量级人体关键点检测算法，单人场景FP16推理可达到122FPS、精度51.8%AP，具有精度高速度快、检测人数无限制、微小目标效果好的特点。|[快速开始](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/keypoint/tiny_pose)|
|PaddleDetection|PP-Human|产业级实时行人分析工，支持属性分析、行为识别、流量计数/轨迹留存三大功能，覆盖目标检测、多目标跟踪、属性识别、关键点检测、行为识别和跨镜跟踪六大核心技术。|[快速开始](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/pphuman)|
|PaddleSeg|PP-HumanSeg|PP-HumanSeg是在大规模人像数据上训练的人像分割系列模型，提供了多种模型，满足在Web端、移动端、服务端多种使用场景的需求。其中PP-HumanSeg-Lite采用轻量级网络设计、连通性学习策略、非结构化稀疏技术，实现体积、速度和精度的SOTA平衡。（参数量137K，速度达95FPS，mIoU达93%）|[快速开始](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/configs/pp_humanseg_lite)|
|PaddleSeg|PP-HumanMatting|PP-HumanMatting通过低分辨粗预测和高分辨率Refine的两阶段设计，在增加小量计算量的情况下，有效保持了高分辨率(>2048)人像扣图中细节信息。|[快速开始](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/Matting)|
|PaddleSeg|PP-LiteSeg|PP-LiteSeg是通用轻量级语义分割模型，使用灵活高效的解码模块、统一注意力融合模块、轻量的上下文模块，针对Nvidia GPU上的产业级分割任务，实现精度和速度的SOTA平衡。在1080ti上精度为mIoU 72.0（Cityscapes数据集）时，速度高达273.6 FPS|[快速开始](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/configs/pp_liteseg)|
|PaddleSeg|PP-Matting|PP-Matting 通过引导流设计，实现语义引导下的高分辨率细节预测，进而实现trimap-free高精度图像抠图。在公开数据集Composition-1k和Distinctions-646测试集取得了SOTA的效果 。|[快速开始](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/Matting)|
|PaddleOCR|PP-OCR|PP-OCR是一个两阶段超轻量OCR系统，包括文本检测、方向分类器、文本识别三个部分，支持竖排文本识别。PP-OCR mobile中英文模型3.5M，英文数字模型2.8M。在通用场景下达到产业级SOTA标准|[快速开始](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/quickstart.md)|
|PaddleOCR|PP-OCRv2|PP-OCRv2在PP-OCR的基础上进行优化，平衡PP-OCR模型的精度和速度，效果相比PP-OCR mobile 提升7%；推理速度相比于PP-OCR server提升220%；支持80种多语言模型|[快速开始](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/quickstart.md)|
|PaddleOCR|PP-OCRv3|PP-OCRv3进一步在原先系统上优化，在中文场景效果相比于PP-OCRv2再提升5%，英文场景提升11%，80语种多语言模型平均识别准确率提升5%以上|[快速开始](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/quickstart.md)|
|PaddleOCR|PP-Structure|PP-Structure是一套智能文档分析系统，支持版面分析、表格识别（含Excel导出）、关键信息提取与DocVQA（含语义实体识别和关系抽取）|[快速开始](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/ppstructure/docs/quickstart.md)|
|PaddeleGAN|PP-MSVSR|高精度视频超分算法，提供1.45M和7.4M两种参数量大小的模型，峰值信噪比与结构相似度均高于其他开源算法，以PSNR 32.53、SSIM 0.9083达到业界SOTA，同时对输入视频的分辨率不限制，支持分辨率一次提升400%。|[快速开始](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md)|
|PaddleVideo|PP-TSM|高精度2D实用视频分类模型PP-TSM。在不增加参数量和计算量的情况下，在UCF-101、Kinetics-400等数据集上精度显著超过TSM原始模型|[快速开始](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/pp-tsm.md)|
|PaddleNLP|ERNIE 3.0-Medium|本模型是在文心大模型ERNIE 3.0 基础上通过**在线蒸馏技术**得到的轻量级模型，模型结构与 ERNIE 2.0 保持一致，相比 ERNIE 2.0 具有更强的中文效果。|[快速开始](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0)|
|PaddleSpeech|PP-ASR|PP-ASR是一套基于端到端神经网络结构模型的流式语音识别系统，支持实时语音识别服务，支持Language Model解码与个性化识别|[快速开始](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/asr/PPASR_cn.md)|
|PaddleSpeech|PP-TTS|PP-TTS是一套基于基于端到端神经网络结构的流式语音合成系统，支持流式声学模型与流式声码器，开源快速部署流式合成服务方案|[快速开始](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/tts/PPTTS_cn.md)|
|PaddleSpeech|PP-VPR|PP-VPR是一套声纹提取与检索系统，使用ECAPA-TDNN模型提取声纹特征，识别等错误率（EER，Equal error rate）低至0.95%，并且通过串联Mysql和Milvus，搭建完整的音频检索系统，实现毫秒级声音检索。|[快速开始](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/vpr/PPVPR_cn.md)|

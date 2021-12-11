# 社区模型库

飞桨目前包含100+个社区模型，覆盖CV、NLP、推荐等多个领域，详细内容如下表：

<table>
    <tr>
        <th>序号</th>
        <th>论文名称(链接)</th>
        <th>摘要</th>
        <th>数据集</th>
        <th width='10%'>快速开始</th>
    </tr>
    <tr>
        <td>1</td>
        <td><a href="https://arxiv.org/abs/1612.08242">YOLO9000: Better, Faster, Stronger</a></td>
        <td><details><summary>Abstract</summary><div>We present YOLO, a new approach to object detection. Prior work on objectdetection repurposes classifiers to perform detection. Instead, we frame objectdetection as a regression problem to spatially separated bounding boxes andassociated class probabilities. A single neural network predicts bounding boxesand class probabilities directly from full images in one evaluation. Since thewhole detection pipeline is a single network, it can be optimized end-to-enddirectly on detection performance.Our unified architecture is extremely fast. Our base YOLO model processesimages in real-time at 45 frames per second. A smaller version of the network,Fast YOLO, processes an astounding 155 frames per second while still achievingdouble the mAP of other real-time detectors. Compared to state-of-the-artdetection systems, YOLO makes more localization errors but is far less likelyto predict false detections where nothing exists. Finally, YOLO learns verygeneral representations of objects. It outperforms all other detection methods,including DPM and R-CNN, by a wide margin when generalizing from natural imagesto artwork on both the Picasso Dataset and the People-Art Dataset.</div></details></td>
        <td>YOLOv2 416x416 mAP: 76.8 参考原论文P4 Table3</td>
        <td><a href="https://github.com/nuaaceieyty/Paddle-YOLOv2">快速开始</a></td>
    </tr>
    <tr>
        <td>2</td>
        <td><a href="https://arxiv.org/abs/1506.02640">You Only Look Once: Unified, Real-Time Object Detection</a></td>
        <td><details><summary>Abstract</summary><div>Model efficiency has become increasingly important in computer vision. Inthis paper, we systematically study neural network architecture design choicesfor object detection and propose several key optimizations to improveefficiency. First, we propose a weighted bi-directional feature pyramid network(BiFPN), which allows easy and fast multiscale feature fusion; Second, wepropose a compound scaling method that uniformly scales the resolution, depth,and width for all backbone, feature network, and box/class prediction networksat the same time. Based on these optimizations and better backbones, we havedeveloped a new family of object detectors, called EfficientDet, whichconsistently achieve much better efficiency than prior art across a widespectrum of resource constraints. In particular, with single model andsingle-scale, our EfficientDet-D7 achieves state-of-the-art 55.1 AP on COCOtest-dev with 77M parameters and 410B FLOPs, being 4x - 9x smaller and using13x - 42x fewer FLOPs than previous detectors. Code is available atthis https URL.</div></details></td>
        <td>VOC2017 YOLO mAP:63.4 参考原论文Table1</td>
        <td><a href="https://github.com/sunlizhuang/YOLOv1-PaddlePaddle">快速开始</a></td>
    </tr>
    <tr>
        <td>3</td>
        <td><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html">EfficientDet: Scalable and Efficient Object Detection</a></td>
        <td><details><summary>Abstract</summary><div>Accurate depth estimation from images is a fundamental task in manyapplications including scene understanding and reconstruction. Existingsolutions for depth estimation often produce blurry approximations of lowresolution. This paper presents a convolutional neural network for computing ahigh-resolution depth map given a single RGB image with the help of transferlearning. Following a standard encoder-decoder architecture, we leveragefeatures extracted using high performing pre-trained networks when initializingour encoder along with augmentation and training strategies that lead to moreaccurate results. We show how, even for a very simple decoder, our method isable to achieve detailed high-resolution depth maps. Our network, with fewerparameters and training iterations, outperforms state-of-the-art on twodatasets and also produces qualitatively better results that capture objectboundaries more faithfully. Code and corresponding pre-trained weights are madepublicly available.</div></details></td>
        <td>efficientdet_d0 mAP: 33.6</td>
        <td><a href="https://github.com/GuoQuanhao/EfficientDet-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>4</td>
        <td><a href="https://arxiv.org/abs/1812.11941">High Quality Monocular Depth Estimation via Transfer Learning</a></td>
        <td><details><summary>Abstract</summary><div>We show that the YOLOv4 object detection neural network based on the CSPapproach, scales both up and down and is applicable to small and large networkswhile maintaining optimal speed and accuracy. We propose a network scalingapproach that modifies not only the depth, width, resolution, but alsostructure of the network. YOLOv4-large model achieves state-of-the-art results:55.5% AP (73.4% AP50) for the MS COCO dataset at a speed of ~16 FPS on TeslaV100, while with the test time augmentation, YOLOv4-large achieves 56.0% AP(73.3 AP50). To the best of our knowledge, this is currently the highestaccuracy on the COCO dataset among any published work. The YOLOv4-tiny modelachieves 22.0% AP (42.0% AP50) at a speed of 443 FPS on RTX 2080Ti, while byusing TensorRT, batch size = 4 and FP16-precision the YOLOv4-tiny achieves 1774FPS.</div></details></td>
        <td>NYU Depth v2  δ1：0.895 参考原论文table1</td>
        <td><a href="https://github.com/stunback/DenseDepth-paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>5</td>
        <td><a href="https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Scaled-YOLOv4_Scaling_Cross_Stage_Partial_Network_CVPR_2021_paper.html">Scaled-YOLOv4: Scaling Cross Stage Partial Network</a></td>
        <td><details><summary>Abstract</summary><div>The highest accuracy object detectors to date are based on a two-stageapproach popularized by R-CNN, where a classifier is applied to a sparse set ofcandidate object locations. In contrast, one-stage detectors that are appliedover a regular, dense sampling of possible object locations have the potentialto be faster and simpler, but have trailed the accuracy of two-stage detectorsthus far. In this paper, we investigate why this is the case. We discover thatthe extreme foreground-background class imbalance encountered during trainingof dense detectors is the central cause. We propose to address this classimbalance by reshaping the standard cross entropy loss such that itdown-weights the loss assigned to well-classified examples. Our novel FocalLoss focuses training on a sparse set of hard examples and prevents the vastnumber of easy negatives from overwhelming the detector during training. Toevaluate the effectiveness of our loss, we design and train a simple densedetector we call RetinaNet. Our results show that when trained with the focalloss, RetinaNet is able to match the speed of previous one-stage detectorswhile surpassing the accuracy of all existing state-of-the-art two-stagedetectors. Code is at: this https URL.</div></details></td>
        <td>YOLOv4-P5 mAP: 51.2</td>
        <td><a href="https://github.com/GuoQuanhao/ScaledYOLOv4-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>6</td>
        <td><a href="https://arxiv.org/pdf/1708.02002v2.pdf">Focal Loss for Dense Object Detection</a></td>
        <td><details><summary>Abstract</summary><div>There are a huge number of features which are said to improve ConvolutionalNeural Network (CNN) accuracy. Practical testing of combinations of suchfeatures on large datasets, and theoretical justification of the result, isrequired. Some features operate on certain models exclusively and for certainproblems exclusively, or only for small-scale datasets; while some features,such as batch-normalization and residual-connections, are applicable to themajority of models, tasks, and datasets. We assume that such universal featuresinclude Weighted-Residual-Connections (WRC), Cross-Stage-Partial-connections(CSP), Cross mini-Batch Normalization (CmBN), Self-adversarial-training (SAT)and Mish-activation. We use new features: WRC, CSP, CmBN, SAT, Mish activation,Mosaic data augmentation, CmBN, DropBlock regularization, and CIoU loss, andcombine some of them to achieve state-of-the-art results: 43.5% AP (65.7% AP50)for the MS COCO dataset at a realtime speed of ~65 FPS on Tesla V100. Sourcecode is at this https URL</div></details></td>
        <td>RetinaNet R-50-FPN 1x mAP: 35.7 参考github</td>
        <td><a href="https://github.com/FL77N/RetinaNet-Based-on-PPdet">快速开始</a></td>
    </tr>
    <tr>
        <td>7</td>
        <td><a href="https://arxiv.org/pdf/2004.10934v1.pdf">YOLOv4: Optimal Speed and Accuracy of Object Detection</a></td>
        <td><details><summary>Abstract</summary><div>Cascade is a classic yet powerful architecture that has boosted performanceon various tasks. However, how to introduce cascade to instance segmentationremains an open question. A simple combination of Cascade R-CNN and Mask R-CNNonly brings limited gain. In exploring a more effective approach, we find thatthe key to a successful instance segmentation cascade is to fully leverage thereciprocal relationship between detection and segmentation. In this work, wepropose a new framework, Hybrid Task Cascade (HTC), which differs in twoimportant aspects: (1) instead of performing cascaded refinement on these twotasks separately, it interweaves them for a joint multi-stage processing; (2)it adopts a fully convolutional branch to provide spatial context, which canhelp distinguishing hard foreground from cluttered background. Overall, thisframework can learn more discriminative features progressively whileintegrating complementary features together in each stage. Without bells andwhistles, a single HTC obtains 38.4 and 1.5 improvement over a strong CascadeMask R-CNN baseline on MSCOCO dataset. Moreover, our overall system achieves48.6 mask AP on the test-challenge split, ranking 1st in the COCO 2018Challenge Object Detection Task. Code is available at:this https URL.</div></details></td>
        <td>input size: 416x416,在MS COCO上mAP=41.2</td>
        <td><a href="https://github.com/nuaaceieyty/Paddle-YOLOv4">快速开始</a></td>
    </tr>
    <tr>
        <td>8</td>
        <td><a href="https://arxiv.org/abs/1901.07518">Hybrid Task Cascade for Instance Segmentation</a></td>
        <td><details><summary>Abstract</summary><div>We trained a convolutional neural network (CNN) to map raw pixels from asingle front-facing camera directly to steering commands. This end-to-endapproach proved surprisingly powerful. With minimum training data from humansthe system learns to drive in traffic on local roads with or without lanemarkings and on highways. It also operates in areas with unclear visualguidance such as in parking lots and on unpaved roads.The system automatically learns internal representations of the necessaryprocessing steps such as detecting useful road features with only the humansteering angle as the training signal. We never explicitly trained it todetect, for example, the outline of roads.Compared to explicit decomposition of the problem, such as lane markingdetection, path planning, and control, our end-to-end system optimizes allprocessing steps simultaneously. We argue that this will eventually lead tobetter performance and smaller systems. Better performance will result becausethe internal components self-optimize to maximize overall system performance,instead of optimizing human-selected intermediate criteria, e.g., lanedetection. Such criteria understandably are selected for ease of humaninterpretation which doesn't automatically guarantee maximum systemperformance. Smaller networks are possible because the system learns to solvethe problem with the minimal number of processing steps.We used an NVIDIA DevBox and Torch 7 for training and an NVIDIA DRIVE(TM) PXself-driving car computer also running Torch 7 for determining where to drive.The system operates at 30 frames per second (FPS).</div></details></td>
        <td>HTC R-50-FPN 1x box AP: 42.3 mask AP: 37.4 参考github</td>
        <td><a href="https://github.com/laihuihui/htc">快速开始</a></td>
    </tr>
    <tr>
        <td>9</td>
        <td><a href="https://arxiv.org/pdf/1604.07316.pdf">End to End Learning for Self-Driving Cars</a></td>
        <td><details><summary>Abstract</summary><div>Point cloud is an important type of geometric data structure. Due to itsirregular format, most researchers transform such data to regular 3D voxelgrids or collections of images. This, however, renders data unnecessarilyvoluminous and causes issues. In this paper, we design a novel type of neuralnetwork that directly consumes point clouds and well respects the permutationinvariance of points in the input. Our network, named PointNet, provides aunified architecture for applications ranging from object classification, partsegmentation, to scene semantic parsing. Though simple, PointNet is highlyefficient and effective. Empirically, it shows strong performance on par oreven better than state of the art. Theoretically, we provide analysis towardsunderstanding of what the network has learnt and why the network is robust withrespect to input perturbation and corruption.</div></details></td>
        <td>能在模拟器上运行不偏离路面，模拟器地址https://github.com/udacity/self-driving-car-sim</td>
        <td><a href="https://github.com/jm12138/car-behavioral-cloning-paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>10</td>
        <td><a href="https://arxiv.org/abs/1612.00593">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a></td>
        <td><details><summary>Abstract</summary><div>Few prior works study deep learning on point sets. PointNet by Qi et al. is apioneer in this direction. However, by design PointNet does not capture localstructures induced by the metric space points live in, limiting its ability torecognize fine-grained patterns and generalizability to complex scenes. In thiswork, we introduce a hierarchical neural network that applies PointNetrecursively on a nested partitioning of the input point set. By exploitingmetric space distances, our network is able to learn local features withincreasing contextual scales. With further observation that point sets areusually sampled with varying densities, which results in greatly decreasedperformance for networks trained on uniform densities, we propose novel setlearning layers to adaptively combine features from multiple scales.Experiments show that our network called PointNet++ is able to learn deep pointset features efficiently and robustly. In particular, results significantlybetter than state-of-the-art have been obtained on challenging benchmarks of 3Dpoint clouds.</div></details></td>
        <td>ModelNet40数据集分类精度89.2</td>
        <td><a href="https://github.com/Phimos/Paddle-PointNet">快速开始</a></td>
    </tr>
    <tr>
        <td>11</td>
        <td><a href="https://arxiv.org/abs/1706.02413">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a></td>
        <td><details><summary>Abstract</summary><div>The ability to perform pixel-wise semantic segmentation in real-time is ofparamount importance in mobile applications. Recent deep neural networks aimedat this task have the disadvantage of requiring a large number of floatingpoint operations and have long run-times that hinder their usability. In thispaper, we propose a novel deep neural network architecture named ENet(efficient neural network), created specifically for tasks requiring lowlatency operation. ENet is up to 18$\times$ faster, requires 75$\times$ lessFLOPs, has 79$\times$ less parameters, and provides similar or better accuracyto existing models. We have tested it on CamVid, Cityscapes and SUN datasetsand report on comparisons with existing state-of-the-art methods, and thetrade-offs between accuracy and processing time of a network. We presentperformance measurements of the proposed architecture on embedded systems andsuggest possible software improvements that could make ENet even faster.</div></details></td>
        <td>ModelNet40数据集分类精度90.7</td>
        <td><a href="https://github.com/SY-Xuan/pointnet_plus_plus_paddlepaddle">快速开始</a></td>
    </tr>
    <tr>
        <td>12</td>
        <td><a href="https://arxiv.org/abs/1606.02147">ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation</a></td>
        <td><details><summary>Abstract</summary><div>We develop a new edge detection algorithm that tackles two important issuesin this long-standing vision problem: (1) holistic image training andprediction; and (2) multi-scale and multi-level feature learning. Our proposedmethod, holistically-nested edge detection (HED), performs image-to-imageprediction by means of a deep learning model that leverages fully convolutionalneural networks and deeply-supervised nets. HED automatically learns richhierarchical representations (guided by deep supervision on side responses)that are important in order to approach the human ability resolve thechallenging ambiguity in edge and object boundary detection. We significantlyadvance the state-of-the-art on the BSD500 dataset (ODS F-score of .782) andthe NYU Depth dataset (ODS F-score of .746), and do so with an improved speed(0.4 second per image) that is orders of magnitude faster than some recentCNN-based edge detection algorithms.</div></details></td>
        <td>Cityscapes mIoU 58.3%</td>
        <td><a href="https://github.com/Shun14/enet">快速开始</a></td>
    </tr>
    <tr>
        <td>13</td>
        <td><a href="https://arxiv.org/abs/1504.06375">Holistically-Nested Edge Detection</a></td>
        <td><details><summary>Abstract</summary><div>Humans recognize the visual world at multiple levels: we effortlesslycategorize scenes and detect objects inside, while also identifying thetextures and surfaces of the objects along with their different compositionalparts. In this paper, we study a new task called Unified Perceptual Parsing,which requires the machine vision systems to recognize as many visual conceptsas possible from a given image. A multi-task framework called UPerNet and atraining strategy are developed to learn from heterogeneous image annotations.We benchmark our framework on Unified Perceptual Parsing and show that it isable to effectively segment a wide range of concepts from images. The trainednetworks are further applied to discover visual knowledge in natural scenes.Models are available at \url{this https URL}.</div></details></td>
        <td>BSD500 dataset (ODS F-score of .782)</td>
        <td><a href="https://github.com/txyugood/hed">快速开始</a></td>
    </tr>
    <tr>
        <td>14</td>
        <td><a href="https://arxiv.org/abs/1807.10221">Unified Perceptual Parsing for Scene Understanding</a></td>
        <td><details><summary>Abstract</summary><div>In this work we address the task of semantic image segmentation with DeepLearning and make three main contributions that are experimentally shown tohave substantial practical merit. First, we highlight convolution withupsampled filters, or 'atrous convolution', as a powerful tool in denseprediction tasks. Atrous convolution allows us to explicitly control theresolution at which feature responses are computed within Deep ConvolutionalNeural Networks. It also allows us to effectively enlarge the field of view offilters to incorporate larger context without increasing the number ofparameters or the amount of computation. Second, we propose atrous spatialpyramid pooling (ASPP) to robustly segment objects at multiple scales. ASPPprobes an incoming convolutional feature layer with filters at multiplesampling rates and effective fields-of-views, thus capturing objects as well asimage context at multiple scales. Third, we improve the localization of objectboundaries by combining methods from DCNNs and probabilistic graphical models.The commonly deployed combination of max-pooling and downsampling in DCNNsachieves invariance but has a toll on localization accuracy. We overcome thisby combining the responses at the final DCNN layer with a fully connectedConditional Random Field (CRF), which is shown both qualitatively andquantitatively to improve localization performance. Our proposed "DeepLab"system sets the new state-of-art at the PASCAL VOC-2012 semantic imagesegmentation task, reaching 79.7% mIOU in the test set, and advances theresults on three other datasets: PASCAL-Context, PASCAL-Person-Part, andCityscapes. All of our code is made publicly available online.</div></details></td>
        <td>Cityscapes mIoU 80.1%</td>
        <td><a href="https://github.com/Shun14/UperNet">快速开始</a></td>
    </tr>
    <tr>
        <td>15</td>
        <td><a href="https://arxiv.org/abs/1606.00915">DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs</a></td>
        <td><details><summary>Abstract</summary><div>We introduce a real-time, high-resolution background replacement techniquewhich operates at 30fps in 4K resolution, and 60fps for HD on a modern GPU. Ourtechnique is based on background matting, where an additional frame of thebackground is captured and used in recovering the alpha matte and theforeground layer. The main challenge is to compute a high-quality alpha matte,preserving strand-level hair details, while processing high-resolution imagesin real-time. To achieve this goal, we employ two neural networks; a basenetwork computes a low-resolution result which is refined by a second networkoperating at high-resolution on selective patches. We introduce two largescalevideo and image matting datasets: VideoMatte240K and PhotoMatte13K/85. Ourapproach yields higher quality results compared to the previousstate-of-the-art in background matting, while simultaneously yielding adramatic boost in both speed and resolution.</div></details></td>
        <td>Cityscapes mIoU 71.4%</td>
        <td><a href="https://github.com/Shun14/deeplab-paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>16</td>
        <td><a href="https://arxiv.org/abs/2012.07810">Real-Time High-Resolution Background Matting</a></td>
        <td><details><summary>Abstract</summary><div>The recently introduced panoptic segmentation task has renewed ourcommunity's interest in unifying the tasks of instance segmentation (for thingclasses) and semantic segmentation (for stuff classes). However, currentstate-of-the-art methods for this joint task use separate and dissimilarnetworks for instance and semantic segmentation, without performing any sharedcomputation. In this work, we aim to unify these methods at the architecturallevel, designing a single network for both tasks. Our approach is to endow MaskR-CNN, a popular instance segmentation method, with a semantic segmentationbranch using a shared Feature Pyramid Network (FPN) backbone. Surprisingly,this simple baseline not only remains effective for instance segmentation, butalso yields a lightweight, top-performing method for semantic segmentation. Inthis work, we perform a detailed study of this minimally extended version ofMask R-CNN with FPN, which we refer to as Panoptic FPN, and show it is a robustand accurate baseline for both tasks. Given its effectiveness and conceptualsimplicity, we hope our method can serve as a strong baseline and aid futureresearch in panoptic segmentation.</div></details></td>
        <td>PhotoMatte85 SAD8.65、MSE9.57</td>
        <td><a href="https://github.com/zackzhao1/BackgroundMattingV2-paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>17</td>
        <td><a href="https://arxiv.org/abs/1901.02446">Panoptic Feature Pyramid Networks</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>Cityscapes mIoU 75.8%</td>
        <td><a href="https://github.com/Shun14/panopticFPN-paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>18</td>
        <td><a href="https://arxiv.org/abs/1605.07146">Wide Residual Networks</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>CIFAR-10(WRN-28-20-dropout):96.55%</td>
        <td><a href="https://github.com/xmy0916/wide_resnet_paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>19</td>
        <td><a href="https://arxiv.org/pdf/1603.08511v5.pdf">Colorful Image Colorization</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>AuC：non-rebal=89.5%rebal=67.3%VGG Top-1 Class Acc=56%AMT Labeled Real=32.3%</td>
        <td><a href="https://github.com/Callifrey/Paddle-CIC">快速开始</a></td>
    </tr>
    <tr>
        <td>20</td>
        <td><a href="https://proceedings.neurips.cc/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf">Prototypical Networks for Few-shot Learning</a></td>
        <td><details><summary>Abstract</summary><div>Dropout is a powerful and widely used technique to regularize the training ofdeep neural networks. In this paper, we introduce a simple regularizationstrategy upon dropout in model training, namely R-Drop, which forces the outputdistributions of different sub models generated by dropout to be consistentwith each other. Specifically, for each training sample, R-Drop minimizes thebidirectional KL-divergence between the output distributions of two sub modelssampled by dropout. Theoretical analysis reveals that R-Drop reduces thefreedom of the model parameters and complements dropout. Experiments on$\bf{5}$ widely used deep learning tasks ($\bf{18}$ datasets in total),including neural machine translation, abstractive summarization, languageunderstanding, language modeling, and image classification, show that R-Drop isuniversally effective. In particular, it yields substantial improvements whenapplied to fine-tune large-scale pre-trained models, e.g., ViT, RoBERTa-large,and BART, and achieves state-of-the-art (SOTA) performances with the vanillaTransformer model on WMT14 English$\to$German translation ($\bf{30.91}$ BLEU)and WMT14 English$\to$French translation ($\bf{43.95}$ BLEU), even surpassingmodels trained with extra large-scale data and expert-designed advancedvariants of Transformer models. Our code is available atGitHub{\url{this https URL}}.</div></details></td>
        <td>1-shot=49.42%5-shot=68.2%（论文）</td>
        <td><a href="https://github.com/hrdwsong/ProtoNet-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>21</td>
        <td><a href="https://arxiv.org/pdf/2106.14448v1.pdf">R-Drop: Regularized Dropout for Neural Networks</a></td>
        <td><details><summary>Abstract</summary><div>Inspired by recent work in machine translation and object detection, weintroduce an attention based model that automatically learns to describe thecontent of images. We describe how we can train this model in a deterministicmanner using standard backpropagation techniques and stochastically bymaximizing a variational lower bound. We also show through visualization howthe model is able to automatically learn to fix its gaze on salient objectswhile generating the corresponding words in the output sequence. We validatethe use of attention with state-of-the-art performance on three benchmarkdatasets: Flickr8k, Flickr30k and MS COCO.</div></details></td>
        <td>ViT-B/16+RD=93.29</td>
        <td><a href="https://github.com/zbp-xxxp/R-Drop-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>22</td>
        <td><a href="https://arxiv.org/pdf/1502.03044.pdf">Show, Attend and Tell: Neural Image Caption Generation with Visual Attention</a></td>
        <td><details><summary>Abstract</summary><div>We present an approach to efficiently detect the 2D pose of multiple peoplein an image. The approach uses a nonparametric representation, which we referto as Part Affinity Fields (PAFs), to learn to associate body parts withindividuals in the image. The architecture encodes global context, allowing agreedy bottom-up parsing step that maintains high accuracy while achievingrealtime performance, irrespective of the number of people in the image. Thearchitecture is designed to jointly learn part locations and their associationvia two branches of the same sequential prediction process. Our method placedfirst in the inaugural COCO 2016 keypoints challenge, and significantly exceedsthe previous state-of-the-art result on the MPII Multi-Person benchmark, bothin performance and efficiency.</div></details></td>
        <td>bleu-1: 67%, bleu-2: 45.7%, bleu-3: 31.4%, bleu-4: 21.3%</td>
        <td><a href="https://github.com/Lieberk/Paddle-VA-Captioning">快速开始</a></td>
    </tr>
    <tr>
        <td>23</td>
        <td><a href="https://arxiv.org/pdf/1611.08050.pdf">Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields</a></td>
        <td><details><summary>Abstract</summary><div>Top-down visual attention mechanisms have been used extensively in imagecaptioning and visual question answering (VQA) to enable deeper imageunderstanding through fine-grained analysis and even multiple steps ofreasoning. In this work, we propose a combined bottom-up and top-down attentionmechanism that enables attention to be calculated at the level of objects andother salient image regions. This is the natural basis for attention to beconsidered. Within our approach, the bottom-up mechanism (based on FasterR-CNN) proposes image regions, each with an associated feature vector, whilethe top-down mechanism determines feature weightings. Applying this approach toimage captioning, our results on the MSCOCO test server establish a newstate-of-the-art for the task, achieving CIDEr / SPICE / BLEU-4 scores of117.9, 21.5 and 36.9, respectively. Demonstrating the broad applicability ofthe method, applying the same approach to VQA we obtain first place in the 2017VQA Challenge.</div></details></td>
        <td>full test set 75.6%</td>
        <td><a href="https://github.com/Xingyu-Romantic/RMPose_PAF">快速开始</a></td>
    </tr>
    <tr>
        <td>24</td>
        <td><a href="https://arxiv.org/pdf/1707.07998.pdf">Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering</a></td>
        <td><details><summary>Abstract</summary><div>Recent work has demonstrated that deep neural networks are vulnerable toadversarial examples---inputs that are almost indistinguishable from naturaldata and yet classified incorrectly by the network. In fact, some of the latestfindings suggest that the existence of adversarial attacks may be an inherentweakness of deep learning models. To address this problem, we study theadversarial robustness of neural networks through the lens of robustoptimization. This approach provides us with a broad and unifying view on muchof the prior work on this topic. Its principled nature also enables us toidentify methods for both training and attacking neural networks that arereliable and, in a certain sense, universal. In particular, they specify aconcrete security guarantee that would protect against any adversary. Thesemethods let us train networks with significantly improved resistance to a widerange of adversarial attacks. They also suggest the notion of security againsta first-order adversary as a natural and broad security guarantee. We believethat robustness against such well-defined classes of adversaries is animportant stepping stone towards fully resistant deep learning models. Code andpre-trained models are available at this https URLand this https URL.</div></details></td>
        <td>coco 2014 bleu-1=79.8%</td>
        <td><a href="https://github.com/Lieberk/Paddle-BUTD-Captioning">快速开始</a></td>
    </tr>
    <tr>
        <td>25</td>
        <td><a href="https://arxiv.org/pdf/1706.06083.pdf">Towards Deep Learning Models Resistant to Adversarial Attacks</a></td>
        <td><details><summary>Abstract</summary><div>This work introduces a novel convolutional network architecture for the taskof human pose estimation. Features are processed across all scales andconsolidated to best capture the various spatial relationships associated withthe body. We show how repeated bottom-up, top-down processing used inconjunction with intermediate supervision is critical to improving theperformance of the network. We refer to the architecture as a "stackedhourglass" network based on the successive steps of pooling and upsampling thatare done to produce a final set of predictions. State-of-the-art results areachieved on the FLIC and MPII benchmarks outcompeting all recent methods.</div></details></td>
        <td>PGD-steps100-restarts20-sourceA: 89.3%PGD-steps100-restarts20-sourceA: 95.7%PGD-steps40-restarts1-sourceB: 96.4%</td>
        <td><a href="https://github.com/hrdwsong/TDLMR2AA-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>26</td>
        <td><a href="https://arxiv.org/abs/1603.06937">Stacked Hourglass Networks for Human Pose Estimation</a></td>
        <td><details><summary>Abstract</summary><div>Imaging in low light is challenging due to low photon count and low SNR.Short-exposure images suffer from noise, while long exposure can induce blurand is often impractical. A variety of denoising, deblurring, and enhancementtechniques have been proposed, but their effectiveness is limited in extremeconditions, such as video-rate imaging at night. To support the development oflearning-based pipelines for low-light image processing, we introduce a datasetof raw short-exposure low-light images, with corresponding long-exposurereference images. Using the presented dataset, we develop a pipeline forprocessing low-light images, based on end-to-end training of afully-convolutional network. The network operates directly on raw sensor dataand replaces much of the traditional image processing pipeline, which tends toperform poorly on such data. We report promising results on the new dataset,analyze factors that affect performance, and highlight opportunities for futurework. The results are shown in the supplementary video atthis https URL</div></details></td>
        <td>MPII Human Pose Dataset， hourglass52 网络， size:384x384, mean@0.1: 0.366 size:256x256, mean@0.1: 0.317</td>
        <td><a href="https://github.com/txyugood/PaddlePose">快速开始</a></td>
    </tr>
    <tr>
        <td>27</td>
        <td><a href="https://arxiv.org/abs/1805.01934">Learning to See in the Dark</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>PSNR/SSIM: 28.88/0.787</td>
        <td><a href="https://github.com/WangChen0902/SID-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>28</td>
        <td><a href="https://arxiv.org/pdf/1711.10925.pdf">Deep Image Prior</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>8× super-resolution, avg psnr=24.15%</td>
        <td><a href="https://github.com/KunStats/Paddle-DIP">快速开始</a></td>
    </tr>
    <tr>
        <td>29</td>
        <td><a href="https://arxiv.org/pdf/1511.05644v2.pdf">Adversarial Autoencoders</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>MNIST测试集上，Log-likelihood(10K样本)：340±2</td>
        <td><a href="https://github.com/keil555/AAE_paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>30</td>
        <td><a href="https://arxiv.org/pdf/2004.11362v5.pdf">Supervised Contrastive Learning</a></td>
        <td><details><summary>Abstract</summary><div>We present a novel and practical deep fully convolutional neural networkarchitecture for semantic pixel-wise segmentation termed SegNet. This coretrainable segmentation engine consists of an encoder network, a correspondingdecoder network followed by a pixel-wise classification layer. The architectureof the encoder network is topologically identical to the 13 convolutionallayers in the VGG16 network. The role of the decoder network is to map the lowresolution encoder feature maps to full input resolution feature maps forpixel-wise classification. The novelty of SegNet lies is in the manner in whichthe decoder upsamples its lower resolution input feature map(s). Specifically,the decoder uses pooling indices computed in the max-pooling step of thecorresponding encoder to perform non-linear upsampling. This eliminates theneed for learning to upsample. The upsampled maps are sparse and are thenconvolved with trainable filters to produce dense feature maps. We compare ourproposed architecture with the widely adopted FCN and also with the well knownDeepLab-LargeFOV, DeconvNet architectures. This comparison reveals the memoryversus accuracy trade-off involved in achieving good segmentation performance.SegNet was primarily motivated by scene understanding applications. Hence, itis designed to be efficient both in terms of memory and computational timeduring inference. It is also significantly smaller in the number of trainableparameters than other competing architectures. We also performed a controlledbenchmark of SegNet and other architectures on both road scenes and SUN RGB-Dindoor scene segmentation tasks. We show that SegNet provides good performancewith competitive inference time and more efficient inference memory-wise ascompared to other architectures. We also provide a Caffe implementation ofSegNet and a web demo at this http URL.</div></details></td>
        <td>基于Contrastive loss，CIFAR10数据集在ResNet-50上，Top-1 Acc=96%</td>
        <td><a href="https://github.com/paddorch/SupContrast.paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>31</td>
        <td><a href="https://arxiv.org/pdf/1511.00561.pdf">SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation</a></td>
        <td><details><summary>Abstract</summary><div>We present a simple, fully-convolutional model for real-time instancesegmentation that achieves 29.8 mAP on MS COCO at 33.5 fps evaluated on asingle Titan Xp, which is significantly faster than any previous competitiveapproach. Moreover, we obtain this result after training on only one GPU. Weaccomplish this by breaking instance segmentation into two parallel subtasks:(1) generating a set of prototype masks and (2) predicting per-instance maskcoefficients. Then we produce instance masks by linearly combining theprototypes with the mask coefficients. We find that because this processdoesn't depend on repooling, this approach produces very high-quality masks andexhibits temporal stability for free. Furthermore, we analyze the emergentbehavior of our prototypes and show they learn to localize instances on theirown in a translation variant manner, despite being fully-convolutional.Finally, we also propose Fast NMS, a drop-in 12 ms faster replacement forstandard NMS that only has a marginal performance penalty.</div></details></td>
        <td>image size:360× 480;Dataset: Camvid;mIOU:60.1</td>
        <td><a href="https://github.com/stuartchen1949/segnet_paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>32</td>
        <td><a href="https://arxiv.org/pdf/1904.02689.pdf">YOLACT: Real-time Instance Segmentation</a></td>
        <td><details><summary>Abstract</summary><div>We develop an algorithm that can detect pneumonia from chest X-rays at alevel exceeding practicing radiologists. Our algorithm, CheXNet, is a 121-layerconvolutional neural network trained on ChestX-ray14, currently the largestpublicly available chest X-ray dataset, containing over 100,000 frontal-viewX-ray images with 14 diseases. Four practicing academic radiologists annotate atest set, on which we compare the performance of CheXNet to that ofradiologists. We find that CheXNet exceeds average radiologist performance onthe F1 metric. We extend CheXNet to detect all 14 diseases in ChestX-ray14 andachieve state of the art results on all 14 diseases.</div></details></td>
        <td>Image size:550 Resnet101-FPN FPS=33.5 mAP=29.8</td>
        <td><a href="https://github.com/jay-z20/yolact-paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>33</td>
        <td><a href="https://arxiv.org/pdf/1711.05225.pdf">CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning</a></td>
        <td><details><summary>Abstract</summary><div>We present a simple, fully-convolutional model for real-time (>30 fps)instance segmentation that achieves competitive results on MS COCO evaluated ona single Titan Xp, which is significantly faster than any previousstate-of-the-art approach. Moreover, we obtain this result after training ononly one GPU. We accomplish this by breaking instance segmentation into twoparallel subtasks: (1) generating a set of prototype masks and (2) predictingper-instance mask coefficients. Then we produce instance masks by linearlycombining the prototypes with the mask coefficients. We find that because thisprocess doesn't depend on repooling, this approach produces very high-qualitymasks and exhibits temporal stability for free. Furthermore, we analyze theemergent behavior of our prototypes and show they learn to localize instanceson their own in a translation variant manner, despite beingfully-convolutional. We also propose Fast NMS, a drop-in 12 ms fasterreplacement for standard NMS that only has a marginal performance penalty.Finally, by incorporating deformable convolutions into the backbone network,optimizing the prediction head with better anchor scales and aspect ratios, andadding a novel fast mask re-scoring branch, our YOLACT++ model can achieve 34.1mAP on MS COCO at 33.5 fps, which is fairly close to the state-of-the-artapproaches while still running at real-time.</div></details></td>
        <td>Resnet50 AUC：Atelectasis=0.707 Cardiomegaly=0.81 Effusion=0.73 Infiltration =0.61 Mass=0.56Nodule=0.71 Pneumonia=0.63 Pneumothorax=0.78</td>
        <td><a href="https://github.com/jm12138/Paddle-CheXNet">快速开始</a></td>
    </tr>
    <tr>
        <td>34</td>
        <td><a href="https://arxiv.org/pdf/1912.06218.pdf">YOLACT++: Better Real-time Instance Segmentation</a></td>
        <td><details><summary>Abstract</summary><div>We present a new method for efficient high-quality image segmentation ofobjects and scenes. By analogizing classical computer graphics methods forefficient rendering with over- and undersampling challenges faced in pixellabeling tasks, we develop a unique perspective of image segmentation as arendering problem. From this vantage, we present the PointRend (Point-basedRendering) neural network module: a module that performs point-basedsegmentation predictions at adaptively selected locations based on an iterativesubdivision algorithm. PointRend can be flexibly applied to both instance andsemantic segmentation tasks by building on top of existing state-of-the-artmodels. While many concrete implementations of the general idea are possible,we show that a simple design already achieves excellent results. Qualitatively,PointRend outputs crisp object boundaries in regions that are over-smoothed byprevious methods. Quantitatively, PointRend yields significant gains on COCOand Cityscapes, for both instance and semantic segmentation. PointRend'sefficiency enables output resolutions that are otherwise impractical in termsof memory or computation compared to existing approaches. Code has been madeavailable atthis https URL.</div></details></td>
        <td>Image size:550 Resnet50-FPN FPS=33.5 mAP=34.1</td>
        <td><a href="https://github.com/jay-z20/yolact2-paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>35</td>
        <td><a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.pdf">PointRend: Image Segmentation as Rendering</a></td>
        <td><details><summary>Abstract</summary><div>Recent works have widely explored the contextual dependencies to achieve moreaccurate segmentation results. However, most approaches rarely distinguishdifferent types of contextual dependencies, which may pollute the sceneunderstanding. In this work, we directly supervise the feature aggregation todistinguish the intra-class and inter-class context clearly. Specifically, wedevelop a Context Prior with the supervision of the Affinity Loss. Given aninput image and corresponding ground truth, Affinity Loss constructs an idealaffinity map to supervise the learning of Context Prior. The learned ContextPrior extracts the pixels belonging to the same category, while the reversedprior focuses on the pixels of different classes. Embedded into a conventionaldeep CNN, the proposed Context Prior Layer can selectively capture theintra-class and inter-class contextual dependencies, leading to robust featurerepresentation. To validate the effectiveness, we design an effective ContextPrior Network (CPNet). Extensive quantitative and qualitative evaluationsdemonstrate that the proposed model performs favorably against state-of-the-artsemantic segmentation approaches. More specifically, our algorithm achieves46.3% mIoU on ADE20K, 53.9% mIoU on PASCAL-Context, and 81.3% mIoU onCityscapes. Code is available at this https URL.</div></details></td>
        <td>cityscapes resnet50+FPN mIoU 78.3% 参考P5 architecture</td>
        <td><a href="https://github.com/CuberrChen/PointRend-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>36</td>
        <td><a href="https://arxiv.org/abs/2004.01547">Context Prior for Scene Segmentation</a></td>
        <td><details><summary>Abstract</summary><div>BiSeNet has been proved to be a popular two-stream network for real-timesegmentation. However, its principle of adding an extra path to encode spatialinformation is time-consuming, and the backbones borrowed from pretrainedtasks, e.g., image classification, may be inefficient for image segmentationdue to the deficiency of task-specific design. To handle these problems, wepropose a novel and efficient structure named Short-Term Dense Concatenatenetwork (STDC network) by removing structure redundancy. Specifically, wegradually reduce the dimension of feature maps and use the aggregation of themfor image representation, which forms the basic module of STDC network. In thedecoder, we propose a Detail Aggregation module by integrating the learning ofspatial information into low-level layers in single-stream manner. Finally, thelow-level features and deep features are fused to predict the finalsegmentation results. Extensive experiments on Cityscapes and CamVid datasetdemonstrate the effectiveness of our method by achieving promising trade-offbetween segmentation accuracy and inference speed. On Cityscapes, we achieve71.9% mIoU on the test set with a speed of 250.4 FPS on NVIDIA GTX 1080Ti,which is 45.2% faster than the latest methods, and achieve 76.8% mIoU with 97.0FPS while inferring on higher resolution images.</div></details></td>
        <td>cityscapes resnet101 mIoU 81.3% 参考论文table6</td>
        <td><a href="https://github.com/AndPuQing/ContextPrior_Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>37</td>
        <td><a href="https://arxiv.org/abs/2104.13188">Rethinking BiSeNet For Real-time Semantic Segmentation</a></td>
        <td><details><summary>Abstract</summary><div>We introduce a light-weight, power efficient, and general purposeconvolutional neural network, ESPNetv2, for modeling visual and sequentialdata. Our network uses group point-wise and depth-wise dilated separableconvolutions to learn representations from a large effective receptive fieldwith fewer FLOPs and parameters. The performance of our network is evaluated onfour different tasks: (1) object classification, (2) semantic segmentation, (3)object detection, and (4) language modeling. Experiments on these tasks,including image classification on the ImageNet and language modeling on thePenTree bank dataset, demonstrate the superior performance of our method overthe state-of-the-art methods. Our network outperforms ESPNet by 4-5% and has2-4x fewer FLOPs on the PASCAL VOC and the Cityscapes dataset. Compared toYOLOv2 on the MS-COCO object detection, ESPNetv2 delivers 4.4% higher accuracywith 6x fewer FLOPs. Our experiments show that ESPNetv2 is much more powerefficient than existing state-of-the-art efficient methods includingShuffleNets and MobileNets. Our code is open-source and available atthis https URL</div></details></td>
        <td>imgsize 512 × 1024 cityscapes STDC2-Seg50  mIoU 74.2% 参见table6</td>
        <td><a href="https://github.com/CuberrChen/STDCNet-Paddle/tree/master">快速开始</a></td>
    </tr>
    <tr>
        <td>38</td>
        <td><a href="https://arxiv.org/abs/1811.11431">ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network</a></td>
        <td><details><summary>Abstract</summary><div>Current semantic segmentation methods focus only on mining "local" context,i.e., dependencies between pixels within individual images, bycontext-aggregation modules (e.g., dilated convolution, neural attention) orstructure-aware optimization criteria (e.g., IoU-like loss). However, theyignore "global" context of the training data, i.e., rich semantic relationsbetween pixels across different images. Inspired by the recent advance inunsupervised contrastive representation learning, we propose a pixel-wisecontrastive framework for semantic segmentation in the fully supervisedsetting. The core idea is to enforce pixel embeddings belonging to a samesemantic class to be more similar than embeddings from different classes. Itraises a pixel-wise metric learning paradigm for semantic segmentation, byexplicitly exploring the structures of labeled pixels, which were rarelyexplored before. Our method can be effortlessly incorporated into existingsegmentation frameworks without extra overhead during testing. Weexperimentally show that, with famous segmentation models (i.e., DeepLabV3,HRNet, OCR) and backbones (i.e., ResNet, HR-Net), our method brings consistentperformance improvements across diverse datasets (i.e., Cityscapes,PASCAL-Context, COCO-Stuff, CamVid). We expect this work will encourage ourcommunity to rethink the current de facto training paradigm in fully supervisedsemantic segmentation.</div></details></td>
        <td>cityscapes ESPNetv2-val mIoU 66.4% 参见论文 fig7</td>
        <td><a href="https://github.com/justld/EspnetV2_paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>39</td>
        <td><a href="https://arxiv.org/abs/2101.11939">Exploring Cross-Image Pixel Contrast for Semantic Segmentation</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>HRNet-W48 Cityscaoes mIOU=80.18</td>
        <td><a href="https://github.com/justld/contrast_seg_paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>40</td>
        <td><a href="https://www.researchgate.net/profile/Yawei-Luo/publication/349909034_Category-Level_Adversarial_Adaptation_for_Semantic_Segmentation_using_Purified_Features/links/60a76d82299bf1031fba288c/Category-Level-Adversarial-Adaptation-for-Semantic-Segmentation-using-Purified-Features.pdf">Category-Level Adversarial Adaptation for Semantic Segmentation using Purified Features </a></td>
        <td><details><summary>Abstract</summary><div>We describe a new training methodology for generative adversarial networks.The key idea is to grow both the generator and discriminator progressively:starting from a low resolution, we add new layers that model increasingly finedetails as training progresses. This both speeds the training up and greatlystabilizes it, allowing us to produce images of unprecedented quality, e.g.,CelebA images at 1024^2. We also propose a simple way to increase the variationin generated images, and achieve a record inception score of 8.80 inunsupervised CIFAR10. Additionally, we describe several implementation detailsthat are important for discouraging unhealthy competition between the generatorand discriminator. Finally, we suggest a new metric for evaluating GAN results,both in terms of image quality and variation. As an additional contribution, weconstruct a higher-quality version of the CelebA dataset.</div></details></td>
        <td>Resnet101 Cityscapes mIoU 45.5%</td>
        <td><a href="https://github.com/zhangxin-xd/CLAN-paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>41</td>
        <td><a href="https://paperswithcode.com/paper/progressive-growing-of-gans-for-improved">Progressive Growing of GANs for Improved Quality, Stability, and Variation</a></td>
        <td><details><summary>Abstract</summary><div>Existing deep learning based image inpainting methods use a standardconvolutional network over the corrupted image, using convolutional filterresponses conditioned on both valid pixels as well as the substitute values inthe masked holes (typically the mean value). This often leads to artifacts suchas color discrepancy and blurriness. Post-processing is usually used to reducesuch artifacts, but are expensive and may fail. We propose the use of partialconvolutions, where the convolution is masked and renormalized to beconditioned on only valid pixels. We further include a mechanism toautomatically generate an updated mask for the next layer as part of theforward pass. Our model outperforms other methods for irregular masks. We showqualitative and quantitative comparisons with other methods to validate ourapproach.</div></details></td>
        <td>CelebA  MS-SSIM=0.2838, SWD=2.64(64)</td>
        <td><a href="https://github.com/GXU-GMU-MICCAI/PGAN-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>42</td>
        <td><a href="https://paperswithcode.com/paper/image-inpainting-for-irregular-holes-using">Image Inpainting for Irregular Holes Using Partial Convolutions</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>CelebA  人眼评估生成的图像（参考论文中展示的生成图片Figure8）</td>
        <td><a href="https://github.com/maruru-hub/PartialConv-paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>43</td>
        <td><a href="https://paperswithcode.com/paper/generative-adversarial-text-to-image">Generative Adversarial Text to Image Synthesis</a></td>
        <td><details><summary>Abstract</summary><div>Synthesizing high resolution photorealistic images has been a long-standingchallenge in machine learning. In this paper we introduce new methods for theimproved training of generative adversarial networks (GANs) for imagesynthesis. We construct a variant of GANs employing label conditioning thatresults in 128x128 resolution image samples exhibiting global coherence. Weexpand on previous work for image quality assessment to provide two newanalyses for assessing the discriminability and diversity of samples fromclass-conditional image synthesis models. These analyses demonstrate that highresolution samples provide class information not present in low resolutionsamples. Across 1000 ImageNet classes, 128x128 samples are more than twice asdiscriminable as artificially resized 32x32 samples. In addition, 84.7% of theclasses have samples exhibiting diversity comparable to real ImageNet data.</div></details></td>
        <td>Oxford-102  人眼评估生成的图像（参考论文中展示的生成图片）</td>
        <td><a href="https://github.com/Caimthefool/Paddle_T2I">快速开始</a></td>
    </tr>
    <tr>
        <td>44</td>
        <td><a href="https://paperswithcode.com/paper/conditional-image-synthesis-with-auxiliary">Conditional Image Synthesis With Auxiliary Classifier GANs</a></td>
        <td><details><summary>Abstract</summary><div>We introduce SinGAN, an unconditional generative model that can be learnedfrom a single natural image. Our model is trained to capture the internaldistribution of patches within the image, and is then able to generate highquality, diverse samples that carry the same visual content as the image.SinGAN contains a pyramid of fully convolutional GANs, each responsible forlearning the patch distribution at a different scale of the image. This allowsgenerating new samples of arbitrary size and aspect ratio, that havesignificant variability, yet maintain both the global structure and the finetextures of the training image. In contrast to previous single image GANschemes, our approach is not limited to texture images, and is not conditional(i.e. it generates samples from noise). User studies confirm that the generatedsamples are commonly confused to be real images. We illustrate the utility ofSinGAN in a wide range of image manipulation tasks.</div></details></td>
        <td>ImageNet  人眼评估生成的图像（参考论文中展示的生成图片）</td>
        <td><a href="https://github.com/Callifrey/ACGAN-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>45</td>
        <td><a href="https://paperswithcode.com/paper/singan-learning-a-generative-model-from-a">SinGAN: Learning a Generative Model from a Single Natural Image</a></td>
        <td><details><summary>Abstract</summary><div>We propose spatially-adaptive normalization, a simple but effective layer forsynthesizing photorealistic images given an input semantic layout. Previousmethods directly feed the semantic layout as input to the deep network, whichis then processed through stacks of convolution, normalization, andnonlinearity layers. We show that this is suboptimal as the normalizationlayers tend to ``wash away'' semantic information. To address the issue, wepropose using the input layout for modulating the activations in normalizationlayers through a spatially-adaptive, learned transformation. Experiments onseveral challenging datasets demonstrate the advantage of the proposed methodover existing approaches, regarding both visual fidelity and alignment withinput layouts. Finally, our model allows user control over both semantic andstyle. Code is available at this https URL .</div></details></td>
        <td>任意一张图片  人眼评估生成的图像（可参考论文中展示的生成图片Figure6）</td>
        <td><a href="https://github.com/icey-zhang/paddle_SinGAN">快速开始</a></td>
    </tr>
    <tr>
        <td>46</td>
        <td><a href="https://paperswithcode.com/paper/semantic-image-synthesis-with-spatially">Semantic Image Synthesis with Spatially-Adaptive Normalization</a></td>
        <td><details><summary>Abstract</summary><div>We present a generic image-to-image translation framework, pixel2style2pixel(pSp). Our pSp framework is based on a novel encoder network that directlygenerates a series of style vectors which are fed into a pretrained StyleGANgenerator, forming the extended W+ latent space. We first show that our encodercan directly embed real images into W+, with no additional optimization. Next,we propose utilizing our encoder to directly solve image-to-image translationtasks, defining them as encoding problems from some input domain into thelatent domain. By deviating from the standard invert first, edit latermethodology used with previous StyleGAN encoders, our approach can handle avariety of tasks even when the input image is not represented in the StyleGANdomain. We show that solving translation tasks through StyleGAN significantlysimplifies the training process, as no adversary is required, has bettersupport for solving tasks without pixel-to-pixel correspondence, and inherentlysupports multi-modal synthesis via the resampling of styles. Finally, wedemonstrate the potential of our framework on a variety of facialimage-to-image translation tasks, even when compared to state-of-the-artsolutions designed specifically for a single task, and further show that it canbe extended beyond the human facial domain.</div></details></td>
        <td>cityscapes  mIoU=62.3，accu=81.9，FID=71.8，及人眼观察可视化效果</td>
        <td><a href="https://github.com/ctkindle/SPADE-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>47</td>
        <td><a href="https://paperswithcode.com/paper/encoding-in-style-a-stylegan-encoder-for">Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation</a></td>
        <td><details><summary>Abstract</summary><div>The task of age transformation illustrates the change of an individual'sappearance over time. Accurately modeling this complex transformation over aninput facial image is extremely challenging as it requires making convincing,possibly large changes to facial features and head shape, while stillpreserving the input identity. In this work, we present an image-to-imagetranslation method that learns to directly encode real facial images into thelatent space of a pre-trained unconditional GAN (e.g., StyleGAN) subject to agiven aging shift. We employ a pre-trained age regression network to explicitlyguide the encoder in generating the latent codes corresponding to the desiredage. In this formulation, our method approaches the continuous aging process asa regression task between the input age and desired target age, providingfine-grained control over the generated image. Moreover, unlike approaches thatoperate solely in the latent space using a prior on the path controlling age,our method learns a more disentangled, non-linear path. Finally, we demonstratethat the end-to-end nature of our approach, coupled with the rich semanticlatent space of StyleGAN, allows for further editing of the generated images.Qualitative and quantitative evaluations show the advantages of our methodcompared to state-of-the-art approaches.</div></details></td>
        <td>CelebA   LPIPS=0.17，similarity=0.56，MSE=0.03 （task of StyleGAN Inversion）</td>
        <td><a href="https://github.com/771979972/Paddle_pSp">快速开始</a></td>
    </tr>
    <tr>
        <td>48</td>
        <td><a href="https://paperswithcode.com/paper/only-a-matter-of-style-age-transformation">Only a Matter of Style: Age Transformation Using a Style-Based Regression Model</a></td>
        <td><details><summary>Abstract</summary><div>We aim at accelerating super-resolution (SR) networks on large images(2K-8K). The large images are usually decomposed into small sub-images inpractical usages. Based on this processing, we found that different imageregions have different restoration difficulties and can be processed bynetworks with different capacities. Intuitively, smooth areas are easier tosuper-solve than complex textures. To utilize this property, we can adoptappropriate SR networks to process different sub-images after thedecomposition. On this basis, we propose a new solution pipeline -- ClassSRthat combines classification and SR in a unified framework. In particular, itfirst uses a Class-Module to classify the sub-images into different classesaccording to restoration difficulties, then applies an SR-Module to perform SRfor different classes. The Class-Module is a conventional classificationnetwork, while the SR-Module is a network container that consists of theto-be-accelerated SR network and its simplified versions. We further introducea new classification method with two losses -- Class-Loss and Average-Loss toproduce the classification results. After joint training, a majority ofsub-images will pass through smaller networks, thus the computational cost canbe significantly reduced. Experiments show that our ClassSR can help mostexisting methods (e.g., FSRCNN, CARN, SRResNet, RCAN) save up to 50% FLOPs onDIV8K datasets. This general framework can also be applied in other low-levelvision tasks.</div></details></td>
        <td>CelebA  人眼评估生成的图像（参考论文中展示的生成图片Figure4，6，8）</td>
        <td><a href="https://github.com/771979972/paddle-SAM">快速开始</a></td>
    </tr>
    <tr>
        <td>49</td>
        <td><a href="https://paperswithcode.com/paper/classsr-a-general-framework-to-accelerate">ClassSR: A General Framework to Accelerate Super-Resolution Networks by Data Characteristic</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>DIV2K PSNR=26.39， FLOPs=21.22G(65%)（Test2K，ClassSR-RCAN）</td>
        <td><a href="https://github.com/Scallions/ClassSR_paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>50</td>
        <td><a href="https://paperswithcode.com/paper/self-attention-generative-adversarial">Self-Attention Generative Adversarial Networks</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>ImageNet  FID=18.28  Inception score=52.52</td>
        <td><a href="https://github.com/Atmosphere-art/Self-Attention-GAN">快速开始</a></td>
    </tr>
    <tr>
        <td>51</td>
        <td><a href="https://arxiv.org/pdf/1908.07442v5.pdf">TabNet: Attentive Interpretable Tabular Learning</a></td>
        <td><details><summary>Abstract</summary><div>Though tremendous strides have been made in uncontrolled face detection,accurate and efficient face localisation in the wild remains an open challenge.This paper presents a robust single-stage face detector, named RetinaFace,which performs pixel-wise face localisation on various scales of faces bytaking advantages of joint extra-supervised and self-supervised multi-tasklearning. Specifically, We make contributions in the following five aspects:(1) We manually annotate five facial landmarks on the WIDER FACE dataset andobserve significant improvement in hard face detection with the assistance ofthis extra supervision signal. (2) We further add a self-supervised meshdecoder branch for predicting a pixel-wise 3D shape face information inparallel with the existing supervised branches. (3) On the WIDER FACE hard testset, RetinaFace outperforms the state of the art average precision (AP) by 1.1%(achieving AP equal to 91.4%). (4) On the IJB-C test set, RetinaFace enablesstate of the art methods (ArcFace) to improve their results in faceverification (TAR=89.59% for FAR=1e-6). (5) By employing light-weight backbonenetworks, RetinaFace can run real-time on a single CPU core for aVGA-resolution image. Extra annotations and code have been made available at:this https URL.</div></details></td>
        <td>在Forest Cover Type数据集上，acc=96.99%</td>
        <td><a href="https://github.com/txyugood/tabnet">快速开始</a></td>
    </tr>
    <tr>
        <td>52</td>
        <td><a href="https://arxiv.org/pdf/1905.00641v2.pdf">RetinaFace: Single-stage Dense Face Localisation in the Wild</a></td>
        <td><details><summary>Abstract</summary><div>We propose a novel Connectionist Text Proposal Network (CTPN) that accuratelylocalizes text lines in natural image. The CTPN detects a text line in asequence of fine-scale text proposals directly in convolutional feature maps.We develop a vertical anchor mechanism that jointly predicts location andtext/non-text score of each fixed-width proposal, considerably improvinglocalization accuracy. The sequential proposals are naturally connected by arecurrent neural network, which is seamlessly incorporated into theconvolutional network, resulting in an end-to-end trainable model. This allowsthe CTPN to explore rich context information of image, making it powerful todetect extremely ambiguous text. The CTPN works reliably on multi-scale andmulti- language text without further post-processing, departing from previousbottom-up methods requiring multi-step post-processing. It achieves 0.88 and0.61 F-measure on the ICDAR 2013 and 2015 benchmarks, surpass- ing recentresults [8, 35] by a large margin. The CTPN is computationally efficient with0:14s/image, by using the very deep VGG16 model [27]. Online demo is availableat: this http URL.</div></details></td>
        <td>MAP:52.318</td>
        <td><a href="https://github.com/GuoQuanhao/RetinaFace-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>53</td>
        <td><a href="https://arxiv.org/pdf/1609.03605v1.pdf">Detecting Text in Natural Image with Connectionist Text Proposal Network</a></td>
        <td><details><summary>Abstract</summary><div>In this paper we propose an implement a general convolutional neural network(CNN) building framework for designing real-time CNNs. We validate our modelsby creating a real-time vision system which accomplishes the tasks of facedetection, gender classification and emotion classification simultaneously inone blended step using our proposed CNN architecture. After presenting thedetails of the training procedure setup we proceed to evaluate on standardbenchmark sets. We report accuracies of 96% in the IMDB gender dataset and 66%in the FER-2013 emotion dataset. Along with this we also introduced the veryrecent real-time enabled guided back-propagation visualization technique.Guided back-propagation uncovers the dynamics of the weight changes andevaluates the learned features. We argue that the careful implementation ofmodern CNN architectures, the use of the current regularization methods and thevisualization of previously hidden features are necessary in order to reducethe gap between slow performances and real-time architectures. Our system hasbeen validated by its deployment on a Care-O-bot 3 robot used duringRoboCup@Home competitions. All our code, demos and pre-trained architectureshave been released under an open-source license in our public repository.</div></details></td>
        <td>icdar2015：0.61</td>
        <td><a href="https://github.com/BADBADBADBOY/paddle.ctpn">快速开始</a></td>
    </tr>
    <tr>
        <td>54</td>
        <td><a href="https://arxiv.org/pdf/1710.07557v1.pdf">Real-time Convolutional Neural Networks for Emotion and Gender Classification</a></td>
        <td><details><summary>Abstract</summary><div>The primary aim of single-image super-resolution is to constructhigh-resolution (HR) images from corresponding low-resolution (LR) inputs. Inprevious approaches, which have generally been supervised, the trainingobjective typically measures a pixel-wise average distance between thesuper-resolved (SR) and HR images. Optimizing such metrics often leads toblurring, especially in high variance (detailed) regions. We propose analternative formulation of the super-resolution problem based on creatingrealistic SR images that downscale correctly. We present an algorithmaddressing this problem, PULSE (Photo Upsampling via Latent Space Exploration),which generates high-resolution, realistic images at resolutions previouslyunseen in the literature. It accomplishes this in an entirely self-supervisedfashion and is not confined to a specific degradation operator used duringtraining, unlike previous methods (which require supervised training ondatabases of LR-HR image pairs). Instead of starting with the LR image andslowly adding detail, PULSE traverses the high-resolution natural imagemanifold, searching for images that downscale to the original LR image. This isformalized through the "downscaling loss," which guides exploration through thelatent space of a generative model. By leveraging properties ofhigh-dimensional Gaussians, we restrict the search space to guarantee realisticoutputs. PULSE thereby generates super-resolved images that both are realisticand downscale correctly. We show proof of concept of our approach in the domainof face super-resolution (i.e., face hallucination). We also present adiscussion of the limitations and biases of the method as currently implementedwith an accompanying model card with relevant metrics. Our method outperformsstate-of-the-art methods in perceptual quality at higher resolutions and scalefactors than previously possible.</div></details></td>
        <td>IMDB: 96%</td>
        <td><a href="https://github.com/wapping/FaceClassification">快速开始</a></td>
    </tr>
    <tr>
        <td>55</td>
        <td><a href="https://arxiv.org/pdf/2003.03808v3.pdf">PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>CelebA HQ  3.6</td>
        <td><a href="https://github.com/Martion-z/Paddle-PULSE">快速开始</a></td>
    </tr>
    <tr>
        <td>56</td>
        <td><a href="https://arxiv.org/pdf/1505.05424v2.pdf">Weight Uncertainty in Neural Networks</a></td>
        <td><details><summary>Abstract</summary><div>Learning from a few examples remains a key challenge in machine learning.Despite recent advances in important domains such as vision and language, thestandard supervised deep learning paradigm does not offer a satisfactorysolution for learning new concepts rapidly from little data. In this work, weemploy ideas from metric learning based on deep neural features and from recentadvances that augment neural networks with external memories. Our frameworklearns a network that maps a small labelled support set and an unlabelledexample to its label, obviating the need for fine-tuning to adapt to new classtypes. We then define one-shot learning problems on vision (using Omniglot,ImageNet) and language tasks. Our algorithm improves one-shot accuracy onImageNet from 87.6% to 93.2% and from 88.0% to 93.8% on Omniglot compared tocompeting approaches. We also demonstrate the usefulness of the same model onlanguage modeling by introducing a one-shot task on the Penn Treebank.</div></details></td>
        <td>MNIST测试集上，Test Error=1.32%</td>
        <td><a href="https://github.com/hrdwsong/BayesianCNN-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>57</td>
        <td><a href="https://paperswithcode.com/paper/matching-networks-for-one-shot-learning">Matching Networks for One Shot Learning</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>omniglot k-way=5, n-shot=1, 精度98.1</td>
        <td><a href="https://github.com/ranpeng-git/few-shot-paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>58</td>
        <td><a href="https://paperswithcode.com/paper/pixel-recurrent-neural-networks">Pixel Recurrent Neural Networks</a></td>
        <td><details><summary>Abstract</summary><div>In this work, we propose "Residual Attention Network", a convolutional neuralnetwork using attention mechanism which can incorporate with state-of-art feedforward network architecture in an end-to-end training fashion. Our ResidualAttention Network is built by stacking Attention Modules which generateattention-aware features. The attention-aware features from different moduleschange adaptively as layers going deeper. Inside each Attention Module,bottom-up top-down feedforward structure is used to unfold the feedforward andfeedback attention process into a single feedforward process. Importantly, wepropose attention residual learning to train very deep Residual AttentionNetworks which can be easily scaled up to hundreds of layers. Extensiveanalyses are conducted on CIFAR-10 and CIFAR-100 datasets to verify theeffectiveness of every module mentioned above. Our Residual Attention Networkachieves state-of-the-art object recognition performance on three benchmarkdatasets including CIFAR-10 (3.90% error), CIFAR-100 (20.45% error) andImageNet (4.8% single model and single crop, top-5 error). Note that, ourmethod achieves 0.6% top-1 accuracy improvement with 46% trunk depth and 69%forward FLOPs comparing to ResNet-200. The experiment also demonstrates thatour network is robust against noisy labels.</div></details></td>
        <td>NLL test 81.3</td>
        <td><a href="https://github.com/guguguzi/PixelCNN-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>59</td>
        <td><a href="https://paperswithcode.com/paper/residual-attention-network-for-image">Residual Attention Network for Image Classification</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>Attention-92 top1 error 4.99%</td>
        <td><a href="https://github.com/pooruss/Residual-Attention-Network-Paddle2.1.2">快速开始</a></td>
    </tr>
    <tr>
        <td>60</td>
        <td><a href="https://paperswithcode.com/paper/modeling-relational-data-with-graph">Modeling Relational Data with Graph Convolutional Networks</a></td>
        <td><details><summary>Abstract</summary><div>Recognizing arbitrary multi-character text in unconstrained naturalphotographs is a hard problem. In this paper, we address an equally hardsub-problem in this domain viz. recognizing arbitrary multi-digit numbers fromStreet View imagery. Traditional approaches to solve this problem typicallyseparate out the localization, segmentation, and recognition steps. In thispaper we propose a unified approach that integrates these three steps via theuse of a deep convolutional neural network that operates directly on the imagepixels. We employ the DistBelief implementation of deep neural networks inorder to train large, distributed neural networks on high quality images. Wefind that the performance of this approach increases with the depth of theconvolutional network, with the best performance occurring in the deepestarchitecture we trained, with eleven hidden layers. We evaluate this approachon the publicly available SVHN dataset and achieve over $96\%$ accuracy inrecognizing complete street numbers. We show that on a per-digit recognitiontask, we improve upon the state-of-the-art, achieving $97.84\%$ accuracy. Wealso evaluate this approach on an even more challenging dataset generated fromStreet View imagery containing several tens of millions of street numberannotations and achieve over $90\%$ accuracy. To further explore theapplicability of the proposed system to broader text recognition tasks, weapply it to synthetic distorted text from reCAPTCHA. reCAPTCHA is one of themost secure reverse turing tests that uses distorted text to distinguish humansfrom bots. We report a $99.8\%$ accuracy on the hardest category of reCAPTCHA.Our evaluations on both tasks indicate that at specific operating thresholds,the performance of the proposed system is comparable to, and in some casesexceeds, that of human operators.</div></details></td>
        <td>acc=95.83%</td>
        <td><a href="https://github.com/JiabenLi/rgcn_paddlepaddle">快速开始</a></td>
    </tr>
    <tr>
        <td>61</td>
        <td><a href="https://paperswithcode.com/paper/multi-digit-number-recognition-from-street">Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks</a></td>
        <td><details><summary>Abstract</summary><div>Deep neural networks are typically trained by optimizing a loss function withan SGD variant, in conjunction with a decaying learning rate, untilconvergence. We show that simple averaging of multiple points along thetrajectory of SGD, with a cyclical or constant learning rate, leads to bettergeneralization than conventional training. We also show that this StochasticWeight Averaging (SWA) procedure finds much flatter solutions than SGD, andapproximates the recent Fast Geometric Ensembling (FGE) approach with a singlemodel. Using SWA we achieve notable improvement in test accuracy overconventional SGD training on a range of state-of-the-art residual networks,PyramidNets, DenseNets, and Shake-Shake networks on CIFAR-10, CIFAR-100, andImageNet. In short, SWA is extremely easy to implement, improvesgeneralization, and has almost no computational overhead.</div></details></td>
        <td>Accuracy=95.65%</td>
        <td><a href="https://github.com/JennyVanessa/Paddle-VisualAttention">快速开始</a></td>
    </tr>
    <tr>
        <td>62</td>
        <td><a href="https://paperswithcode.com/paper/averaging-weights-leads-to-wider-optima-and">Averaging Weights Leads to Wider Optima and Better Generalization</a></td>
        <td><details><summary>Abstract</summary><div>Deep learning (DL) based semantic segmentation methods have been providingstate-of-the-art performance in the last few years. More specifically, thesetechniques have been successfully applied to medical image classification,segmentation, and detection tasks. One deep learning technique, U-Net, hasbecome one of the most popular for these applications. In this paper, wepropose a Recurrent Convolutional Neural Network (RCNN) based on U-Net as wellas a Recurrent Residual Convolutional Neural Network (RRCNN) based on U-Netmodels, which are named RU-Net and R2U-Net respectively. The proposed modelsutilize the power of U-Net, Residual Network, as well as RCNN. There areseveral advantages of these proposed architectures for segmentation tasks.First, a residual unit helps when training deep architecture. Second, featureaccumulation with recurrent residual convolutional layers ensures betterfeature representation for segmentation tasks. Third, it allows us to designbetter U-Net architecture with same number of network parameters with betterperformance for medical image segmentation. The proposed models are tested onthree benchmark datasets such as blood vessel segmentation in retina images,skin cancer segmentation, and lung lesion segmentation. The experimentalresults show superior performance on segmentation tasks compared to equivalentmodels including U-Net and residual U-Net (ResU-Net).</div></details></td>
        <td>VGG16+SWA 1budget, CIFAR10 top1=93.59</td>
        <td><a href="https://github.com/zlwangustc/SWA_paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>63</td>
        <td><a href="https://paperswithcode.com/paper/recurrent-residual-convolutional-neural">Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation</a></td>
        <td><details><summary>Abstract</summary><div>Over the last years, deep convolutional neural networks (ConvNets) havetransformed the field of computer vision thanks to their unparalleled capacityto learn high level semantic image features. However, in order to successfullylearn those features, they usually require massive amounts of manually labeleddata, which is both expensive and impractical to scale. Therefore, unsupervisedsemantic feature learning, i.e., learning without requiring manual annotationeffort, is of crucial importance in order to successfully harvest the vastamount of visual data that are available today. In our work we propose to learnimage features by training ConvNets to recognize the 2d rotation that isapplied to the image that it gets as input. We demonstrate both qualitativelyand quantitatively that this apparently simple task actually provides a verypowerful supervisory signal for semantic feature learning. We exhaustivelyevaluate our method in various unsupervised feature learning benchmarks and weexhibit in all of them state-of-the-art performance. Specifically, our resultson those benchmarks demonstrate dramatic improvements w.r.t. priorstate-of-the-art approaches in unsupervised representation learning and thussignificantly close the gap with supervised feature learning. For instance, inPASCAL VOC 2007 detection task our unsupervised pre-trained AlexNet modelachieves the state-of-the-art (among unsupervised methods) mAP of 54.4% that isonly 2.4 points lower from the supervised case. We get similarly strikingresults when we transfer our unsupervised learned features on various othertasks, such as ImageNet classification, PASCAL classification, PASCALsegmentation, and CIFAR-10 classification. The code and models of our paperwill be published on: this https URL .</div></details></td>
        <td>R2U-Net  F1-score=0.8171</td>
        <td><a href="https://github.com/zhaoxing-zstar/R2UNet-paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>64</td>
        <td><a href="https://paperswithcode.com/paper/unsupervised-representation-learning-by-1">Unsupervised Representation Learning by Predicting Image Rotations</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>RotNet+conv, CIFAR-10上top1=91.16</td>
        <td><a href="https://github.com/Dylan-get/Feature-Learning-RotNet">快速开始</a></td>
    </tr>
    <tr>
        <td>65</td>
        <td><a href="https://arxiv.org/pdf/1505.03540.pdf">Brain Tumor Segmentation with Deep Neural Networks</a></td>
        <td><details><summary>Abstract</summary><div>Point clouds provide a flexible geometric representation suitable forcountless applications in computer graphics; they also comprise the raw outputof most 3D data acquisition devices. While hand-designed features on pointclouds have long been proposed in graphics and vision, however, the recentoverwhelming success of convolutional neural networks (CNNs) for image analysissuggests the value of adapting insight from CNN to the point cloud world. Pointclouds inherently lack topological information so designing a model to recovertopology can enrich the representation power of point clouds. To this end, wepropose a new neural network module dubbed EdgeConv suitable for CNN-basedhigh-level tasks on point clouds including classification and segmentation.EdgeConv acts on graphs dynamically computed in each layer of the network. Itis differentiable and can be plugged into existing architectures. Compared toexisting modules operating in extrinsic space or treating each pointindependently, EdgeConv has several appealing properties: It incorporates localneighborhood information; it can be stacked applied to learn global shapeproperties; and in multi-layer systems affinity in feature space capturessemantic characteristics over potentially long distances in the originalembedding. We show the performance of our model on standard benchmarksincluding ModelNet40, ShapeNetPart, and S3DIS.</div></details></td>
        <td>BRATS 2013 test: Dice Complete 0.84, Core 0.72, Enhancing 0.57</td>
        <td><a href="https://github.com/tbymiracle/Brain-Tumor-Segmentation-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>66</td>
        <td><a href="https://arxiv.org/abs/1801.07829">Dynamic Graph CNN for Learning on Point Clouds</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>mIOU=85.2% 参考论文 Table.6的实现</td>
        <td><a href="https://github.com/JingfeiHuang/DGCNN-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>67</td>
        <td><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Adaptive_Pyramid_Context_Network_for_Semantic_Segmentation_CVPR_2019_paper.pdf">Adaptive Pyramid Context Network for Semantic Segmentation</a></td>
        <td><details><summary>Abstract</summary><div>Semi-supervised learning (SSL) provides an effective means of leveragingunlabeled data to improve a model's performance. In this paper, we demonstratethe power of a simple combination of two common SSL methods: consistencyregularization and pseudo-labeling. Our algorithm, FixMatch, first generatespseudo-labels using the model's predictions on weakly-augmented unlabeledimages. For a given image, the pseudo-label is only retained if the modelproduces a high-confidence prediction. The model is then trained to predict thepseudo-label when fed a strongly-augmented version of the same image. Despiteits simplicity, we show that FixMatch achieves state-of-the-art performanceacross a variety of standard semi-supervised learning benchmarks, including94.93% accuracy on CIFAR-10 with 250 labels and 88.61% accuracy with 40 -- just4 labels per class. Since FixMatch bears many similarities to existing SSLmethods that achieve worse performance, we carry out an extensive ablationstudy to tease apart the experimental factors that are most important toFixMatch's success. We make our code available atthis https URL.</div></details></td>
        <td>Cityscapes数据集mIOU=79.28%</td>
        <td><a href="https://github.com/Dylan-get/APCNet">快速开始</a></td>
    </tr>
    <tr>
        <td>68</td>
        <td><a href="https://arxiv.org/pdf/2001.07685v2.pdf">FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence</a></td>
        <td><details><summary>Abstract</summary><div>Over the last decade, Convolutional Neural Network (CNN) models have beenhighly successful in solving complex vision problems. However, these deepmodels are perceived as "black box" methods considering the lack ofunderstanding of their internal functioning. There has been a significantrecent interest in developing explainable deep learning models, and this paperis an effort in this direction. Building on a recently proposed method calledGrad-CAM, we propose a generalized method called Grad-CAM++ that can providebetter visual explanations of CNN model predictions, in terms of better objectlocalization as well as explaining occurrences of multiple object instances ina single image, when compared to state-of-the-art. We provide a mathematicalderivation for the proposed method, which uses a weighted combination of thepositive partial derivatives of the last convolutional layer feature maps withrespect to a specific class score as weights to generate a visual explanationfor the corresponding class label. Our extensive experiments and evaluations,both subjective and objective, on standard datasets showed that Grad-CAM++provides promising human-interpretable visual explanations for a given CNNarchitecture across multiple tasks including classification, image captiongeneration and 3D action recognition; as well as in new settings such asknowledge distillation.</div></details></td>
        <td>cifar 10, 40label: 93.6%, 250 label 95.31%, 4000 labels 95.77%</td>
        <td><a href="https://github.com/S-HuaBomb/FixMatch-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>69</td>
        <td><a href="https://arxiv.org/abs/1710.11063">Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks</a></td>
        <td><details><summary>Abstract</summary><div>This paper addresses the visualisation of image classification models, learntusing deep Convolutional Networks (ConvNets). We consider two visualisationtechniques, based on computing the gradient of the class score with respect tothe input image. The first one generates an image, which maximises the classscore [Erhan et al., 2009], thus visualising the notion of the class, capturedby a ConvNet. The second technique computes a class saliency map, specific to agiven image and class. We show that such maps can be employed for weaklysupervised object segmentation using classification ConvNets. Finally, weestablish the connection between the gradient-based ConvNet visualisationmethods and deconvolutional networks [Zeiler et al., 2013].</div></details></td>
        <td>可视化方法</td>
        <td><a href="https://github.com/vcowwy/paddle-grad-cam">快速开始</a></td>
    </tr>
    <tr>
        <td>70</td>
        <td><a href="https://arxiv.org/pdf/1312.6034v2.pdf">Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps</a></td>
        <td><details><summary>Abstract</summary><div>Relational reasoning is a central component of generally intelligentbehavior, but has proven difficult for neural networks to learn. In this paperwe describe how to use Relation Networks (RNs) as a simple plug-and-play moduleto solve problems that fundamentally hinge on relational reasoning. We testedRN-augmented networks on three tasks: visual question answering using achallenging dataset called CLEVR, on which we achieve state-of-the-art,super-human performance; text-based question answering using the bAbI suite oftasks; and complex reasoning about dynamic physical systems. Then, using acurated dataset called Sort-of-CLEVR we show that powerful convolutionalnetworks do not have a general capacity to solve relational questions, but cangain this capacity when augmented with RNs. Our work shows how a deep learningarchitecture equipped with an RN module can implicitly discover and learn toreason about entities and their relations.</div></details></td>
        <td>可视化方法</td>
        <td><a href="https://github.com/632652101/VisualizeCNN-Pd">快速开始</a></td>
    </tr>
    <tr>
        <td>71</td>
        <td><a href="https://arxiv.org/pdf/1706.01427v1.pdf">A simple neural network module for relational reasoning</a></td>
        <td><details><summary>Abstract</summary><div>We introduce the variational graph auto-encoder (VGAE), a framework forunsupervised learning on graph-structured data based on the variationalauto-encoder (VAE). This model makes use of latent variables and is capable oflearning interpretable latent representations for undirected graphs. Wedemonstrate this model using a graph convolutional network (GCN) encoder and asimple inner product decoder. Our model achieves competitive results on a linkprediction task in citation networks. In contrast to most existing models forunsupervised learning on graph-structured data and link prediction, our modelcan naturally incorporate node features, which significantly improvespredictive performance on a number of benchmark datasets.</div></details></td>
        <td>CLEVR数据集上准确率95.5%</td>
        <td><a href="https://github.com/yanchunyu71/relational-networks-paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>72</td>
        <td><a href="https://arxiv.org/abs/1611.07308">Variational Graph Auto-Encoders</a></td>
        <td><details><summary>Abstract</summary><div>This paper presents a self-supervised framework for training interest pointdetectors and descriptors suitable for a large number of multiple-view geometryproblems in computer vision. As opposed to patch-based neural networks, ourfully-convolutional model operates on full-sized images and jointly computespixel-level interest point locations and associated descriptors in one forwardpass. We introduce Homographic Adaptation, a multi-scale, multi-homographyapproach for boosting interest point detection repeatability and performingcross-domain adaptation (e.g., synthetic-to-real). Our model, when trained onthe MS-COCO generic image dataset using Homographic Adaptation, is able torepeatedly detect a much richer set of interest points than the initialpre-adapted deep model and any other traditional corner detector. The finalsystem gives rise to state-of-the-art homography estimation results on HPatcheswhen compared to LIFT, SIFT and ORB.</div></details></td>
        <td>CiteSeer1. AUC:90.8, AP:92</td>
        <td><a href="https://github.com/JiabenLi/gae_paddlepaddle">快速开始</a></td>
    </tr>
    <tr>
        <td>73</td>
        <td><a href="https://paperswithcode.com/paper/superpoint-self-supervised-interest-point">SuperPoint: Self-Supervised Interest Point Detection and Description</a></td>
        <td><details><summary>Abstract</summary><div>The demand of applying semantic segmentation model on mobile devices has beenincreasing rapidly. Current state-of-the-art networks have enormous amount ofparameters hence unsuitable for mobile devices, while other small memoryfootprint models follow the spirit of classification network and ignore theinherent characteristic of semantic segmentation. To tackle this problem, wepropose a novel Context Guided Network (CGNet), which is a light-weight andefficient network for semantic segmentation. We first propose the ContextGuided (CG) block, which learns the joint feature of both local feature andsurrounding context, and further improves the joint feature with the globalcontext. Based on the CG block, we develop CGNet which captures contextualinformation in all stages of the network and is specially tailored forincreasing segmentation accuracy. CGNet is also elaborately designed to reducethe number of parameters and save memory footprint. Under an equivalent numberof parameters, the proposed CGNet significantly outperforms existingsegmentation networks. Extensive experiments on Cityscapes and CamVid datasetsverify the effectiveness of the proposed approach. Specifically, without anypost-processing and multi-scale testing, the proposed CGNet achieves 64.8% meanIoU on Cityscapes with less than 0.5 M parameters. The source code for thecomplete system can be found at this https URL.</div></details></td>
        <td>MS-COCO 2014 HPatches Homography Estimation, e=1 0.460</td>
        <td><a href="https://github.com/vcowwy/paddle-superpoint">快速开始</a></td>
    </tr>
    <tr>
        <td>74</td>
        <td><a href="https://arxiv.org/pdf/1811.08201">CGNet: A Light-weight Context Guided Network for Semantic Segmentation</a></td>
        <td><details><summary>Abstract</summary><div>We focus on the challenging task of real-time semantic segmentation in thispaper. It finds many practical applications and yet is with fundamentaldifficulty of reducing a large portion of computation for pixel-wise labelinference. We propose an image cascade network (ICNet) that incorporatesmulti-resolution branches under proper label guidance to address thischallenge. We provide in-depth analysis of our framework and introduce thecascade feature fusion unit to quickly achieve high-quality segmentation. Oursystem yields real-time inference on a single GPU card with decent qualityresults evaluated on challenging datasets like Cityscapes, CamVid andCOCO-Stuff.</div></details></td>
        <td>在Cityscapes valset 上基于M3N21，mIOU=68.27%</td>
        <td><a href="https://github.com/632652101/VisualizeCNN-Pd">快速开始</a></td>
    </tr>
    <tr>
        <td>75</td>
        <td><a href="https://arxiv.org/pdf/1704.08545.pdf">ICNet for Real-Time Semantic Segmentation on High-Resolution Images</a></td>
        <td><details><summary>Abstract</summary><div>Recent deep learning based approaches have shown promising results for thechallenging task of inpainting large missing regions in an image. These methodscan generate visually plausible image structures and textures, but often createdistorted structures or blurry textures inconsistent with surrounding areas.This is mainly due to ineffectiveness of convolutional neural networks inexplicitly borrowing or copying information from distant spatial locations. Onthe other hand, traditional texture and patch synthesis approaches areparticularly suitable when it needs to borrow textures from the surroundingregions. Motivated by these observations, we propose a new deep generativemodel-based approach which can not only synthesize novel image structures butalso explicitly utilize surrounding image features as references during networktraining to make better predictions. The model is a feed-forward, fullyconvolutional neural network which can process images with multiple holes atarbitrary locations and with variable sizes during the test time. Experimentson multiple datasets including faces (CelebA, CelebA-HQ), textures (DTD) andnatural images (ImageNet, Places2) demonstrate that our proposed approachgenerates higher-quality inpainting results than existing ones. Code, demo andmodels are available at: this https URL.</div></details></td>
        <td>Cityscapes mIOU 69.6%</td>
        <td><a href="https://github.com/pooruss/ICNet-Paddle2.2.0rc">快速开始</a></td>
    </tr>
    <tr>
        <td>76</td>
        <td><a href="https://paperswithcode.com/paper/generative-image-inpainting-with-contextual">Generative Image Inpainting with Contextual Attention</a></td>
        <td><details><summary>Abstract</summary><div>We apply basic statistical reasoning to signal reconstruction by machinelearning -- learning to map corrupted observations to clean signals -- with asimple and powerful conclusion: it is possible to learn to restore images byonly looking at corrupted examples, at performance at and sometimes exceedingtraining using clean data, without explicit image priors or likelihood modelsof the corruption. In practice, we show that a single model learns photographicnoise removal, denoising synthetic Monte Carlo images, and reconstruction ofundersampled MRI scans -- all corrupted by different processes -- based onnoisy data only.</div></details></td>
        <td>L1Loss=8.6%，L2Loss=2.1%，PSNR=18.91，TVLoss=25.3%</td>
        <td><a href="https://github.com/JennyVanessa/Deepfill-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>77</td>
        <td><a href="https://paperswithcode.com/paper/noise2noise-learning-image-restoration">Noise2Noise: Learning Image Restoration without Clean Data</a></td>
        <td><details><summary>Abstract</summary><div>While humans easily recognize relations between data from different domainswithout any supervision, learning to automatically discover them is in generalvery challenging and needs many ground-truth pairs that illustrate therelations. To avoid costly pairing, we address the task of discoveringcross-domain relations given unpaired data. We propose a method based ongenerative adversarial networks that learns to discover relations betweendifferent domains (DiscoGAN). Using the discovered relations, our proposednetwork successfully transfers style from one domain to another whilepreserving key attributes such as orientation and face identity. Source codefor official implementation is publicly availablethis https URL</div></details></td>
        <td>Denoised 与clear image PSNR持平 （Gaussian noise (σ = 25)）</td>
        <td><a href="https://github.com/WangChen0902/noise2noise-paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>78</td>
        <td><a href="https://paperswithcode.com/paper/learning-to-discover-cross-domain-relations">Learning to Discover Cross-Domain Relations with Generative Adversarial Networks</a></td>
        <td><details><summary>Abstract</summary><div>We propose to restore old photos that suffer from severe degradation througha deep learning approach. Unlike conventional restoration tasks that can besolved through supervised learning, the degradation in real photos is complexand the domain gap between synthetic images and real old photos makes thenetwork fail to generalize. Therefore, we propose a novel triplet domaintranslation network by leveraging real photos along with massive syntheticimage pairs. Specifically, we train two variational autoencoders (VAEs) torespectively transform old photos and clean photos into two latent spaces. Andthe translation between these two latent spaces is learned with syntheticpaired data. This translation generalizes well to real photos because thedomain gap is closed in the compact latent space. Besides, to address multipledegradations mixed in one old photo, we design a global branch with apartialnonlocal block targeting to the structured defects, such as scratches and dustspots, and a local branch targeting to the unstructured defects, such as noisesand blurriness. Two branches are fused in the latent space, leading to improvedcapability to restore old photos from multiple defects. Furthermore, we applyanother face refinement network to recover fine details of faces in the oldphotos, thus ultimately generating photos with enhanced perceptual quality.With comprehensive experiments, the proposed pipeline demonstrates superiorperformance over state-of-the-art methods as well as existing commercial toolsin terms of visual quality for old photos restoration.</div></details></td>
        <td>可视化，参考图7，8，9</td>
        <td><a href="https://github.com/S-HuaBomb/DiscoGAN-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>79</td>
        <td><a href="https://paperswithcode.com/paper/old-photo-restoration-via-deep-latent-space">Old Photo Restoration via Deep Latent Space Translation</a></td>
        <td><details><summary>Abstract</summary><div>We present a novel method for constructing Variational Autoencoder (VAE).Instead of using pixel-by-pixel loss, we enforce deep feature consistencybetween the input and the output of a VAE, which ensures the VAE's output topreserve the spatial correlation characteristics of the input, thus leading theoutput to have a more natural visual appearance and better perceptual quality.Based on recent deep learning works such as style transfer, we employ apre-trained deep convolutional neural network (CNN) and use its hidden featuresto define a feature perceptual loss for VAE training. Evaluated on the CelebAface dataset, we show that our model produces better results than other methodsin the literature. We also show that our method can produce latent vectors thatcan capture the semantic information of face expressions and can be used toachieve state-of-the-art performance in facial attribute prediction.</div></details></td>
        <td>待更新  PSNR=23.33，SSIM= 0.69，LPIPS=0.25，FID=134.35（table2）</td>
        <td><a href="https://github.com/buriedms/Old2Life-Paddle.git">快速开始</a></td>
    </tr>
    <tr>
        <td>80</td>
        <td><a href="https://paperswithcode.com/paper/deep-feature-consistent-variational">Deep Feature Consistent Variational Autoencoder</a></td>
        <td><details><summary>Abstract</summary><div>Scene text detection, an important step of scene text reading systems, haswitnessed rapid development with convolutional neural networks. Nonetheless,two main challenges still exist and hamper its deployment to real-worldapplications. The first problem is the trade-off between speed and accuracy.The second one is to model the arbitrary-shaped text instance. Recently, somemethods have been proposed to tackle arbitrary-shaped text detection, but theyrarely take the speed of the entire pipeline into consideration, which may fallshort in practical this http URL this paper, we propose an efficient andaccurate arbitrary-shaped text detector, termed Pixel Aggregation Network(PAN), which is equipped with a low computational-cost segmentation head and alearnable post-processing. More specifically, the segmentation head is made upof Feature Pyramid Enhancement Module (FPEM) and Feature Fusion Module (FFM).FPEM is a cascadable U-shaped module, which can introduce multi-levelinformation to guide the better segmentation. FFM can gather the features givenby the FPEMs of different depths into a final feature for segmentation. Thelearnable post-processing is implemented by Pixel Aggregation (PA), which canprecisely aggregate text pixels by predicted similarity vectors. Experiments onseveral standard benchmarks validate the superiority of the proposed PAN. It isworth noting that our method can achieve a competitive F-measure of 79.9% at84.2 FPS on CTW1500.</div></details></td>
        <td>Average accuracies=88.73 （Table1.  VAE-345）</td>
        <td><a href="https://github.com/Dylan-get/Deep-Feature-Consistent-VAE">快速开始</a></td>
    </tr>
    <tr>
        <td>81</td>
        <td><a href="Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network (zhihu.com)">Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network</a></td>
        <td><details><summary>Abstract</summary><div>Attention-based scene text recognizers have gained huge success, whichleverages a more compact intermediate representation to learn 1d- or 2d-attention by a RNN-based encoder-decoder architecture. However, such methodssuffer from attention-drift problem because high similarity among encodedfeatures leads to attention confusion under the RNN-based local attentionmechanism. Moreover, RNN-based methods have low efficiency due to poorparallelization. To overcome these problems, we propose the MASTER, aself-attention based scene text recognizer that (1) not only encodes theinput-output attention but also learns self-attention which encodesfeature-feature and target-target relationships inside the encoder and decoderand (2) learns a more powerful and robust intermediate representation tospatial distortion, and (3) owns a great training efficiency because of hightraining parallelization and a high-speed inference because of an efficientmemory-cache mechanism. Extensive experiments on various benchmarks demonstratethe superior performance of our MASTER on both regular and irregular scenetext. Pytorch code can be found at this https URL,and Tensorflow code can be found at this https URL.</div></details></td>
        <td>ResNet18  ctw1500 0.806</td>
        <td><a href="https://github.com/JennyVanessa/PANet-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>82</td>
        <td><a href="MASTER: Multi-Aspect Non-local Network for Scene Text Recognition (arxiv.org)">MASTER: Multi-Aspect Non-local Network for Scene Text Recognition</a></td>
        <td><details><summary>Abstract</summary><div>Temporal action proposal generation is an important and challenging task invideo understanding, which aims at detecting all temporal segments containingaction instances of interest. The existing proposal generation approaches aregenerally based on pre-defined anchor windows or heuristic bottom-up boundarymatching strategies. This paper presents a simple and efficient framework(RTD-Net) for direct action proposal generation, by re-purposing aTransformer-alike architecture. To tackle the essential visual differencebetween time and space, we make three important improvements over the originaltransformer detection framework (DETR). First, to deal with slowness prior invideos, we replace the original Transformer encoder with a boundary attentivemodule to better capture long-range temporal information. Second, due to theambiguous temporal boundary and relatively sparse annotations, we present arelaxed matching scheme to relieve the strict criteria of single assignment toeach groundtruth. Finally, we devise a three-branch head to further improve theproposal confidence estimation by explicitly predicting its completeness.Extensive experiments on THUMOS14 and ActivityNet-1.3 benchmarks demonstratethe effectiveness of RTD-Net, on both tasks of temporal action proposalgeneration and temporal action detection. Moreover, due to its simplicity indesign, our framework is more efficient than previous proposal generationmethods, without non-maximum suppression post-processing. The code and modelsare made available at this https URL.</div></details></td>
        <td> IIIT5K: 95 SVT:90.6 IC03: 96.4 IC13:95.3IC15: 79.4  SVTP:834.5 CT80:84.5   avg: 89.81</td>
        <td><a href="https://github.com/S-HuaBomb/MASTER-paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>83</td>
        <td><a href="https://paperswithcode.com/paper/relaxed-transformer-decoders-for-direct">Relaxed Transformer Decoders for Direct Action Proposal Generation</a></td>
        <td><details><summary>Abstract</summary><div>Convolutional Neural Networks (CNNs) are the go-to model for computer vision.Recently, attention-based networks, such as the Vision Transformer, have alsobecome popular. In this paper we show that while convolutions and attention areboth sufficient for good performance, neither of them are necessary. We presentMLP-Mixer, an architecture based exclusively on multi-layer perceptrons (MLPs).MLP-Mixer contains two types of layers: one with MLPs applied independently toimage patches (i.e. "mixing" the per-location features), and one with MLPsapplied across patches (i.e. "mixing" spatial information). When trained onlarge datasets, or with modern regularization schemes, MLP-Mixer attainscompetitive scores on image classification benchmarks, with pre-training andinference cost comparable to state-of-the-art models. We hope that theseresults spark further research beyond the realms of well established CNNs andTransformers.</div></details></td>
        <td>THUMOS14, AR@50=41.52</td>
        <td><a href="https://github.com/rainyBJ/RTD_RePro">快速开始</a></td>
    </tr>
    <tr>
        <td>84</td>
        <td><a href="https://arxiv.org/pdf/2105.01601v4.pdf">MLP-Mixer: An all-MLP Architecture for Vision</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>Mixer-B/16CIFAR-10upstream:ImageNet96.72%upstream:ImageNet-21k96.82%（官方JAX repo提供）</td>
        <td><a href="https://github.com/MiuGod0126/Mlp-Mixer-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>85</td>
        <td><a href="https://arxiv.org/pdf/1603.09382v3.pdf">Deep Networks with Stochastic Depth</a></td>
        <td><details><summary>Abstract</summary><div>Multi-horizon forecasting problems often contain a complex mix of inputs --including static (i.e. time-invariant) covariates, known future inputs, andother exogenous time series that are only observed historically -- without anyprior information on how they interact with the target. While several deeplearning models have been proposed for multi-step prediction, they typicallycomprise black-box models which do not account for the full range of inputspresent in common scenarios. In this paper, we introduce the Temporal FusionTransformer (TFT) -- a novel attention-based architecture which combineshigh-performance multi-horizon forecasting with interpretable insights intotemporal dynamics. To learn temporal relationships at different scales, the TFTutilizes recurrent layers for local processing and interpretable self-attentionlayers for learning long-term dependencies. The TFT also uses specializedcomponents for the judicious selection of relevant features and a series ofgating layers to suppress unnecessary components, enabling high performance ina wide range of regimes. On a variety of real-world datasets, we demonstratesignificant performance improvements over existing benchmarks, and showcasethree practical interpretability use-cases of TFT.</div></details></td>
        <td>CIFAR-10 test error=5.25（论文指标）</td>
        <td><a href="https://github.com/zpc-666/Paddle-Stochastic-Depth-ResNet110">快速开始</a></td>
    </tr>
    <tr>
        <td>86</td>
        <td><a href="https://arxiv.org/abs/1912.09363">Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>Dataset: Electricity  P90 loss: 0.027</td>
        <td><a href="https://github.com/Scallions/tft_paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>87</td>
        <td><a href="https://www.ijcai.org/proceedings/2019/0568.pdf">Comprehensive Semi-Supervised Multi-Modal Learning</a></td>
        <td><details><summary>Abstract</summary><div>We present ViLBERT (short for Vision-and-Language BERT), a model for learningtask-agnostic joint representations of image content and natural language. Weextend the popular BERT architecture to a multi-modal two-stream model,pro-cessing both visual and textual inputs in separate streams that interactthrough co-attentional transformer layers. We pretrain our model through twoproxy tasks on the large, automatically collected Conceptual Captions datasetand then transfer it to multiple established vision-and-language tasks --visual question answering, visual commonsense reasoning, referring expressions,and caption-based image retrieval -- by making only minor additions to the basearchitecture. We observe significant improvements across tasks compared toexisting task-specific models -- achieving state-of-the-art on all four tasks.Our work represents a shift away from learning groundings between vision andlanguage only as part of task training and towards treating visual grounding asa pretrainable and transferable capability.</div></details></td>
        <td>Coverage：2.669 Average Precision：0.914  Ranking Loss：0.058 Example AUC：0.942 Micro AUC：0.94 Macro AUC：0.932</td>
        <td><a href="https://github.com/biubiu13/CMML-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>88</td>
        <td><a href="https://arxiv.org/abs/1908.02265">Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks</a></td>
        <td><details><summary>Abstract</summary><div>This paper proposes a new model for extracting an interpretable sentenceembedding by introducing self-attention. Instead of using a vector, we use a2-D matrix to represent the embedding, with each row of the matrix attending ona different part of the sentence. We also propose a self-attention mechanismand a special regularization term for the model. As a side effect, theembedding comes with an easy way of visualizing what specific parts of thesentence are encoded into the embedding. We evaluate our model on 3 differenttasks: author profiling, sentiment classification, and textual entailment.Results show that our model yields a significant performance gain compared toother sentence embedding methods in all of the 3 tasks.</div></details></td>
        <td>RefCOCO+-val=72.34</td>
        <td><a href="https://github.com/fuqianya/ViLBERT-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>89</td>
        <td><a href="https://arxiv.org/pdf/1703.03130v1.pdf">A Structured Self-attentive Sentence Embedding</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>SNLI测试集accuracy=84.4%（见论文Table 2）</td>
        <td><a href="https://github.com/paddorch/SelfAttnSent.paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>90</td>
        <td><a href="https://arxiv.org/pdf/1503.08895v5.pdf">End-To-End Memory Networks</a></td>
        <td><details><summary>Abstract</summary><div>This article offers an empirical exploration on the use of character-levelconvolutional networks (ConvNets) for text classification. We constructedseveral large-scale datasets to show that character-level convolutionalnetworks could achieve state-of-the-art or competitive results. Comparisons areoffered against traditional models such as bag of words, n-grams and theirTFIDF variants, and deep learning models such as word-based ConvNets andrecurrent neural networks.</div></details></td>
        <td> Penn Treebank测试集上ppl=111，Text8测试集上ppl=147</td>
        <td><a href="https://github.com/yulangz/End-To-End-Memory-Networks-in-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>91</td>
        <td><a href="https://arxiv.org/pdf/1509.01626v3.pdf">Character-level Convolutional Networks for Text Classification</a></td>
        <td><details><summary>Abstract</summary><div>Building open-domain chatbots is a challenging area for machine learningresearch. While prior work has shown that scaling neural models in the numberof parameters and the size of the data they are trained on gives improvedresults, we show that other ingredients are important for a high-performingchatbot. Good conversation requires a number of skills that an expertconversationalist blends in a seamless way: providing engaging talking pointsand listening to their partners, and displaying knowledge, empathy andpersonality appropriately, while maintaining a consistent persona. We show thatlarge scale models can learn these skills when given appropriate training dataand choice of generation strategy. We build variants of these recipes with 90M,2.7B and 9.4B parameter models, and make our models and code publiclyavailable. Human evaluations show our best models are superior to existingapproaches in multi-turn dialogue in terms of engagingness and humannessmeasurements. We then discuss the limitations of this work by analyzing failurecases of our models.</div></details></td>
        <td>Amazon Review Full测试集error rate=40.45%，Yahoo! Answers测试集error rate=28.80%</td>
        <td><a href="https://github.com/paddorch/CharCNN.paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>92</td>
        <td><a href="https://aclanthology.org/2021.eacl-main.24.pdf">Recipes for building an open-domain chatbot</a></td>
        <td><details><summary>Abstract</summary><div>Pre-trained language models like BERT and its variants have recently achievedimpressive performance in various natural language understanding tasks.However, BERT heavily relies on the global self-attention block and thussuffers large memory footprint and computation cost. Although all its attentionheads query on the whole input sequence for generating the attention map from aglobal perspective, we observe some heads only need to learn localdependencies, which means the existence of computation redundancy. We thereforepropose a novel span-based dynamic convolution to replace these self-attentionheads to directly model local dependencies. The novel convolution heads,together with the rest self-attention heads, form a new mixed attention blockthat is more efficient at both global and local context learning. We equip BERTwith this mixed attention design and build a ConvBERT model. Experiments haveshown that ConvBERT significantly outperforms BERT and its variants in variousdownstream tasks, with lower training cost and fewer model parameters.Remarkably, ConvBERTbase model achieves 86.4 GLUE score, 0.7 higher thanELECTRAbase, while using less than 1/4 training cost. Code and pre-trainedmodels will be released.</div></details></td>
        <td>-</td>
        <td><a href="https://github.com/kevinng77/blenderbot_paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>93</td>
        <td><a href="https://arxiv.org/pdf/2008.02496.pdf">ConvBERT: Improving BERT with Span-based Dynamic Convolution</a></td>
        <td><details><summary>Abstract</summary><div>Natural Language Processing (NLP) has recently achieved great success byusing huge pre-trained models with hundreds of millions of parameters. However,these models suffer from heavy model sizes and high latency such that theycannot be deployed to resource-limited mobile devices. In this paper, wepropose MobileBERT for compressing and accelerating the popular BERT model.Like the original BERT, MobileBERT is task-agnostic, that is, it can begenerically applied to various downstream NLP tasks via simple fine-tuning.Basically, MobileBERT is a thin version of BERT_LARGE, while equipped withbottleneck structures and a carefully designed balance between self-attentionsand feed-forward networks. To train MobileBERT, we first train a speciallydesigned teacher model, an inverted-bottleneck incorporated BERT_LARGE model.Then, we conduct knowledge transfer from this teacher to MobileBERT. Empiricalstudies show that MobileBERT is 4.3x smaller and 5.5x faster than BERT_BASEwhile achieving competitive results on well-known benchmarks. On the naturallanguage inference tasks of GLUE, MobileBERT achieves a GLUEscore o 77.7 (0.6lower than BERT_BASE), and 62 ms latency on a Pixel 4 phone. On the SQuADv1.1/v2.0 question answering task, MobileBERT achieves a dev F1 score of90.0/79.2 (1.5/2.1 higher than BERT_BASE).</div></details></td>
        <td>conv-bert-base模型指标：QNLI测试集accuracy=93.2%（见论文Table 3），SQuAD v1.1验证集上Exact Match=84.7%，F1=90.9%，SQuAD v2.0验证集Exact Match=80.6%，F1=83.1%（见论文Table 4）</td>
        <td><a href="https://github.com/JunnYu/paddle_convbert">快速开始</a></td>
    </tr>
    <tr>
        <td>94</td>
        <td><a href="https://arxiv.org/pdf/2004.02984.pdf">MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices</a></td>
        <td><details><summary>Abstract</summary><div>BERT adopts masked language modeling (MLM) for pre-training and is one of themost successful pre-training models. Since BERT neglects dependency amongpredicted tokens, XLNet introduces permuted language modeling (PLM) forpre-training to address this problem. However, XLNet does not leverage the fullposition information of a sentence and thus suffers from position discrepancybetween pre-training and fine-tuning. In this paper, we propose MPNet, a novelpre-training method that inherits the advantages of BERT and XLNet and avoidstheir limitations. MPNet leverages the dependency among predicted tokensthrough permuted language modeling (vs. MLM in BERT), and takes auxiliaryposition information as input to make the model see a full sentence and thusreducing the position discrepancy (vs. PLM in XLNet). We pre-train MPNet on alarge-scale dataset (over 160GB text corpora) and fine-tune on a variety ofdown-streaming tasks (GLUE, SQuAD, etc). Experimental results show that MPNetoutperforms MLM and PLM by a large margin, and achieves better results on thesetasks compared with previous state-of-the-art pre-trained methods (e.g., BERT,XLNet, RoBERTa) under the same model setting. The code and the pre-trainedmodels are available at: this https URL.</div></details></td>
        <td>"google/mobilebert-uncased"模型指标：MNLI验证集-m/mm accuracy=83.3/82.6（见论文table 4），SQuAD 1.1验证集F1/EM=90.0/82.9，SQuAD 2.0验证集F1/EM=79.2/76.2（见论文table 5）</td>
        <td><a href="https://github.com/nosaydomore/MobileBert_paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>95</td>
        <td><a href="https://arxiv.org/pdf/2004.09297.pdf">MPNet: Masked and Permuted Pre-training for Language Understanding</a></td>
        <td><details><summary>Abstract</summary><div>Large Transformer models routinely achieve state-of-the-art results on anumber of tasks but training these models can be prohibitively costly,especially on long sequences. We introduce two techniques to improve theefficiency of Transformers. For one, we replace dot-product attention by onethat uses locality-sensitive hashing, changing its complexity from O($L^2$) toO($L\log L$), where $L$ is the length of the sequence. Furthermore, we usereversible residual layers instead of the standard residuals, which allowsstoring activations only once in the training process instead of $N$ times,where $N$ is the number of layers. The resulting model, the Reformer, performson par with Transformer models while being much more memory-efficient and muchfaster on long sequences.</div></details></td>
        <td>"microsoft/mpnet-base"模型指标：QQP验证集accuracy=91.9（见论文Table 3），SQuAD 1.1 F1/EM (dev set)=92.7/86.9，SQuAD 2.0 F1/EM (dev set)=85.7/82.7（见论文Table 4）</td>
        <td><a href="https://github.com/junnyu/paddle-mpnet">快速开始</a></td>
    </tr>
    <tr>
        <td>96</td>
        <td><a href="https://arxiv.org/pdf/2001.04451.pdf">Reformer: The Efficient Transformer</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>-</td>
        <td><a href="https://github.com/junnyu/paddle_reformer">快速开始</a></td>
    </tr>
    <tr>
        <td>97</td>
        <td><a href="https://arxiv.org/pdf/2006.11316.pdf">SqueezeBERT: What can computer vision teach NLP about efficient neural networks?</a></td>
        <td><details><summary>Abstract</summary><div>Transfer learning, where a model is first pre-trained on a data-rich taskbefore being fine-tuned on a downstream task, has emerged as a powerfultechnique in natural language processing (NLP). The effectiveness of transferlearning has given rise to a diversity of approaches, methodology, andpractice. In this paper, we explore the landscape of transfer learningtechniques for NLP by introducing a unified framework that converts alltext-based language problems into a text-to-text format. Our systematic studycompares pre-training objectives, architectures, unlabeled data sets, transferapproaches, and other factors on dozens of language understanding tasks. Bycombining the insights from our exploration with scale and our new ``ColossalClean Crawled Corpus'', we achieve state-of-the-art results on many benchmarkscovering summarization, question answering, text classification, and more. Tofacilitate future work on transfer learning for NLP, we release our data set,pre-trained models, and code.</div></details></td>
        <td>"squeezebert/squeezebert-mnli-headless"模型指标：QQP验证集accuracy=89.4（见论文Table 2）4. SqueezeBERT模型加速比对比BERT-Base达到4.3x（见论文Table 2）</td>
        <td><a href="https://github.com/renmada/squeezebert-paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>98</td>
        <td><a href="https://arxiv.org/pdf/1910.10683.pdf">Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>"t5-base"模型指标：GLUE dev set上达到平均指标85.97，CNNDM dev set上达到ROUGE-2=20.90（见论文Table 15）</td>
        <td><a href="https://github.com/junnyu/paddle_t5">快速开始</a></td>
    </tr>
    <tr>
        <td>99</td>
        <td><a href="https://arxiv.org/pdf/1908.03557.pdf">VisualBERT: A Simple and Performant Baseline for Vision and Language</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>在VQA测试集上Test-Dev=70.80, Test-Std=71.00，NLVR验证集accuracy=67.4（见论文Table1，Table3）</td>
        <td><a href="https://github.com/chenkangyang/paddle_visual_bert">快速开始</a></td>
    </tr>
    <tr>
        <td>100</td>
        <td><a href="https://arxiv.org/pdf/2106.16038.pdf">ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>"ChineseBERT-large"模型指标：CMRC dev/test=70.70/78.05（见论文Table 2），XNLI dev/test=82.7/81.6（见论文Table 4），ChnSentiCorp dev/test=95.8/95.9</td>
        <td><a href="https://github.com/27182812/ChineseBERT_paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>101</td>
        <td><a href="https://arxiv.org/pdf/1909.05858.pdf">CTRL: A Conditional Transformer Language Model for Controllable Generation</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>-</td>
        <td><a href="https://github.com/junnyu/paddle_ctrl">快速开始</a></td>
    </tr>
    <tr>
        <td>102</td>
        <td><a href="https://arxiv.org/pdf/2006.03236.pdf">Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td> "funnel-transformer/xlarge"模型指标：QNLI验证集上accuracy=95.1%（见论文table 3），SQuAD v1.1验证集上F1/EM=94.7/89.0，SQuAD v2.0验证集F1/EM=90.4/87.6（见论文table 5）</td>
        <td><a href="https://github.com/chfhf/funnel-paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>103</td>
        <td><a href="https://paperswithcode.com/paper/few-shot-question-answering-by-pretraining">Splinter: Few-Shot Question Answering by Pretraining Span Selection</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>SQuAD 1.1验证集，16 examples F1=54.6, 128 examples F1=72.7，1024 Examples F1=82.8（见论文Table1）</td>
        <td><a href="https://github.com/zhoucz97/Splinter-paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>104</td>
        <td><a href="https://paperswithcode.com/paper/unified-language-model-pre-training-for">UNILMv1: Unified Language Model Pre-training for Natural Language Understanding and Generation</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>QNLI测试集达到92.7(见论文table11), CoQA验证集F1=82.5(见论文 table 7)</td>
        <td><a href="https://github.com/fuqiang-git-hub/unilmv1-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>105</td>
        <td><a href="https://paperswithcode.com/paper/bert-for-joint-intent-classification-and-slot">BERT for Joint Intent Classification and Slot Filling</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>在Snips测试集指标达到98.6, 97.0, 92.8; 在ATIS测试集上指标达到97.9, 96.0, 88.6 (见table2)</td>
        <td><a href="https://github.com/zhoucz97/JointBERT-paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>106</td>
        <td><a href="https://paperswithcode.com/paper/fastformer-additive-attention-is-all-you-need">Fastformer: Additive Attention Can Be All You Need</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>Amazon F1=43.23(见论文table4), Pubmed测试集R-L=34.81(见论文table6), </td>
        <td><a href="https://github.com/rainyBJ/Fastformer-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>107</td>
        <td><a href="https://arxiv.org/pdf/2009.09931v2.pdf">Field-Embedded Factorization Machines for Click-through rate prediction</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>criteo auc >0.8</td>
        <td><a href="https://github.com/thinkall/deepfefm">快速开始</a></td>
    </tr>
    <tr>
        <td>108</td>
        <td><a href="https://arxiv.org/pdf/1906.00091v1.pdf">Deep Learning Recommendation Model for Personalization and Recommendation Systems</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>criteo auc > 0.79</td>
        <td><a href="https://github.com/Andy1314Chen/DLRM-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>109</td>
        <td><a href="https://www.ijcai.org/Proceedings/2020/0434.pdf">A Dual Input-aware Factorization Machine for CTR Prediction</a></td>
        <td><details><summary>Abstract</summary><div>Click through rate (CTR) estimation is a fundamental task in personalizedadvertising and recommender systems. Recent years have witnessed the success ofboth the deep learning based model and attention mechanism in various tasks incomputer vision (CV) and natural language processing (NLP). How to combine theattention mechanism with deep CTR model is a promising direction because it mayensemble the advantages of both sides. Although some CTR model such asAttentional Factorization Machine (AFM) has been proposed to model the weightof second order interaction features, we posit the evaluation of featureimportance before explicit feature interaction procedure is also important forCTR prediction tasks because the model can learn to selectively highlight theinformative features and suppress less useful ones if the task has many inputfeatures. In this paper, we propose a new neural CTR model named FieldAttentive Deep Field-aware Factorization Machine (FAT-DeepFFM) by combining theDeep Field-aware Factorization Machine (DeepFFM) with Compose-Excitationnetwork (CENet) field attention mechanism which is proposed by us as anenhanced version of Squeeze-Excitation network (SENet) to highlight the featureimportance. We conduct extensive experiments on two real-world datasets and theexperiment results show that FAT-DeepFFM achieves the best performance andobtains different improvements over the state-of-the-art methods. We alsocompare two kinds of attention mechanisms (attention before explicit featureinteraction vs. attention after explicit feature interaction) and demonstratethat the former one outperforms the latter one significantly.</div></details></td>
        <td>crito auc >0.799</td>
        <td><a href="https://github.com/Andy1314Chen/DIFM-Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>110</td>
        <td><a href="https://arxiv.org/abs/1905.06336">FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>crito AUC>=0.8099</td>
        <td><a href="https://github.com/LinJayan/FAT_DeepFFM_Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>111</td>
        <td><a href="https://arxiv.org/pdf/1904.06690v2.pdf">BERT4Rec：Sequential Recommendation with Bidirectional Encoder Representations from Transformer</a></td>
        <td><details><summary>Abstract</summary><div>Top-$N$ sequential recommendation models each user as a sequence of itemsinteracted in the past and aims to predict top-$N$ ranked items that a userwill likely interact in a `near future'. The order of interaction implies thatsequential patterns play an important role where more recent items in asequence have a larger impact on the next item. In this paper, we propose aConvolutional Sequence Embedding Recommendation Model (\emph{Caser}) as asolution to address this requirement. The idea is to embed a sequence of recentitems into an `image' in the time and latent spaces and learn sequentialpatterns as local features of the image using convolutional filters. Thisapproach provides a unified and flexible network structure for capturing bothgeneral preferences and sequential patterns. The experiments on public datasetsdemonstrated that Caser consistently outperforms state-of-the-art sequentialrecommendation methods on a variety of common evaluation metrics.</div></details></td>
        <td>1、Beauty HR@10=0.30252、Steam HR@10=0.40133、ML-1m HR@10=0.69704、ML-20m HR@10=0.7473</td>
        <td><a href="https://github.com/jinweiluo/BERT4Rec_AC">快速开始</a></td>
    </tr>
    <tr>
        <td>112</td>
        <td><a href="https://arxiv.org/pdf/1809.07426v1.pdf">Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>1、MovieLens MAP=0.15072、Gowalla MAP=0.09283、Foursquare MAP=0.09094、Tmall MAP=0.0310</td>
        <td><a href="https://github.com/LinJayan/Caser_Paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>113</td>
        <td><a href="https://arxiv.org/pdf/1808.09781v1.pdf">SASRec：Self-Attentive Sequential Recommendation</a></td>
        <td><details><summary>Abstract</summary><div>Existing Collaborative Filtering (CF) methods are mostly designed based onthe idea of matching, i.e., by learning user and item embeddings from datausing shallow or deep models, they try to capture the associative relevancepatterns in data, so that a user embedding can be matched with relevant itemembeddings using designed or learned similarity functions. However, as acognition rather than a perception intelligent task, recommendation requiresnot only the ability of pattern recognition and matching from data, but alsothe ability of cognitive reasoning in data. In this paper, we propose toadvance Collaborative Filtering (CF) to Collaborative Reasoning (CR), whichmeans that each user knows part of the reasoning space, and they collaboratefor reasoning in the space to estimate preferences for each other. Technically,we propose a Neural Collaborative Reasoning (NCR) framework to bridge learningand reasoning. Specifically, we integrate the power of representation learningand logical reasoning, where representations capture similarity patterns indata from perceptual perspectives, and logic facilitates cognitive reasoningfor informed decision making. An important challenge, however, is to bridgedifferentiable neural networks and symbolic reasoning in a shared architecturefor optimization and inference. To solve the problem, we propose a modularizedreasoning architecture, which learns logical operations such as AND ($\wedge$),OR ($\vee$) and NOT ($\neg$) as neural modules for implication reasoning($\rightarrow$). In this way, logical expressions can be equivalently organizedas neural networks, so that logical reasoning and prediction can be conductedin a continuous space. Experiments on real-world datasets verified theadvantages of our framework compared with both shallow, deep and reasoningmodels.</div></details></td>
        <td>Hit Rate@10(Recall@10; Precision@10) andNDCG@10,</td>
        <td><a href="https://github.com/paddorch/SASRec.paddle">快速开始</a></td>
    </tr>
    <tr>
        <td>114</td>
        <td><a href="https://arxiv.org/pdf/2005.08129.pdf">Neural Collaborative Reasoning</a></td>
        <td><details><summary>Abstract</summary><div></div></details></td>
        <td>ML100K :HR@K >0.68</td>
        <td><a href="https://github.com/gsq7474741/Paddle-NCR">快速开始</a></td>
    </tr>

</table>

# Sound tagging in paddlepaddle

## Introduction

<b> Sound (events) classification/tagging/detection/localization in PaddlePaddle. </b>








### Finetuning on ESC-50 dataset
- Use pretrained model Cnn14 from <a href="https://github.com/qiuqiangkong/audioset_tagging_cnn">audioset_tagging_cnn</a>[1]. The model is pretrained on audioset, which is the largest weakly-labelled(only class info, without exact event time location) sound event dataset. 

- Add a linear head (2048x50) for <a href="https://github.com/karolpiczak/ESC-50">ESC-50</a> sound classifation dataset. Use softmax and cross-entropy loss in paddle. Use logsoftmax and NLL loss in torch 
- No extra dropout is added except those already existed in original CNN14
- No extra data auemgntation except random cropping the mel-spectrograms. In train/eval/test, spectrogram is of 384 frames, randomly cropped out of 501 frames (so the eval/test result is not determinitic).
- Mel-spectrograms are extrated before training starts, for faster training. mel-config: sr=32000,win-size=1024,hop-size=320,mel_bins=64, fmin=50,fmax=14000. Hence, 5s audio results in 501 mel frames. 
- In training, all layers are training simultaneously, using lr=3e-4, which decreases by multiplying 0.1 at every 20 epochs.

- No weight-decaying is used

- In training, spectrogram is of 384 frames. Random cropping is used.
- In test/eval, all 501 frames are used (so the result is also determinitic. 
- Same lr decreasing scheduler as in baseline
- Use large dropout: 



### Results
<b>ESC-50 Test acc (5-fold cross-validation) </b>
- Paddle: 0.920


Without any tricks, paddle acheived 0.920 acc, ranking <b> No. 2 </b> in the leader board. 
## Training
### Requirements
```
pip install -r requirements.txt
```

### Data preparation

The <a ref="https://github.com/karolpiczak/ESC-50/">**ESC-50 dataset**</a> is a labeled collection of 2000 environmental audio recordings suitable for benchmarking methods of environmental sound classification.

The dataset consists of 5-second-long recordings organized into 50 semantical classes (with 40 examples per class) loosely arranged into 5 major categories:

| <sub>Animals</sub> | <sub>Natural soundscapes & water sounds </sub> | <sub>Human, non-speech sounds</sub> | <sub>Interior/domestic sounds</sub> | <sub>Exterior/urban noises</sub> |
| :--- | :--- | :--- | :--- | :--- |
| <sub>Dog</sub> | <sub>Rain</sub> | <sub>Crying baby</sub> | <sub>Door knock</sub> | <sub>Helicopter</sub></sub> |
| <sub>Rooster</sub> | <sub>Sea waves</sub> | <sub>Sneezing</sub> | <sub>Mouse click</sub> | <sub>Chainsaw</sub> |
| <sub>Pig</sub> | <sub>Crackling fire</sub> | <sub>Clapping</sub> | <sub>Keyboard typing</sub> | <sub>Siren</sub> |
| <sub>Cow</sub> | <sub>Crickets</sub> | <sub>Breathing</sub> | <sub>Door, wood creaks</sub> | <sub>Car horn</sub> |
| <sub>Frog</sub> | <sub>Chirping birds</sub> | <sub>Coughing</sub> | <sub>Can opening</sub> | <sub>Engine</sub> |
| <sub>Cat</sub> | <sub>Water drops</sub> | <sub>Footsteps</sub> | <sub>Washing machine</sub> | <sub>Train</sub> |
| <sub>Hen</sub> | <sub>Wind</sub> | <sub>Laughing</sub> | <sub>Vacuum cleaner</sub> | <sub>Church bells</sub> |
| <sub>Insects (flying)</sub> | <sub>Pouring water</sub> | <sub>Brushing teeth</sub> | <sub>Clock alarm</sub> | <sub>Airplane</sub> |
| <sub>Sheep</sub> | <sub>Toilet flush</sub> | <sub>Snoring</sub> | <sub>Clock tick</sub> | <sub>Fireworks</sub> |
| <sub>Crow</sub> | <sub>Thunderstorm</sub> | <sub>Drinking, sipping</sub> | <sub>Glass breaking</sub> | <sub>Hand saw</sub> |


To get the data ready for training, first config the wav and mel path in config.py, then run
```
python preprocess_esc50.py
```
which will  convert all audio to mel.

### Training your own model
Edit config.py accordingly, or choose one from ./configs/ directory. Then start training,
``` 
./train_multifold.sh
```

Log files are found in './log', use visualdl to visualize: 
```
visualdl --logdir='./log'
```

 



### Results



The following are the current leader board from <a href="https://github.com/karolpiczak/ESC-50">ESC-50</a> dataset 


| <sub>Title</sub> | <sub>Notes</sub> | <sub>Accuracy</sub> | <sub>Paper</sub> | <sub>Code</sub> |
| :--- | :--- | :--- | :--- | :--- |
| <sub>**A Sequential Self Teaching Approach for Improving Generalization in Sound Event Recognition**</sub> | <sub>Multi-stage sequential learning with knowledge transfer from Audioset</sub> | <sub>94.10%</sub> | <sub>[kumar2020](https://arxiv.org/pdf/2007.00144.pdf)</sub> |  |
| <sub>**Urban Sound Tagging using Multi-Channel Audio Feature with Convolutional Neural Networks**</sub> | <sub>Pretrained model with multi-channel features</sub> | <sub>89.50%</sub> | <sub>[kim2020](http://dcase.community/documents/challenge2020/technical_reports/DCASE2020_JHKim_21_t5.pdf)</sub> | <a href="https://github.com/JaehunKim-DeepLearning/Dcase2020_Task5">:scroll:</a> |
| <sub>**An Ensemble of Convolutional Neural Networks for Audio Classification**</sub> | <sub>CNN ensemble with data augmentation</sub> | <sub>88.65%</sub> | <sub>[nanni2020](https://arxiv.org/pdf/2007.07966.pdf)</sub> | <a href="https://github.com/LorisNanni/Ensemble-of-Convolutional-Neural-Networks-for-Bioimage-Classification">:scroll:</a> |
| <sub>**Unsupervised Filterbank Learning Using Convolutional Restricted Boltzmann Machine for Environmental Sound Classification**</sub> | <sub>CNN with filterbanks learned using convolutional RBM + fusion with GTSC and mel energies</sub> | <sub>86.50%</sub> | <sub>[sailor2017](https://pdfs.semanticscholar.org/f6fd/1be38a2d764d900b11b382a379efe88b3ed6.pdf)</sub> |  |
| <sub>**AclNet: efficient end-to-end audio classification CNN**</sub> | <sub>CNN with mixup and data augmentation</sub> | <sub>85.65%</sub> | <sub>[huang2018](https://arxiv.org/pdf/1811.06669.pdf)</sub> |  |
| <sub>**On Open-Set Classification with L3-Net Embeddings for Machine Listening Applications**</sub> | <sub>x-vector network with openll3 embeddings</sub> | <sub>85.00%</sub> | <sub>[wilkinghoff2020](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2020/pdfs/0000800.pdf)</sub> |  |
| <sub>**Learning from Between-class Examples for Deep Sound Recognition**</sub> | <sub>EnvNet-v2 ([tokozume2017a](http://www.mi.t.u-tokyo.ac.jp/assets/publication/LEARNING_ENVIRONMENTAL_SOUNDS_WITH_END-TO-END_CONVOLUTIONAL_NEURAL_NETWORK-poster.pdf)) + data augmentation + Between-Class learning</sub> | <sub>84.90%</sub> | <sub>[tokozume2017b](https://openreview.net/forum?id=B1Gi6LeRZ)</sub> |  |
| <sub>**Novel Phase Encoded Mel Filterbank Energies for Environmental Sound Classification**</sub> | <sub>CNN working with phase encoded mel filterbank energies (PEFBEs), fusion with Mel energies</sub> | <sub>84.15%</sub> | <sub>[tak2017](https://www.researchgate.net/profile/Dharmesh_Agrawal/publication/320733074_Novel_Phase_Encoded_Mel_Filterbank_Energies_for_Environmental_Sound_Classification/links/5a084c780f7e9b68229c8947/Novel-Phase-Encoded-Mel-Filterbank-Energies-for-Environmental-Sound-Classification.pdf)</sub> |  |
| <sub>**Knowledge Transfer from Weakly Labeled Audio using Convolutional Neural Network for Sound Events and Scenes**</sub> | <sub>CNN pretrained on AudioSet</sub> | <sub>83.50%</sub> | <sub>[kumar2017](https://arxiv.org/pdf/1711.01369.pdf)</sub> | <a href="https://github.com/anuragkr90/weak_feature_extractor">:scroll:</a> |
| <sub>**Unsupervised Filterbank Learning Using Convolutional Restricted Boltzmann Machine for Environmental Sound Classification**</sub> | <sub>CNN with filterbanks learned using convolutional RBM + fusion with GTSC</sub> | <sub>83.00%</sub> | <sub>[sailor2017](https://pdfs.semanticscholar.org/f6fd/1be38a2d764d900b11b382a379efe88b3ed6.pdf)</sub> |  |
| <sub>**Deep Multimodal Clustering for Unsupervised Audiovisual Learning**</sub> | <sub>CNN + unsupervised audio-visual learning</sub> | <sub>82.60%</sub> | <sub>[hu2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_Deep_Multimodal_Clustering_for_Unsupervised_Audiovisual_Learning_CVPR_2019_paper.pdf)</sub> |  |
| <sub>**Novel TEO-based Gammatone Features for Environmental Sound Classification**</sub> | <sub>Fusion of GTSC & TEO-GTSC with CNN</sub> | <sub>81.95%</sub> | <sub>[agrawal2017](http://www.eurasip.org/Proceedings/Eusipco/Eusipco2017/papers/1570347591.pdf)</sub> |  |
| <sub>**Learning from Between-class Examples for Deep Sound Recognition**</sub> | <sub>EnvNet-v2 ([tokozume2017a](http://www.mi.t.u-tokyo.ac.jp/assets/publication/LEARNING_ENVIRONMENTAL_SOUNDS_WITH_END-TO-END_CONVOLUTIONAL_NEURAL_NETWORK-poster.pdf)) + Between-Class learning</sub> | <sub>81.80%</sub> | <sub>[tokozume2017b](https://openreview.net/forum?id=B1Gi6LeRZ)</sub> |  |
| :headphones: <sub>***Human accuracy***</sub> | <sub>Crowdsourcing experiment in classifying ESC-50 by human listeners</sub> | <sub>81.30%</sub> | <sub>[piczak2015a](http://karol.piczak.com/papers/Piczak2015-ESC-Dataset.pdf)</sub> | <a href="https://github.com/karoldvl/paper-2015-esc-dataset">:scroll:</a> |
| <sub>**Objects that Sound**</sub> | <sub>*Look, Listen and Learn* (L3) network ([arandjelovic2017a](https://arxiv.org/pdf/1705.08168.pdf)) with stride 2, larger batches and learning rate schedule</sub> | <sub>79.80%</sub> | <sub>[arandjelovic2017b](https://arxiv.org/pdf/1712.06651.pdf)</sub> |  |
| <sub>**Look, Listen and Learn**</sub> | <sub>8-layer convolutional subnetwork pretrained on an audio-visual correspondence task</sub> | <sub>79.30%</sub> | <sub>[arandjelovic2017a](https://arxiv.org/pdf/1705.08168.pdf)</sub> |  |


## Reference
[1] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." arXiv preprint arXiv:1912.10211 (2019).

[2] Urban Sound Tagging using Multi-Channel Audio Feature with Convolutional Neural Networks










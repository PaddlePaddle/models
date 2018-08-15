# Introduction to models

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://github.com/PaddlePaddle/models)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://github.com/PaddlePaddle/models)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

PaddlePaddle provides a rich set of computational units to enable users to adopt a modular approach to solving various learning problems. In this repo, we demonstrate how to use PaddlePaddle to solve common machine learning tasks, providing several different neural network model that anyone can easily learn and use.

## 1. Word Embedding

The word embedding expresses words with a real vector. Each dimension of the vector represents some of the latent grammatical or semantic features of the text and is one of the most successful concepts in the field of natural language processing. The generalized word vector can also be applied to discrete features. The study of word vector is usually an unsupervised learning. Therefore, it is possible to take full advantage of massive unmarked data to capture the relationship between features and to solve the problem of sparse features, missing tag data, and data noise. However, in the common word vector learning method, the last layer of the model often encounters a large-scale classification problem, which is the bottleneck of computing performance.

In the example of word vectors, we show how to use Hierarchical-Sigmoid and Noise Contrastive Estimation (NCE) to accelerate word-vector learning.

- 1.1 [Hsigmoid Accelerated Word Vector Training](https://github.com/PaddlePaddle/models/tree/develop/hsigmoid)
- 1.2 [Noise Contrastive Estimation Accelerated Word Vector Training](https://github.com/PaddlePaddle/models/tree/develop/nce_cost)


## 2. RNN language model

The language model is important in the field of natural language processing. In addition to getting the word vector (a by-product of language model training), it can also help us to generate text. Given a number of words, the language model can help us predict the next most likely word. In the example of using the language model to generate text, we focus on the recurrent neural network language model. We can use the instructions in the document quickly adapt to their training corpus, complete automatic writing poetry, automatic writing prose and other interesting models.

- 2.1 [Generate text using the RNN language model](https://github.com/PaddlePaddle/models/tree/develop/generate_sequence_by_rnn_lm)

## 3. Click-Through Rate prediction
The click-through rate model predicts the probability that a user will click on an ad. This is widely used for advertising technology. Logistic Regression has a good learning performance for large-scale sparse features in the early stages of the development of click-through rate prediction. In recent years, DNN model because of its strong learning ability to gradually take the banner rate of the task of the banner.

In the example of click-through rate estimates, we first give the Google's Wide & Deep model. This model combines the advantages of DNN and the applicable logistic regression model for DNN and large-scale sparse features. Then we provide the deep factorization machine for click-through rate prediction. The deep factorization machine combines the factorization machine and deep neural networks to model both low order and high order interactions of input features.

- 3.1 [Click-Through Rate Model](https://github.com/PaddlePaddle/models/tree/develop/ctr)
- 3.2 [Deep Factorization Machine for Click-Through Rate prediction](https://github.com/PaddlePaddle/models/tree/develop/deep_fm)

## 4. Text classification

Text classification is one of the most basic tasks in natural language processing. The deep learning method can eliminate the complex feature engineering, and use the original text as input to optimize the classification accuracy.

For text classification, we provide a non-sequential text classification model based on DNN and CNN. (For LSTM-based model, please refer to PaddleBook [Sentiment Analysis](http://www.paddlepaddle.org/docs/develop/book/06.understand_sentiment/index.html)).

- 4.1 [Sentiment analysis based on DNN / CNN](https://github.com/PaddlePaddle/models/tree/develop/text_classification)
- 4.2 [Text classification model based on Nested sequence](https://github.com/PaddlePaddle/models/tree/develop/nested_sequence/text_classification)

## 5. Learning to rank

Learning to rank (LTR) is one of the core problems in information retrieval and search engine research. Training data is used by a learning algorithm to produce a ranking model which computes the relevance of documents for actual queries.
The depth neural network can be used to model the fractional function to form various LTR models based on depth learning.

The algorithms for learning to rank are usually categorized into three groups by their input representation and the loss function. These are pointwise, pairwise and listwise approaches. Here we demonstrate RankLoss loss function method (pairwise approach), and LambdaRank loss function method (listwise approach). (For Pointwise approaches, please refer to [Recommended System](http://www.paddlepaddle.org/docs/develop/book/05.recommender_system/index.html)).

- 5.1 [Learning to rank based on Pairwise and Listwise approches](https://github.com/PaddlePaddle/models/tree/develop/ltr)

## 6. Semantic model
The deep structured semantic model uses the DNN model to learn the vector representation of the low latitude in a continuous semantic space, finally models the semantic similarity between the two sentences.

In this example, we demonstrate how to use PaddlePaddle to implement a generic deep structured semantic model to model the semantic similarity between two strings. The model supports different network structures such as CNN (Convolutional Network), FC (Fully Connected Network), RNN (Recurrent Neural Network), and different loss functions such as classification, regression, and sequencing.

- 6.1 [Deep structured semantic model](https://github.com/PaddlePaddle/models/tree/develop/dssm)

## 7. Sequence tagging

Given the input sequence, the sequence tagging model is one of the most basic tasks in the natural language processing by assigning a category tag to each element in the sequence. Recurrent neural network models with Conditional Random Field (CRF) are commonly used for sequence tagging tasks.

In the example of the sequence tagging, we describe how to train an end-to-end sequence tagging model with the Named Entity Recognition (NER) task as an example.

- 7.1 [Name Entity Recognition](https://github.com/PaddlePaddle/models/tree/develop/sequence_tagging_for_ner)

## 8. Sequence to sequence learning

Sequence-to-sequence model has a wide range of applications. This includes machine translation, dialogue system, and parse tree generation.

As an example for sequence-to-sequence learning, we take the machine translation task. We demonstrate the sequence-to-sequence mapping model without attention mechanism, which is the basis for all sequence-to-sequence learning models. We will use scheduled sampling to improve the problem of error accumulation in the RNN model, and machine translation with external memory mechanism.

- 8.1 [Basic Sequence-to-sequence model](https://github.com/PaddlePaddle/models/tree/develop/nmt_without_attention)
- 8.2 [Improve translation quality using Scheduled Sampling](https://github.com/PaddlePaddle/models/tree/develop/scheduled_sampling)
- 8.3 [Neural machine translation with external memory mechanism](https://github.com/PaddlePaddle/models/tree/develop/mt_with_external_memory)
- 8.4 [Generate chinese poetry](https://github.com/PaddlePaddle/models/tree/develop/generate_chinese_poetry)

## 9. Reading comprehension

When deep learning and various new technologies continue to push forward the field of natural language processing, we cannot help but ask: How should we confirm that the model truly understands human-specific natural language and has a certain ability to understand and reason? Looking at various classic issues in the field of NLP: lexical analysis, syntactic analysis, emotional classification, writing poetry, etc. From the technical principle, the classic solutions of these problems still have a certain distance from the “language understanding”. In order to measure the gap between the existing NLP technology and the ultimate goal of “language comprehension,” we need a task that is difficult enough, quantifiable, and reproducible. This is also the original intention of reading comprehension. Although the current research status indicates that the models that perform well on the current reading comprehension dataset still do not achieve true language comprehension, machine-reading comprehension is still regarded as an important task for the test model to understand the language.

Reading comprehension is essentially a kind of question answering. The model answers the given question after reading a paragraph of text, in this task, we introduce the use of the Learning to Search method, which translates reading comprehension into a multi-step decision process, which looking for the sentence where the answer lies from the paragraph, the starting and ending position of the answer in the sentence.

- 9.1 [Globally Normalized Reader](https://github.com/PaddlePaddle/models/tree/develop/globally_normalized_reader)

## 10 Question Answering

The Question Answering system uses computer to automatically answer the questions raised by users. It is one of the important tasks to verify whether the machine has natural language understanding ability. Its research history can be traced back to the origin of artificial intelligence. Compared with the retrieval system, the question answering system is an advanced form of the information service. The system returns to the user no longer the sorted keyword-based retrieval results, but an accurate natural language answer.

In an automated question answering task, we demonstrate an end-to-end question-answering system based on deep learning, which translates automated question answering into a sequence annotation problem. The end-to-end question answering system attempts to build a joint learning model by learning from high-quality "question-evidence-answer" data, and at the same time learns the semantic mapping relationship between corpora, knowledge bases, and semantic representations of question sentences. The system transform traditional question semantic analysis, text retrieval, answer extraction and generation into a learnable process

- 10.1 [A Factual Auto Answers Model Based on Sequence Labeling](https://github.com/PaddlePaddle/models/tree/develop/neural_qa)


## 11. Image classification

Compared with text, images can provide more vivid, easy to understand and more artistic information, which is an important source of people's transfer and exchange of information. Image classification is to distinguish different types of images based on the semantic information of the image. It is an important basic problem in computer vision and is also the basis of other high-level visual tasks such as image detection, image segmentation, object tracking, and behavior analysis. It is widely used in many fields. Applications. For example, face recognition and intelligent video analysis in the field of security, traffic scene recognition in the traffic field, content-based image retrieval and album auto-categorization in the Internet field, and image recognition in the medical field.

For the example of image classification, we show you how to train AlexNet, VGG, GoogLeNet, ResNet, Inception-v4, Inception-Resnet-V2 and Xception models in PaddlePaddle. It also provides model conversion tools that convert Caffe or TensorFlow trained model files into PaddlePaddle model files.

- 11.1 [convert Caffe model file to PaddlePaddle model file](https://github.com/PaddlePaddle/models/tree/develop/image_classification/caffe2paddle)
- 11.2 [convert TensorFlow model file to PaddlePaddle model file](https://github.com/PaddlePaddle/models/tree/develop/image_classification/tf2paddle)
- 11.3 [AlexNet](https://github.com/PaddlePaddle/models/tree/develop/image_classification)
- 11.4 [VGG](https://github.com/PaddlePaddle/models/tree/develop/image_classification)
- 11.5 [Residual Network](https://github.com/PaddlePaddle/models/tree/develop/image_classification)
- 11.6 [Inception-v4](https://github.com/PaddlePaddle/models/tree/develop/image_classification)
- 11.7 [Inception-Resnet-V2](https://github.com/PaddlePaddle/models/tree/develop/image_classification)
- 11.8 [Xception](https://github.com/PaddlePaddle/models/tree/develop/image_classification)

## 12. Target Detection

The goal of the target detection task is to give an image or video frame, let the computer find the location of all the targets, and give each target a specific category. Target detection is a very simple task for humans. However, computers can only “see” a matrix with values between 0 and 255. It is difficult to solve high-level semantic concepts such as humans or objects in images or video frames, and it is even more difficult to locate which area in the image the target appears in. At the same time, because the target will appear in any position in the image or video frame, the shape of the target is ever-changing, and the background of the image or video frame is very different. Many factors make the target detection to be a challenging problem for the computer.

In the target detection task, we perform to use SSD method to complete the target detection. SSD(Single Shot MultiBox Detector) is one of the newer and better-performing detection algorithms in the field of target detection. It has the features of high detection speed and detection accuracy.

- 12.1 [Single Shot MultiBox Detector](https://github.com/PaddlePaddle/models/tree/develop/ssd/README.cn.md)

## 13. Scene Text Recognition

Many scene images contain rich text information, which plays an important role in understanding image information and can greatly help people to cognize and understand the content of scene images. Scene text recognition is a process of converting image information into a text sequence with complex image background, low resolution, various fonts, and random distribution. It can be considered as a special translation process: translating image input into natural language output. The development of scene image text recognition technology has also promoted the emergence of new applications such as helping Street View applications to obtain more accurate address information by automatically recognizing words in street signs.

In the scene text recognition task, we describe how to combine CNN-based image feature extraction and RNN-based sequence translation techniques to eliminate artificially defined features, avoid character segmentation, and use automatically learned image features to achieve end-to-end unconstrainedness Character positioning and recognition.

- 13.1 [Scene Text Recognition](https://github.com/PaddlePaddle/models/tree/develop/scene_text_recognition)

## 14. Speech Recognize

Auto Speech Recognize(ASR) translates vocabulary content in human speech into computer-readable input, allowing the machine to “understand” human speech and play an important role in applications such as voice assistant, voice input, and voice interaction. Deep learning has achieved remarkable achievements in the field of speech recognition. The end-to-end deep learning method integrates traditional acoustic models, dictionaries, language models and other modules into a whole. It no longer depends on various conditional independence in hidden Markov models, and the model becomes more concise. a neural network model takes speech features as input and directly outputs the recognized text, which has become the most important means of speech recognition.

In the speech recognition task, we provide a complete pipeline based on the DeepSpeech2 model, including: feature extraction, data enhancement, model training, language model, decoding module, etc. At the same time, we provide a trained model and experience example. Everyone can use their own Voice to experience the fun of speech recognition.

14.1 [Speech Recognize: DeepSpeech2](https://github.com/PaddlePaddle/DeepSpeech)


This tutorial is contributed by [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) and licensed under the [Apache-2.0 license](LICENSE).

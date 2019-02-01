# Multi-view Simnet for Personalized recommendation

## Introduction
In personalized recommendation scenario, a user often is provided with several items from personalized interest matching model. In real world application, a user may have multiple views of features, say user-id, age, click-history of items, search queries. A item, e.g. news, may also have multiple views of features like news title, news category, images in news and so on. Multi-view Simnet is matching a model that combine users' and items' multiple views of features into one unified model. The model can be used in many industrial product like Baidu's feed news. The model is adapted from the paper A Multi-View Deep Learning(MV-DNN) Approach for Cross Domain User Modeling in Recommendation Systems, WWW 2015. The difference between our model and the MV-DNN is that we also consider multiple feature views of users.

## Dataset
Currently, synthetic dataset is provided for proof of concept and we aim to add more real world dataset in this project in the future.

## Model
This project aims to provide practical usage of Paddle in personalized matching scenario. The model provides several encoder modules for different views of features. Currently, Bag-of-Embedding encoder, Temporal-Convolutional encoder, Gated-Recurrent-Unit encoder are provided. We will add more practical encoder for sparse features commonly used in recommender systems. Training algorithms used in this model is pairwise ranking in that a negative item with multiple views will be sampled given a pair of positive user-item pair.

## Train
The command line options for training can be listed by `python train.py -h`
```bash
python train.py
```

## Infer
The command line options for inference can be listed by `python infer.py -h`
```bash
python infer.py
```

## Future work
- Multiple types of pairwise loss will be added in this project. For different views of features between a user and an item, multiple losses will be supported. The model will be verified in real world dataset.
- Parallel Executor will be added in this project
- Distributed Training will be added

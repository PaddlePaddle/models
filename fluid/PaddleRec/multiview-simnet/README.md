# Multi-view Simnet for Personalized recommendation

## Introduction
In personalized recommendation scenario, a user often is provided with serveral items from personalized interest matching model. In real world application, a user may have multiple views of features, say userid, age, click-history of items. A item, e.g. news, may also have multiple views of features like news title, news category and so on. Multi-view Simnet is matching a model that combine users' and items' multiple views of features into one unified model. The model can be used in many industrial product like baidu's feed news.

## Dataset
Currently, synthetic dataset is provided for proof of concept and we aim to add more real world dataset in this project in the future.

## Model
This project aims to provide practical usage of Paddle in personalized matching scenario. The model provides serveral encoder modules for different views of features. Currenly, Bag-of-Embedding encoder, Temporal-Convolutional encoder, Gated-Recurrent-Unit encoder are provided. We will add more practical encoder for sparse features commonly used in recommender systems. Training algorithms used in this model is pairwise ranking in that a negative item with multiple views will be sampled given a pair of positive user-item pair.

## Train
The command line options for training can be listed by `python train.py -h`
```bash
python train.py 
```

## Infer
The command line options for inference can be listed by `python infer.py -h`

## Future work
### Multiple types of pairwise loss will be added in this project. For different views of features between a user and an item, multiple losses will be supported. The model will be verified in real world dataset.
### infer will be added
### Parallel Executor will be added in this project
### Distributed Training will be added


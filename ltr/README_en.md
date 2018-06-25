Running the program sample in this directory requires the version of the PaddlePaddle is v0.10.0. If the version is below this requirement, following the [instructions](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html) in the document about installation to update your Paddlepaddle's version.

# Learning To Rank

Learning to rank[1] is a method to build the ranking model of machine learning,which plays an important role in the computer science scene such as information retrieval, natural language processing and data mining. The primary purpose of learning to rank is to order a document that reflects the relevance of any query request for a given set of documents. In this example, using the annotated Corpus training two classical ranking models RankNet[4] and LamdaRank[6],the corresponding ranking model can be generated, and the correlation documents can be sorted by any query request.

## Background Information
Learning to rank is the application of machine learning. On the one hand, the manual ranking rules can not deal with the large scale of the candidate data, on the other hand can not give the appropriate weight for the candidate data of different channels, so it is widely used in daily life.Learning to rank originated in the field of information retrieval and is still the core parts of many information retrieval systems,such as the ranking of search results in search engine,ranking of candidate data in the recommendation system,and online advertising, and so on. In this case, we use the document retrieval task to illustrate the learning to rank model.

![image](https://github.com/PaddlePaddle/models/blob/develop/ltr/images/search_engine_example.png?raw=true)

Figure.1 the role of ranking model in the typical application search engine of document retrieval.

Assuming that there is a set of documents $S$, the document retrieval task is based on the relevance of the requests to give the order of the documents. According to the query request, the query engine will score every document according to the query request, and arrange the documents in reverse order according to the grading, and get the query results.Given a query and corresponding documents, the model is trained based the scoring of the document sorts. When it goes to the predicted phase, the model will generate the document sort according to the query received. The common ranking learning methods are divided into the following three categories.

- Pointwise approach

In this case,the learning-to-rank problem can be viewed as a regression problem.The input single sample is the **score-document**,the correlation score of each query-Document pair is used as the real number or the sequence number,so the individual query-document pairs are uesd as a sample point (the origin of the word pointwise) to train the ranking model.When predicting,the correlation score of query-document pair is given for the specified input.
- Pairwise approach

In this case, the learning-to-rank problem is approximated by a classification problem — learning a binary classifier that can tell which document is better in a given pair of documents.The single input sample is the **label-document pair**.For multiple result documents of one query,any two documents are combined to form document pairs as input samples.Any two documents are combined to form document pairs as the input samples.That is to learn a two classifier, the input is a pair of documents A-B (the origin of Pairwise), according to whether the correlation of A is better than B,the two classifier gives the classification label 1 or 0.After classifying all the document pairs,we can get a set of partial order relations to construct the order relation of the documents.The principle of this kind of the method is to reduce the number of the reverse order document pairs in the order of the given pair of documents $S$,so as to achieve the goal of optimizing the sorting result.
- Listwise approach

These algorithms try to directly optimize the value of one of the above evaluation measures, averaged over all queries in the training data.The single input sample is a **document arranged**. By constructing the appropriate measurement function to measure the difference between the current document ranking and the optimal ranking,then optimizes the evaluation measures to get the ranking model. It is difficult to optimize because most of the ranking loss function are not continuous smooth functions.
![image](https://github.com/PaddlePaddle/models/blob/develop/ltr/images/learning_to_rank.jpg?raw=true)

Figure.2 Three methods of the ranking model

## Experimental data

The experimental data in this example uses the LETOR corpus of benchmarking data in the Ranking learning, part of the query results is from the Gov2 website, which contains about 1700 query request result document lists and has made manual annotations on the relevance of the documents.Among them,a query contains a unique query id,corresponding to a number of related documents,forming a query request result list.The feature vector of Each document is represented by the one-dimensional array,and corresponds to a correlation score between the human annotation and the query.

This example automatically downloads the LETOR MQ2007 dataset and cache when it is first running,without manual downloading.

The Data sets of **mq2007** provide a generation format for three types of the ranking models respectively.Which is need to specify the **format**.

for example,the call interface

```
pairwise_train_dataset = functools.partial(paddle.dataset.mq2007.train, format="pairwise")
for label, left_doc, right_doc in pairwise_train_dataset():
    ...
```
## Model overview

For the ranking model, the RankNet model of the Pairwise method and the LambdaRank of the Listwise method are provided in this example, respectively representing two types of the learning methods. The ranking model of the Pointwise method can be degraded to the regression problem. Please refer to the  [recommendation system](https://github.com/PaddlePaddle/book/blob/develop/05.recommender_system/README.cn.md) in the PaddleBook.

## RankNet model

[RankNet](http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf) is a classic Pairwise ranking learning method, which is a typical forward neural network ranking model. The $i$ document in the document collection $S$ is denoted as $U_i$, its document feature vector is denoted as $x_i$, and for a given document pair $U_i$, $U_j$, RankNet maps the input single document feature vector $x$ to $f(x)$, and gets $s_i=f(x_i),$s_j=f(x_j)$.The probability that  the correlation of $U_i$ is better than $U_j$ is recorded as $P_{i,j}$.

$$P_{i,j}=P(U_{i}>U_{j})=\frac{1}{1+e^{-\sigma (s_{i}-s_{j}))}}$$

Because most of the rank metric functions are mostly non-continuous and non-smooth, the ranknet needs a metric function $C$ that can be optimized.First,the cross entropy is used as a measure function to measure the prediction cost, and the loss function $C$ is recorded as

$$C_{i,j}=-\bar{P_{i,j}}logP_{i,j}-(1-\bar{P_{i,j}})log(1-P_{i,j})$$

The $\bar{P_{i,j}}$ represents the true probability, which is recorded as

$$\bar{P_{i,j}}=\frac{1}{2}(1+S_{i,j})$$

$S_{i,j}$ = {+1,0}, which represents the label of pair consisting of $U_i$ and $U_j$,that is, whethe the Ui correlation is better than $U_j$.

Finally, a derivable metric loss function is obtained

$$C=\frac{1}{2}(1-S_{i,j})\sigma (s_{i}-s{j})+log(1+e^{-\sigma (s_{i }-s_{j})})$$

It can be optimized using conventional gradient descent methods. See [RankNet](http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf) for details.

Meanwhile, get the gradient information of document $U_i$ in the ranking optimization process.

$$\lambda {i,j}=\frac{\partial C}{\partial s{i}} = \frac{1}{2}(1-S_{i,j})-\frac{1} {1+e^{\sigma (s_{i}-s_{j})}}$$

The meaning of the expression is the increase or decrease of the document $U_i$ during this round of sorting optimization.

Based on the above inference, the RankNet network structure is constructed, which is composed of several layers of hidden layers and full connected layers. As shown in the figure, the document features are used in the hidden layers, and the all connected layer is transformed by layer by layer,completing the transformation from the underlying feature space to the high-level feature space. The structure of docA and docB is symmetrical and they are input into the final RankCost layer.

![image](https://github.com/sunshine-2015/models/blob/patch-4/ltr/images/ranknet_en.png?raw=true)

Figure.3 The structure diagram of RankNet network

- Full connected layer: means that each node in the previous layer is connected to the underlying network. In this example, **paddle.layer.fc** is also used. Note that the full connection layer dimension input to the RankCost layer is 1.

- RankCost layer: The RankCost layer is the core of the RankNet ranking network and measures whether the docA correlation is better than the docB. Give the predicted value and compare it with the label. Cross entropy is used as a measure of the loss function, using a gradient descent method for optimization. Details can be found in [RankNet](http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf) [4].

Because the network structure in Pairwise is Left-right symmetrical, half of the network structure can be defined, and the other half share network parameters. The PaddlePaddle allows sharing of connections in the network structure, parameters with the same name will share parameters. Use the PaddlePaddle to implement the RankNet ranking model. The sample code for defining the network structure is given in the **half_ranknet** function in [ranknet.py](https://github.com/PaddlePaddle/models/blob/develop/ltr/ranknet.py).

The structure defined in the ***half_ranknet*** function uses the same model structure as in FIG 3: two hidden layers, a fully connected layer with **hidden_size=10** and a fully connected layer with **hidden_size=1**. In this example, **input_dim** refers to the dimension of the characteristic of the input **single document**. The value of label is 1,0. Each input sample is the structure of **<label>, <docA, docB>**. Take docA as an example, input the **input_dim** document features, turn into 10-dimensional, 1-dimensional features, and finally input into the RankCost layer, compare docA and docB. The RankCost output gives the predicted value.

### RankNet model training

Train **RankNet** model executes on the command line:

```
python train.py --model_type ranknet
```

The initial execution automatically downloads data and trains the RankNet model,which stores the parameters of the model at each round.

### RankNet model prediction
Use the trained **RankNet** model to continue the prediction and execute it on the command line:

```
python infer.py --model_type ranknet --test_model_path models/ranknet_params_0.tar.gz
```

This example provides training and prediction part of the RankNet model. After completing the training,The model is divided into two parts, topology structure (the **rank_cost** is not part of the model topology) and the model's parameter file. In this example, the topology **half_ranknet** during the **ranknet** training is reused, and the parameters of the model are loaded from the external memory. The input of the prediction of the model is the feature vector of a single document,and the model will give a relevance score. Sorting the forecast scores to get the final document relevance ranking result.

## User-defined RankNet data

The above code uses PaddlePaddle's built-in sorting data. If you want to use custom format data, you can refer to the PaddlePaddle's built-in **mq2007** data set and write a new generator function. For example, the input data is in the following format, containing only three documents doc0-doc2.

<query_id> <relevance_score> <feature_vector>(featureid: feature_value)

```
query_id : 1, relevance_score:1, feature_vector 0:0.1, 1:0.2, 2:0.4  #doc0
query_id : 1, relevance_score:2, feature_vector 0:0.3, 1:0.1, 2:0.4  #doc1
query_id : 1, relevance_score:0, feature_vector 0:0.2, 1:0.4, 2:0.1  #doc2
query_id : 2, relevance_score:0, feature_vector 0:0.1, 1:0.4, 2:0.1  #doc0
.....
```


The input sample needs to be converted to Pairwise's input format. for example, the combination of the generated format is same with the structure of the mq2007 Pairwise format.

<label> <docA_feature_vector><docB_feature_vector>

```
1 doc1 doc0
1 doc1 doc2
1 doc0 doc2
....
```


Note that generally, in Pairwise format data, label=1 indicates that the correlation between docA and the query is better than that of docB. In fact, the label information is implicit in the combination of docA and docB. If there is **0 docA docB**, then exchange order to construct **1 docB docA**.

In addition, combining all pairs will make training data redundancy because the total order relationship on the document set can be recovered from the partial partial order relationships. See the  [PairWise approach](http://www.machinelearning.org/proceedings/icml2007/papers/139.pdf) [5] for related research. This example will not repeat.

```
# a customized data generator
def gen_pairwise_data(text_line_of_data):
    """
      return :
      ------
      label : np.array, shape=(1)
      docA_feature_vector : np.array, shape=(1, feature_dimension)
      docA_feature_vector : np.array, shape=(1, feature_dimension)
    """
    return label, docA_feature_vector, docB_feature_vector
```


Corresponding to the paddle input, **integer_value** is a single integer, and **dense_vector** is a real one-dimensional vector, corresponding to the generator, it is necessary to specify the input data correspondence before training the model.


```
# Define the input data order
feeding = { "label":0,
            "left/data" :1,
            "right/data":2}
```

## LambdaRank ranking model
[LambdaRank](https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf)[6] is Listwise's ranking method. It is developed by Bugers et al.[6] from RankNet. It uses the method of constructing lambda function (the origin of LambdaRank name) to optimize the metric NDCG (Normalized Discounted Cumulative Gain). The resulting list is used individually as a training sample. The NDCG is one of the standards in the information theory that measures the ranking quality of the document list. The NDCG scores of the previous $K$ documents are recorded as

$$NDCG@K=Z_{k}\sum (2^{rel_{i}})1/log(k+1)$$

As previously deduced by RankNet, document sorting requires the gradient information of errors from sorting. The NDCG metric function is non-smooth and non-successive. It cannot directly obtain the gradient information. Therefore, the |delta(NDCG)|=|NDCG(new) - NDCG(old)| is introduced to construct the lambda function as

$$\lambda {i,j}=\frac{\partial C}{\partial s{i}}=-\frac{\sigma }{1+e^{\sigma (s_{i}-s{j})}}|\Delta NDCG|$$

Replace the gradient representation in RankNet and get the ranking model called [LambdaRank](https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf)

From the above derivation we can see that the LambdaRank network structure is very similar to the RankNet structure. as the picture shows

![image](https://github.com/sunshine-2015/models/blob/patch-4/ltr/images/LambdaRank_EN.png?raw=true)

Figure 4. Network structure of LambdaRank

Replacing the pair of the document-score pair with the list of the query-related document as input sample, refactoring the LambdaCost layer to RankCost layer, and keep the rest of the network same with Ranket.

- LambdaCost layer: The LambdaCost layer uses the NDCG difference as the Lambda function. The score is a one-dimensional sequence. For a monotonic training sample, the full-connection layer output is a 1x1 sequence, and the length of the both sequence is equal to the number of documents obtained by the query. The **LambdaRank** function's details  is in [LambdaRank](https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf)


An example of a LambdaRank network structure defined using PaddlePaddle is the **lambda_rank** function in [lambda_rank.py](https://github.com/PaddlePaddle/models/blob/develop/ltr/lambda_rank.py).

The same model structure as FIG. 3 is used in the above structure. Similar to RankNet, two fully connected layers of **hidden_size=10** and **hidden_size=1** are used respectively. The input_dim in this example refers to the dimension of the characteristic of the input single document. Each input sample is the structure of label, <docA, docB>. Take docA as an example, with inputing the input_dim's document features,which are turned into 10-dimensional, 1-dimensional features, and finally is inputed into the LambdaCost layer. It should be noted that the label and data formats here are **dense_vector_sequences**, which represent a sequence of document scores or document features.

### LambdaRank model training

Execute on the command line to train the **LambdaRank** model:

```
python train.py --model_type lambdarank
```

The first run of the script will automatically download the data and train the LambdaRank model and store the model of each round.

### LambdaRank model prediction
The prediction process of the LambdaRank model is the same as RankNet. The model's topology in the prediction model reuses the model definition in the code and loads the corresponding parameter file from the external memory. The input during the forecast is a document list, and the output is the relevance score of each document in the document list. The document is re-sorted based the score, to obtaine the final document's sorting result.

Use the trained LambdaRank model to continue the prediction:

```
python infer.py --model_type lambdarank --test_model_path models/lambda_rank_params_0.tar.gz
```

## Customize LambdaRank data
The above code uses the built-in mq2007 data from PaddlePaddle, and if you want to use custom format data, you can refer to the built-in mq2007 dataset in PaddlePaddle and write a generator function. For example, the input data is in the following format, which only has three documents doc0-doc2.

<query_id> <relevance_score> <feature_vector>

```
query_id : 1, relevance_score:1, feature_vector 0:0.1, 1:0.2, 2:0.4  #doc0
query_id : 1, relevance_score:2, feature_vector 0:0.3, 1:0.1, 2:0.4  #doc1
query_id : 1, relevance_score:0, feature_vector 0:0.2, 1:0.4, 2:0.1  #doc2
query_id : 2, relevance_score:0, feature_vector 0:0.1, 1:0.4, 2:0.1  #doc0
query_id : 2, relevance_score:2, feature_vector 0:0.1, 1:0.4, 2:0.1  #doc1
.....
```

Convert the format to the Listwise, for example:

<query_id><relevance_score> <feature_vector>

```
1    1    0.1,0.2,0.4
1    2    0.3,0.1,0.4
1    0    0.2,0.4,0.1

2    0    0.1,0.4,0.1
2    2    0.1,0.4,0.1
......
```
**Note Data format**
- The number of documents corresponding to each sample in the data must be more than the NDCG_num of **lambda_cost** layer.
- If the document of the single sample is 0, the correlation of the document is 0, and the calculation of NDCG is invalid, then we can determine that the query is invalid, and we can filter out such query during training.


```
# self define data generator
def gen_listwise_data(text_all_lines_of_data):
    """
    return :
    ------
    label : np.array, shape=(samples_num, )
    querylist : np.array, shape=(samples_num, feature_dimension)
    """
    return label_list, query_docs_feature_vector_matrix
```

Corresponds to input of PaddlePaddle, the type of **label** as **dense_vector_sequence**, is the sequence of score, the type of **data** for **dense_vector_sequence**, is the input feature vector sequences, **input_dim** is a one-dimensional feature vector dimension of a single document, need to specify the corresponding relations between the input data before training model.


```
# Define the input data order
feeding = {"label":0,
           "data" : 1}
```


## Output custom evaluation index during training.
Here, we take **RankNet** as an example of how to export custom evaluation metrics during training. This method can also be used to obtain the value of an output matrix of the network in the training process.

The RankNet network learns a scoring function to score the inputs of left and right.The greater the difference between the scores of the two inputs,the stronger the discrimination ability of the scoring function for positive and negative , and the better the model's generalization ability. If we want to get the average value of the difference between the right and left inputs in the training process. To compute this custom metric, we need to get the output matrix for each mini-batch after the split layer (which corresponds to the layer in ranknet with the name of left_score and right_score). We can do this with the following two steps:

1. In the event_handler, processing the predefined paddle.Event.EndIteration or the paddle.Event.EndPass events in PaddlePaddle.
2. Call the event.gm.getlayeroutput, pass the name of the specified layer into the network, we can obtain the value of the layer after completing the forward calculation of mini-batch.

Here is the code example:

```
def score_diff(right_score, left_score):
    return np.average(np.abs(right_score - left_score))

def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 25 == 0:
            diff = score_diff(
                event.gm.getLayerOutputs("right_score")["right_score"][
                    "value"],
                event.gm.getLayerOutputs("left_score")["left_score"][
                    "value"])
            logger.info(("Pass %d Batch %d : Cost %.6f, "
                         "average absolute diff scores: %.6f") %
                        (event.pass_id, event.batch_id, event.cost, diff))
```


## Conclusion
LTR is widely used in real life. The construction method of ranking model can generally be divided into PointWise, Pairwise, Listwise. this example adopt the LETOR mq2007 data as an example, expounds the classic method of RankNet in Pairwise and the LambdaRank in Listwise method, shows how to use PaddlePaddle framework to structure the corresponding sorting model, and provides a sample used the custom data types. Paddlepaddles provides a flexible programming interface. At the same time, with using a set of code in a single GPU, The LTR type is implemented by the multi-machine's distributed multi-gpu.
## Attention
1. As a demonstration example of LTR, this example is a small network size. In the application, it is necessary to adjust the network complexity in combination with the actual situation and reset the network scale.
2. In this case, the feature vectors in the experimental data are the joint features of the query-document. When using the independent features of the query-document, [DSSM](https://github.com/PaddlePaddle/models/tree/develop/dssm) can be used to build the network.
## Reference
1. https://en.wikipedia.org/wiki/Learning_to_rank
2. Liu T Y. [Learning to rank for information retrieval](http://ftp.nowpublishers.com/article/DownloadSummary/INR-016)[J]. Foundations and Trends® in Information Retrieval, 2009, 3(3): 225-331.
3. Li H. [Learning to rank for information retrieval and natural language processing](http://www.morganclaypool.com/doi/abs/10.2200/S00607ED2V01Y201410HLT026)[J]. Synthesis Lectures on Human Language Technologies, 2014, 7(3): 1-121.
4. Burges C, Shaked T, Renshaw E, et al. [Learning to rank using gradient descent](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2005_BurgesSRLDHH05.pdf)[C]//Proceedings of the 22nd international conference on Machine learning. ACM, 2005: 89-96.
5. Cao Z, Qin T, Liu T Y, et al. [Learning to rank: from pairwise approach to listwise approach](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2007_CaoQLTL07.pdf)[C]//Proceedings of the 24th international conference on Machine learning. ACM, 2007: 129-136.
6. Burges C J C, Ragno R, Le Q V. [Learning to rank with nonsmooth cost functions](https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf)[C]//NIPS. 2006, 6: 193-200.

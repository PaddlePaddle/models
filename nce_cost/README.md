Running the following program need use PaddlePaddle v0.10.0 version。If your PaddlePaddle installed lower demand，Please following the [install  document ](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html) instruction upgrade PaddlePaddle install version。

---

# Using the NCE(noise contrastive estimation)to accelerate the language model training

##  Why we need NCE

The language model is the basis of many natural language processing tasks, and it is also an effective method for obtaining word vector representations.Neural Probabilistic Language Model(NPLM）Describes the probability of the sequence of words $\omega_1,...,\omega_T$ belonging to a fixed language:
$$P(\omega_1^T)= \prod_{t=1}^{T}P(\omega_t|\omega_1^{t-1})$$

In order to reduce the difficulty of modeling and solving, we often introduce certain conditional independent assumptions: the probability of the word $w_t$ is only affected by the previous $n-1$ words, so there are:

$$ P(\omega_1^T) \approx \prod P(\omega_t|\omega_{t-n-1}^{t-1}) \tag{1}$$

From the formula ($1$), we can using modeling condition probability $P(\omega_t|w_{t-n-1},...,\omega_{t-1})$ and calculate the entire sequence the probability of  $\omega_1,...,\omega_T$ 。So, we can simply summarize the task of language model solving:

**The vector of the sequence of given words represents $h$, called the context, and the model predicts the probability of the next target word $\omega$.**

In[$n$-gram language models](https://github.com/PaddlePaddle/book/tree/develop/04.word2vec)，The context takes a fixed $n-1$ words，[RNN language models](https://github.com/PaddlePaddle/models/tree/develop/generate_sequence_by_rnn_lm)Can handle any length of context.

Given the context $h$, NPLM learns a scoring function $s_\theta(\omega, h)$, $s$ which characterizes the context $h$ vector and all possible next-word vectors representing $ The similarity between \omega'$ is then normalized by dividing the value of the scoring function $s$ in the whole word table space (divided by the normalization factor $Z$) to get the target word $\omega$ The probability distribution, where: $\theta$ is a learnable parameter, this process is expressed by the formula ($2$), which is the calculation of the `Softmax` function.

$$P_\theta^h(\omega) = \frac{\text{exp}{s_\theta(\omega, h)}}{Z}，Z=\sum_{\omega'} \exp{s_\theta(\omega', h)}\tag{2}$$

However, the normalization factor $Z$ must be calculated whether the estimated probability $P_\theta^h(\omega)$ or the gradient of the likelihood is calculated. The calculation of $Z$ increases linearly with the size of the dictionary. When training large-scale language models, for example, when the dictionary grows to a million or more, the training time will become very long. Therefore, we need other possibilities. Learning guidelines, his solution process should be more light and solvable. **

MLE (Maximum Likelihood Estimation) is the most common learning criterion for solving the probability ($2$). However, the normalization factor $Z$ must be calculated whether the estimated probability $P_\theta^h(\omega)$ or the gradient of the likelihood is calculated. The calculation of $Z$ increases linearly with the size of the dictionary. When training large-scale language models, for example, when the dictionary grows to a million or more, the training time will become very long. Therefore, we need other possibilities. Learning guidelines, his solution process should be more light and solvable. **

Another of part the models introduces the use of [Hsigmoid Accelerated Word Vector Training] (https://github.com/PaddlePaddle/models/tree/develop/hsigmoid). Here we introduce another sample-based training model to speed up language training. method：Using NCE（Noise-contrastive estimation, NCE）\[[1](#reference document)\]。

## What is NCE

NCE is a probability density estimation criterion based on the sampling idea and is used for estimation/fitting: the probability function is composed of a non-normalized score function and a normalization factor. Such a special probability function [[1](#References)\]. NCE avoids the calculation of the normalization factor $Z$ in the full dictionary space by constructing an auxiliary problem such as the following, which reduces the computational cost:

Given a context $h$ and any known noise distribution $P_n$, learn a second-class classifier to fit: target $\mega$ from real distribution $P_\theta$ ($D = 1$) or noise distribution The probability of $P_n$ ($D = 0$). Assuming that the number of negative class samples from the noise distribution is $k$ times the target sample, then there are:

$$P(D=1|h,\omega) = \frac{P_\theta(h, \omega)}{P_\theta (h, \omega) + kP_n} \tag{3}$$

We directly use the `Sigmoid` function to characterize a binary class ($3$)：

$$P(D=1|h,\omega) = \sigma (\Delta s_\theta(w,h)) \tag{4}$$

With the above problem set, maximum likelihood estimation can be performed based on the two classifications: increasing the probability of positive samples and reducing the probability of negative samples [[2,3] (#references)], that is, minimizing the following Loss function：

$$
J^h(\theta )=E_{ P_d^h }\left[ \log { P^h(D=1|w,\theta ) }  \right] +kE_{ P_n }\left[ \log P^h (D=0|w,\theta ) \right]$$
$$
 \\\\\qquad =E_{ P_d^h }\left[ \log { \sigma (\Delta s_\theta(w,h)) }  \right] +kE_{ P_n }\left[ \log (1-\sigma (\Delta s_\theta(w,h)))  \right] \tag{5}$$


The formula ($5$) is the NCE loss function defined based on noise comparison estimation. At this point, we have two remaining questions:
1. What is $s_\theta(w,h)$ in ($5$)?
    - In a neural network implementation, $s_\theta(h,\omega)$ is an unnormalized score.
    - The learnable parameter $W$ of the NCE cost layer is a matrix of $|V| \times d$ dimensions, $|V|$ is the dictionary size, and $d$ is the dimension of the context vector $h$;
    - The actual category of the next word $t$ during training is a positive class. Samples of $k$ negative samples from the specified noise distribution are classified as follows: $\{n_1, ..., n_k\}$;
    - Extract the $\{t, n_1, ..., n_k\}$ lines in $W$ (total $k + 1$ lines) and $h$ respectively Calculate the points $s_\theta(w,h)$ The final loss is calculated by ($5$).
2. How to choose the noise distribution?
    - In practice, the noise distribution can be arbitrarily chosen (the noise distribution implies a certain a priori).
    - The most common choices are: Use the `unigram` distribution based on the entire dictionary (word frequency statistics), unbiased uniform distribution.
    - If the user does not specify a noise distribution in the PaddlePaddle, the default is to use a uniform distribution.

When using NCE to accurately train, the calculation cost of the last layer is only linearly related to the number of negative samples. When the number of negative samples gradually increases, the NCE estimation criterion converges to the maximum likelihood estimation. Therefore, when training using the NCE criterion, the quality of the normalized probability distribution can be controlled by controlling the number of negative samples.

## Experimental data
This example uses a Penn Treebank (PTB) data set ([Tomas Mikolov pre-processing version] (http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)) to train a 5-gram language model. PaddlePaddle provides the [paddle.dataset.imikolov](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/dataset/imikolov.py) interface to conveniently use PTB data. When no downloaded data is found, the script automatically downloads and verifies the integrity of the file. The corpus language is English, with a total of 42068 sentence training data and 3761 sentence test data.

## Network Structure
The detailed network structure of the 5-gram neural probabilistic language model is shown in Figure 1:

<p align="center">
<img src="images/network_conf.png" width = "70%" align="center"/><br/>
Figure1. 5-gram Network configuration structure
</p>

The model is mainly divided into the following parts:

1. **Input Layer**: The input sample consists of the original English words. Each English word is first converted to the id in the dictionary.

2. **word vector layer**: id means that the word vector representation that is expressed continuously by the word vector layer function can better reflect the semantic relationship between words. After the training is completed, the semantic similarity between words can be expressed using the distance between the word vectors. The more similar the semantics, the closer the distance.

3. **Word vector splice layer**: Concatenate the word vectors and connect the word vectors end to end to form a long vector. This can facilitate the processing of the fully connected layer behind.

4. **FULL CONNECTION Hidden Layer**: Input the long vector obtained from the previous layer into a hidden layer of neural network and output the feature vector. A fully connected hidden layer can enhance the learning ability of the network.

5. **NCE layer**: `The paddle.layer.nce` provided by PaddlePaddle can be used directly as a loss function during training.

## Training
Run the command ``` python train.py ``` in the command line window to start the training task.

- The first time the program runs, it will detect if the ptb data set is included in the user's cache folder. If it is not included, it will be downloaded automatically.
- Every 10 batch prints the value of model training on the training set during the run
- After each pass is completed, the loss on the test data set is calculated and the latest snapshot of the model is also saved.

The NCE call code in the model file `network_conf.py` is as follows:

```python
return paddle.layer.nce(
            input=hidden_layer,
            label=next_word,
            num_classes=dict_size,
            param_attr=paddle.attr.Param(name="nce_w"),
            bias_attr=paddle.attr.Param(name="nce_b"),
            num_neg_samples=25,
            neg_distribution=None)
```

Some important parameters of the NCE layer are explained as follows:

| Parameter name                    | Parameter function                                 | Introduction                                                                                                                                               |
| :----------------------- | :--------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| param\_attr / bias\_attr | use to set parameter name  |  facilitates the loading of parameters during the forecast phase, as described in the prediction section                                                                                                   |
| num\_neg\_samples        |  Number of Negative Samples                  |  You can control the proportion of positive and negative samples. The range of this value is [1, dictionary size -1]. The more negative samples, the slower the training of the whole model. The accuracy of the model will also be higher                                  |
| neg\_distribution        |  Generates the distribution of negative sample tags. The default is a uniform distribution| You can control the sampling weights for each category when negative samples are sampled. For example, if you want the positive example to be “sunny,” and the negative example “flood” is highlighted more during training, you can increase the weight of the “flood” sample |



## Prediction
1. Runing in command line :
    ```bash
    python infer.py \
      --model_path "models/XX" \
      --batch_size 1 \
      --use_gpu false \
      --trainer_count 1
    ```
    The parameters are as follows：
    - `model_path`：Specify the path where the trained model is located. required.
    - `batch_size`：The number of samples in parallel is predicted at one time. Optional, the default value is `1`.
     - `use_gpu`：Whether use GPU for prediction。Optional，the default value is `False`。
    - `trainer_count` : Number of thread for prediction。Optional，the default is  `1`。**Note: The number of threads used for prediction must be greater than the number of samples to be predicted in parallel**。


2. It should be noted that: **The calculation logic for prediction and training is different**. Predictions using full-join matrix multiplication followed by `softmax` activation, output based on the probability distribution for each category, need to replace the `paddle.train.nce` layer used in training. In PaddlePaddle, the NCE layer stores the learnable parameters as a matrix of `[number of classes × output vector width of the previous layer]`. When predicting, **full-join operation needs to be transferred when loading the NCE layer to learn the parameters**, the code is as follows:
    ```python
    return paddle.layer.mixed(
          size=dict_size,
          input=paddle.layer.trans_full_matrix_projection(
              hidden_layer, param_attr=paddle.attr.Param(name="nce_w")),
          act=paddle.activation.Softmax(),
          bias_attr=paddle.attr.Param(name="nce_b"))
    ```
    The `paddle.layer.mixed` in the above code snippet must be entered as `paddle.layer.×_projection` in PaddlePaddle. `paddle.layer.mixed` sums up multiple `projection` (input can be multiple) calculations as output. `paddle.layer.trans_full_matrix_projection` Transposes the parameter $W$ when calculating matrix multiplication.

3. The forecasted output format is as follows:
     ```text
     0.6734  their   may want to move
     ```

     Each row is a forecast result, internally separated by "\t", for a total of 3 columns:
     - First column: Probability of the next word.
     - Second column: The next word of the model prediction.
     - Third column: $n$ words entered, separated by spaces.


## Reference
1. Gutmann M, Hyvärinen A. [Noise-contrastive estimation: A new estimation principle for unnormalized statistical models](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)[C]//Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics. 2010: 297-304.

1. Mnih A, Kavukcuoglu K. [Learning word embeddings efficiently with noise-contrastive estimation](https://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf)[C]//Advances in neural information processing systems. 2013: 2265-2273.

1. Mnih A, Teh Y W. [A Fast and Simple Algorithm for Training Neural Probabilistic Language Models](http://xueshu.baidu.com/s?wd=paperuri%3A%280735b97df93976efb333ac8c266a1eb2%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Farxiv.org%2Fabs%2F1206.6426&ie=utf-8&sc_us=5770715420073315630)[J]. Computer Science, 2012:1751-1758.

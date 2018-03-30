The minimum PaddlePaddle version needed for the code sample in this directory is v0.11.0. If you are on a version of PaddlePaddle earlier than v0.11.0, [please update your installation](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html).

---

# Text Classification

The following is a description of the files contained in this example:

```text
.
├── images              # Picture in this document
│   ├── cnn_net.png
│   └── dnn_net.png
├── infer.py            # Script for prediction
├── network_conf.py     # The various network structures involved in this example are defined in this file, and if you further modify the model structure, look at this file
├── reader.py           # The interface used to read data, if you use custom format data, look at this file
├── README.md           # The document
├── run.sh              # Running script for training task, if you run the script directly, start the training task with the default parameters
├── train.py            # training script
└── utils.py            # Define common functions. Such as printing logs, parsing command line parameters, building dictionaries, loading dictionaries, and so on
```

## Introduction
Text classification is an important basic work in the filed of Natural Language Processing.The task is to assign a text to one or more classes or categories, The [Emotional Classification](https://github.com/PaddlePaddle/book/blob/develop/06.understand_sentiment/README.md) in [PaddleBook](https://github.com/PaddlePaddle/book) is a typical text classification task. The process is as follows:
1. Collect user comment data from the movie review site.
2. Clean and annotate data.
3. Design the model.
4. Evaluate learning effect of the model.

The trained classifier can **automatically** predict whether the emotion of new user reviews are positive or negative, and play an important role in public opinion monitoring, marketing planning, product brand value assessment and other tasks. The above process is also the routine process that we need to follow for a new text classification task. It can be seen that the great advantage of the deep learning algorithm is that **there is no need to extract complex features and only need to clean and annotate the original text**.

The [Emotional Classification](https://github.com/PaddlePaddle/book/blob/develop/06.understand_sentiment/README.md) in [PaddleBook](https://github.com/PaddlePaddle/book) introduces a complex bidirectional LSTM model. RNN has obvious advantages in some complicated tasks that need to understand language and semantics, but it has a large amount of computation and usually has higher requirements for parameter adjustment techniques. Other models are also considered in a task that has a certain limit on the time of calculation. In addition to time, it is more important that **model selection is the basis for the success of machine learning tasks**. The goal of a machine learning task is always to improve the generalization ability, that is, the ability to predict the unknown new sample：

1. The simple model can't fit the training samples accurately, and can't accurately predict the unknown samples that have not appeared in the training set. This is the **underfitting**.
2. However, too complex models can easily memorizing every sample in training samples, but it has no recognition ability for unknown samples that do not appear in training set. This is the **overfitting**.

"No Free Lunch (NFL)" is one of the basic principles of a machine learning task: No model is superior to others inherently. The design and selection of the model is based on understanding the characteristics of different models, but it is also a process of multiple experimental evaluation. In this case, we continue to introduce some of the most commonly used text classification models. Their ability and complexity are different, which can help you compare learning differences between these models, and choose different models in different scenarios.


## Model Description

The following model is included in the `network_conf.py`：

1. `fc_net`： DNN model，which is a non-sequence model and uses a basic fully connected structure.
2. `convolution_net`：Shallow CNN model，which is a basic sequence model that can handle the variable long sequence input and extract the features within a local region.

We take the task of emotional classification as an example to explain the difference between the sequence model and the non sequence model. Emotional classification is a common text classification task, and the model automatically determines whether the emotion is positive or negative. For example, "not bad" in sentence "The apple is not bad" is the key to determining the emotion of the sentence.

- For the DNN model, we only know that there is a "not" and a "bad" in the sentence. The order relation between them is lost in the input network, and the network no longer has the chance to learn the sequence information between sequences.
- The CNN model accepts text sequences as input and preserves the sequence information between "not bad".

The characteristics of the two models are summarized as follows:

1. The computation complexity of the DNN model can be far lower than the CNN/RNN model, and has the advantage in the tasks with limited time.
2. DNN is often characterized by frequent words, which can be influenced by participle error. But it is still an effective model for some tasks that rely on keyword features, such as spam message detection.
3. In most cases that require some semantic understanding of text classification tasks, the sequence models represented by CNN/RNN are often better than DNN models, such as eliminating ambiguity in context by context.

### 1. DNN model

**DNN model structure:**

<p align="center">
<img src="images/dnn_net_en.png" width = "90%" align="center"/><br/>
Figure 1. DNN text classification model in this example
</p>

The code to implement the DNN structure in PaddlePaddle is seen the `fc_net` function in `network_conf.py`. The model is divided into the following parts：

- **Word vector layer**：In order to better express the semantic relationship between different words, the words are first transformed into a vector of fixed dimensions. After the completion of the training, the similarity between words and words can be expressed by the distance between their vectors. The more similar semantics, the closer distance is. For more information on the word vector, refer to the [Word2Vec](https://github.com/PaddlePaddle/book/tree/develop/04.word2vec) in PaddleBook.

- **Max-pooling layer**：The max-pooling is carried out on the time series, and eliminates the difference in the number of words in different corpus samples, and extracts the maximum value of each position in the word vector. After being pooled, the vector sequence of the word vector layer is transformed into a vector of a fixed dimension. For example, it is assumed that the vector before the max-pooling is `[[2,3,5],[7,3,6],[1,4,0]]`，Then the max-pooling result is `[7,4,6]`。

- **Full connected layer**：After the max-pooling, the vector is sent into two continuous hidden layers, and the hidden layers are full connected.

- **Output layer**：The number of neurons in the output layer is in accordance with the number of the sample classes. For example, in the two classification problem, there are 2 neurons in the output layer. Through the Softmax activation function, the output result is a normalized probability distribution, and the sum is 1. Therefore, the output of the $i$ neuron can be considered as the prediction probability of the sample belonging to class $i$.

The default DNN model is two classification (`class_dim=2`), and the embedding (word vector) dimension is 28 (`emd_dim=28`), and two hidden layers use Tanh activation function (`act=paddle.activation.Tanh()`). It is important to note that the input data of the model is an integer sequence, not the original word sequence. In fact, in order to deal with convenience, we usually id the words in the order of word frequency to convert the words into the serial number in the dictionary.

### 2. CNN model

**CNN model structure：**

<p align="center">
<img src="images/cnn_net_en.png" width = "90%" align="center"/><br/>
Figure 2. CNN text classification model in this example
</p>

The code to implement the CNN structure in PaddlePaddle is seen the `convolution_net` function in `network_conf.py`. The model is divided into the following parts：

- **Word vector layer**：The word vector layer in CNN is the same as in DNN. It transforms words into fixed dimension vectors, and uses the distance between vectors to express the semantic similarity between words. As shown in Figure 2, the word vector is defined as a line vector, and then a matrix is formed by the splicing of all the word vectors in the sentence. If the vector dimension of the word is 5, the sentence "The cat sat on the read mat
" contains 7 words, then the matrix dimension is 7*5. For more information on the word vector, refer to the [Word2Vec](https://github.com/PaddlePaddle/book/tree/develop/04.word2vec) in PaddleBook.

- **Convolution layer**： The convolution in the text classification is carried out on the time series, that is, the width of the convolution kernel is consistent with the matrix of the word vector layer, and the convolution is carried out along the height direction of the matrix. The results obtained after convolution are called "feature map". Assuming that the height of the convolution kernel is $h$, the height of the matrix is $N$ and the convolution step is 1, then the feature map is a vector with a height of $N+1-h$. The convolution kernels with multiple different height can be used at the same time, and multiple feature maps are obtained.

- **Max-pooling layer**: The max-pooling operation is carried out for each feature map obtained by convolution. Because the feature map is already a vector, so the max-pooling is actually simply selecting the largest elements in each vector, and the largest elements are spliced together to form a new vector. Obviously, the dimension of the vector is equal to the number of the feature map, that is, the number of the convolution kernel. For example, suppose we use four different convolution kernels, and the convolution generated feature maps are: `[2,3,5]`、`[8,2,1]`、`[5,7,7,6]` and `[4,5,1,8]`. Because the height of the convolution kernel is different, the size of the feature maps is different. The max-pooling is carried out on the four feature maps, and the results are as follows: `[5]`、`[8]`、`[7]` and `[8]`. Finally, the pooling results are spliced together to get `[5,8,7,8]`。

- **Full connected and output layer**：The max-pooling results are output through the full connected layer. As with the DNN model, the number of neurons in the final output layer is the same as the number of sample classes, and the sum of the output is 1.

The input data of the CNN are consistent with the DNN. The pooling text sequence convolution module has been encapsulated in PaddlePaddle: `paddle.networks.sequence_conv_pool`，which can be invoked directly. The `context_len` parameter of the module is used to specify the length of the text covered by the convolution kernel, that is, the height of the convolution kernel in Figure 2. `hidden_size` is used to specify the number of convolution kernels of this type. This code uses 128 convolution cores with 3 sizes and 128 convolution cores with a size of 4. After max-pooling of the convolution results, a 256 dimensional vector is generated, and the vector passes through a full connected layer to output the final prediction result.

## Running with built-in data of Paddlepaddle

### How to train

Executing the `sh run.sh` command in the terminal will directly run the example with the built-in sentiment classification dataset built by PaddlePaddle: `paddle.dataset.imdb`, and we will see the following input:

```text
Pass 0, Batch 0, Cost 0.696031, {'__auc_evaluator_0__': 0.47360000014305115, 'classification_error_evaluator': 0.5}
Pass 0, Batch 100, Cost 0.544438, {'__auc_evaluator_0__': 0.839249312877655, 'classification_error_evaluator': 0.30000001192092896}
Pass 0, Batch 200, Cost 0.406581, {'__auc_evaluator_0__': 0.9030032753944397, 'classification_error_evaluator': 0.2199999988079071}
Test at Pass 0, {'__auc_evaluator_0__': 0.9289745092391968, 'classification_error_evaluator': 0.14927999675273895}
```
The log is output once every 100 batch, and the output information includes: (1) the Pass sequence number; (2) the Batch sequence number; (3) The results of the current Batch evaluation index. The evaluation index is specified when configuring the network topology, and in the above output, the AUC of the training sample set and the error rate index are output.

### How to predict

After training, the model is stored in the current working directory by default. Execute `python infer.py` in the terminal, and the prediction script will load the trained model for prediction.

- The default use `paddle.data.imdb.train` to train a Pass to produce a DNN model and test the `paddle.dataset.imdb.test`

You will see the following output：

```text
positive        0.9275 0.0725   previous reviewer <unk> <unk> gave a much better <unk> of the films plot details than i could what i recall mostly is that it was just so beautiful in every sense emotionally visually <unk> just <unk> br if you like movies that are wonderful to look at and also have emotional content to which that beauty is relevant i think you will be glad to have seen this extraordinary and unusual work of <unk> br on a scale of 1 to 10 id give it about an <unk> the only reason i shy away from 9 is that it is a mood piece if you are in the mood for a really artistic very romantic film then its a 10 i definitely think its a mustsee but none of us can be in that mood all the time so overall <unk>
negative        0.0300 0.9700   i love scifi and am willing to put up with a lot scifi <unk> are usually <unk> <unk> and <unk> i tried to like this i really did but it is to good tv scifi as <unk> 5 is to star trek the original silly <unk> cheap cardboard sets stilted dialogues cg that doesnt match the background and painfully onedimensional characters cannot be overcome with a scifi setting im sure there are those of you out there who think <unk> 5 is good scifi tv its not its clichéd and <unk> while us viewers might like emotion and character development scifi is a genre that does not take itself seriously <unk> star trek it may treat important issues yet not as a serious philosophy its really difficult to care about the characters here as they are not simply <unk> just missing a <unk> of life their actions and reactions are wooden and predictable often painful to watch the makers of earth know its rubbish as they have to always say gene <unk> earth otherwise people would not continue watching <unk> <unk> must be turning in their <unk> as this dull cheap poorly edited watching it without <unk> breaks really brings this home <unk> <unk> of a show <unk> into space spoiler so kill off a main character and then bring him back as another actor <unk> <unk> all over again
```

Each row of output log is the result of prediction for a sample. It is divided into 3 columns by `\t`, which are: (1) Category labels for prediction; (2) the probability that samples belong to each class, separated by spaces, and (3) input text.

## Using custom data training and prediction

### How to train

1. The structure of data

    Suppose there are training data in the following format: Each line is a sample separated by `\t`, the first column is a class label, and the second column is the content of the input text, and the words are separated by space. The following are two sample data：

    ```
    positive        PaddlePaddle is good
    negative        What a terrible weather
    ```

2. Writing the data reading interface

    A custom data reading interface only needs to write a Python generator to implement the logic of parsing a training sample from the original input text. The following code is implemented to read the original data, and the return type is: `paddle.data_type.integer_value_sequence`(the number of words in the dictionary) and `paddle.data_type.integer_value`(class label), and these 2 inputs give the function of the 2 `data_layer` defined in the network.
    ```python
    def train_reader(data_dir, word_dict, label_dict):
        def reader():
            UNK_ID = word_dict["<UNK>"]
            word_col = 0
            lbl_col = 1

            for file_name in os.listdir(data_dir):
                with open(os.path.join(data_dir, file_name), "r") as f:
                    for line in f:
                        line_split = line.strip().split("\t")
                        word_ids = [
                            word_dict.get(w, UNK_ID)
                            for w in line_split[word_col].split()
                        ]
                        yield word_ids, label_dict[line_split[lbl_col]]

        return reader
    ```

    - For the type of input data accepted by `data_layer` in PaddlePaddle and the return format for the data reading interface, refer to [input-types](http://www.paddlepaddle.org/release_doc/0.9.0/doc_cn/ui/data_provider/pydataprovider2.html#input-types)。
    - The above code is detailed in the `reader.py` script in this directory, and `reader.py` provides all the code to read the test data.

    Next, we only need to pass the data read function `train_reader` as a parameter to the `paddle.batch` interface in the `train.py` script, and then we can read data using the custom data interface. The way to invoke is as follows:

    ```python
    train_reader = paddle.batch(
            paddle.reader.shuffle(
                reader.train_reader(train_data_dir, word_dict, lbl_dict),
                buf_size=1000),
            batch_size=batch_size)
    ```

3. Modifying the command line parameters

    - If data is processed into the same format of sample data, it is necessary to modify `train.py` boot parameters and specify `train_data_dir` parameters in `run.sh` script. Then you can run this example directly without modifying the data read interface `reader.py`.
    - The execution `python train.py --help` can get a detailed description of the startup parameters of the `train.py` script. The main parameters are as follows:
        - `nn_type`：Choosing the model to use, there are two types of support currently: "DNN" or "CNN".
        - `train_data_dir`：Specify the folder where the training data is located, and you must specify this parameter by using custom data training. Otherwise, `paddle.dataset.imdb` training is used, and `test_data_dir`, `word_dict`, and `label_dict` parameters are ignored.
        - `test_data_dir`：Specify the folder where the test data is located, and if it does not specify it will not be tested.
        - `word_dict`：The path of the dictionary file, if not specified, will automatically establish a dictionary from the training data according to the word frequency statistics.
        - `label_dict`：A category label dictionary, which is used to map a class label of a string type to an integer type sequence number.
        - `batch_size`：Specify how many samples are used each time the forward calculation and the backpropagation are used.
        - `num_passes`：How many rounds of training.

### How to predict

1. Modify the following variables in the `infer.py`, specify the model used and the test data.
    ```python
    model_path = "dnn_params_pass_00000.tar.gz"  # The path of the model
    nn_type = "dnn"      # Specify the model used for the test
    test_dir = "./data/test"      # Specify the directory where the test file is located
    word_dict = "./data/dict/word_dict.txt"     # Specify the path of the dictionary
    label_dict = "./data/dict/label_dict.txt"    # The path of the category label dictionary
    ```
2. Execute `python infer.py` in the terminal.

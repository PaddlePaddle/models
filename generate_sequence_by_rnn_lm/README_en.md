The program example running in this directory needs to use the PaddlePaddle v0.10.0 version.If your PaddlePaddle installation version is below this requirement, update the PaddlePaddle installation version according to the instructions in the [installation document](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html).


#Text Generation with RNN

Language Model is a probability distribution model,in a nutshell,it's a model for calculating the probability of a sentence.Taking advantage of it you can determine which word sequence is more likely,or given a number of words,predicting the next most likely word.

###Application scenarios
Language models are used in many fields,for instance:
* **Automatic writing**:The language model can generate the next word base on previous content,generate the entire sentence, paragraph, chapter through the recursive method.
* **QA**:The language model can generate Answer according to Question.
* **Machine translation**:The current mainstream machine translation models are mostly based on Encoder-Decoder mode,decoder is a conditional language model used to generate the target language.
* **Spell check**:The language model can calculate the probability of the sequence. Generally, the probability of the sequence at the misspelling will decrease sharply. It can be used to identify spelling errors and provide correction candidate sets.
* **Speech tagging, syntactic analysis, speech recognition......**

###About this example
This example implements a language model based on RNN and generates text with the language model. The directory structure of this example is as follows:

```
.
├── data
│   └── train_data_examples.txt        # Sample data, refer to sample data format, provide your own data
├── config.py    # Configuration files, including data, train, and infer related configurations
├── generate.py  # Predictive task script, which generates text.
├── beam_search.py    # beam search Algorithm implementation
├── network_conf.py   # The various network structures involved in this example are defined in this file, if you want to further modify the model structure. Please modify this file.
├── reader.py    # Read data interface
├── README.md
├── train.py    # Training task script
└── utils.py    # Define common functions, such as: build a dictionary, load a dictionary, etc.
```

###RNN language model
####Introduction
RNN is a sequence model,the basic idea: at time $t$,put the hidden layer output of the previous time $t-1$ and the word vector of the time $t$ together into the hidden layer to get the characteristic representation of the time $t$,then use this feature to represent the forecast output at time $t$,so recursively on the time dimension.
![](https://github.com/PaddlePaddle/models/blob/develop/generate_sequence_by_rnn_lm/images/rnn.png)

####Model implementation
The implementation of the RNN language model in this example is as follows:
* **Define model parameters** :```config.py```defines parameter variables for the model.
* **Define the model structure** :```rnn_lm``` function in ```network_conf.py```defines the structure of the model,as follow:
 + Input layer : Map the input word (or word) sequence into a vector, the word vector layer : ```embedding```.
 + Middle layer : According to the configuration of the RNN layer, the ```embedding``` vector sequence obtained in the previous step is taken as input.
 + Output layer : Use ```softmax``` to normalize the probability of calculating words.
 + loss : Multiple classes of cross entropy are defined as the loss function of the model.

* **Training model** : Function ```main``` in ```train.py``` implements the training of the model and the implementation process is as follows:
 + Prepare input data : Create and save dictionaries and build readers for train and test data.
 + Initialize the model : Including the structure and parameters of the model
 + Building the trainer : The demo uses the Adam optimization algorithm
 + Define the callback function : Build ```event_handler``` to track changes in loss during training and save model parameters at the end of each round of training.
 + training : Train the model using the trainer.

* **Generate text ** : ```Generate.py``` implements the generation of the text. The implementation flow is as follows:
 + Load trained models and dictionary files
 + Read ```gen_file file```, each line is a sentence prefix, using the [column search algorithm (Beam Search)](https://github.com/PaddlePaddle/book/blob/develop/08.machine_translation/README.cn.md#%E6%9F%B1%E6%90%9C%E7%B4%A2%E7%AE%97%E6%B3%95) to generate text based on the prefix.
 + Save the generated text and its prefix to the file ```gen_result```

####Instructions for use
Here's how to run this example:
* 1, Run the ```python train.py``` command to start the train model (using LSTM by default) until training is over.
* 2, Run ```python generate.py``` to run the text generation. (The input text defaults to ```data/train_data_examples.txt```, and the generated text is saved to ```data/gen_result.txt by default```.)

**If you need to use your own corpus and custom model, you need to modify the configuration in config.py. Details and adaptations are as follows :**
####Language adaptation
* **Clean corpora** : Remove whitespace, tabs, garbled text, remove numbers, punctuation marks, special symbols, etc. as needed.
* **Content format** : Each sentence occupies one line; the words in each line are separated by a space character.
* **Configure the following parameters in ```config.py``` as needed :**
``train_file = "data/train_data_examples.txt"
test_file = ""
vocab_file = "data/word_vocab.txt"
model_save_dir = "models"
``
1. ```rain_file```: Specify the path to training data, **Pre-segmentation is required**
2. ```test_file```: Specify the test data path. If the training data is not empty, the specified test data will be tested at the end of each ```pass``` training.
3. ```vocab_file```: Specify the path of the dictionary. If the dictionary file does not exist, word frequency statistics will be made on the training corpus and a dictionary will be
4. ```model_save_dir```: Specifies the path where the model is saved. If the specified folder does not exist, it will be automatically created.

####Strategies for building a dictionary

* When the specified dictionary file does not exist, word frequency statistics will be performed on the training data. The automatic construction of the dictionary config.py has the following two parameters related to the construction of the dictionary:
``max_word_num = 51200 - 2
cutoff_word_fre = 0``
 1. ```max_word_num```: Specifies how many words are in the dictionary.
 2. ```cutoff_word_fre```: The lowest frequency of words in the dictionary in the training corpus.

* Join specified ```max_word_num = 5000```, and ```cutoff_word_fre = 10```, Word frequency statistics found that there were only 3,000 words in the training corpora that had a frequency higher than 10, and that eventually 3,000 words would constitute a dictionary.
* When you build a dictionary, two special symbols are automatically added:
 1. ```<unk>```: Words that do not appear in the dictionary
 2. ```<e>```: Sentence terminator

Note: It should be noted that the larger the dictionary is, the richer the content is, but the longer the training takes.After the general Chinese word corpus in different words can have tens or even hundreds of thousands, if ```max_word_num``` value is too small then lead to ```<unk>``` proportion is too high, if ```max_word_num``` larger value, then seriously affect the speed of training (for accuracy Also has influence).Therefore, there are also ways to train models by "words",scilicet,Considering each Chinese character as a word, the number of commonly used Chinese characters is also a few thousand, making the size of the dictionary not too large and not losing too much information, but the semantics of the same word in Chinese differs greatly in different words, sometimes leading to models. The effect is not ideal.It is recommended to try more and choose "word training" or "word training" according to the actual situation.

####Model adaptation, training
* Adjust the ```config.py``` configuration as needed to modify the network results of the rnn language model:
``rnn_type = "lstm"  # "gru" or "lstm"
emb_dim = 256
hidden_size = 256
stacked_rnn_num = 2``
1. ```rnn_type``` : Support "gru" or "lstm" two parameters, choose which RNN unit to use.
2. ```emb_dim``` : Set the dimension of the word vector.
3. ```hidden_size``` : Set the RNN cell hidden layer size.
4. ```stacked_rnn_num``` : Set the number of stacked RNN units to form a deeper model.
* Run the ```python train.py``` command to train the model. The model will be saved to the directory specified by ```model_save_dir```.

####Generate text on demand
* Adjust the following variables in config.py as needed, as follows:
``gen_file = "data/train_data_examples.txt"
gen_result = "data/gen_result.txt"
max_gen_len = 25  # the max number of words to generate
beam_size = 5
model_path = "models/rnn_lm_pass_00000.tar.gz" ``
1. ```gen_file``` : Specify the input data file, each line is a sentence prefix, **need to be pre-segmented**.
2. ```gen_result``` : Specify the output file path, and the result will be written to this file.
3. ```max_gen_len``` : Specify the maximum length of each sentence generated. If the model cannot generate ```<e>```, the generation process will automatically terminate when ```max_gen_len``` words are generated.
4. ```beam_size``` : The width of each step of the Beam Search algorithm.
5. ```model_path``` : Specify the path to the trained model.
  Among them, the gen_file holds the text prefix to be generated, and each prefix is in one line. The format is as follows:
  ```若隐若现 地像 幽灵 , 像 死神```
Write the text prefix to be generated into the file in this format;
* Run the ```python generate.py``` command to run the beam search algorithm to generate the text for the input prefix. Here are the results generated by the model :
```81    若隐若现 地像 幽灵 , 像 死神
-12.2542    一样 。 他 是 个 怪物 <e>
-12.6889    一样 。 他 是 个 英雄 <e>
-13.9877    一样 。 他 是 我 的 敌人 <e>
-14.2741    一样 。 他 是 我 的 <e>
-14.6250    一样 。 他 是 我 的 朋友 <e>```
among them:
 1. The first line 81 is looming like a ghost, like death, separated by \t. There are two columns:
  + The first column is the sequence number of the input prefix in the training sample set.
  + The second column is the input prefix.
 2. The second ~beam_size + 1 line is the generated result, also separated into two columns with \t:
  + The first column is the logarithmic probability of the generated sequence(log probability)
  + The second column is the generated text sequence. The normal generation result will end with the symbol ```<e>```. If it does not end with ```<e>```, it means that the maximum sequence length is exceeded and the forced termination is generated.

To run the codes in this directory, we need to use  v0.10.0 version. If your PaddlePaddle version is lower than this version, please update the PaddlePaddle according to the instructions in [installation document][1]

---

# Scene Text Recognition(STR)

## Introduction to scene text recognition task

Many scene image contains rich text information, and it is very useful  to know  the content and meaning of the images. Therefore, scene text recognition is significant  to learn Images.  For example，the character recognition technology has promoted the development of the applications, such as: [[1][2]] . Which  use deep learning to automatically identify signs of words , and help street view application to obtain more accurate address information.

This example demonstrates how to complete the \* \*  Scene Text Recognition (STR) \* \*  task by  PaddlePaddle. Task  prepare a scene image ,which is  shown in the figure below, `STR` need to identify the corresponding word "keep”.
<p align="center">
<img src="./images/503.jpg"/><br/>
pic 1. The input data sample "keep"
</p>


## Train and forecast by PaddlePaddle

### Install dependency
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Modify configuration parameters

 `config.py` The script contains parameters of model configuration and training. 
\`\`\`python
class TrainerConfig(object):

	  # Whether to use GPU in training or not.
	  use_gpu = True
	  # The number of computing threads.
	  trainer_count = 1
	
	  # The training batch size.
	  batch_size = 10
	
	  ...


class ModelConfig(object):

	  # Number of the filters for convolution group.
	  filter_num = 8
	
	  ...
\`\`\`

Modifying the `config.py` script can adjust the parameters. For example, modify `use_gpu` parameters to specify whether GPU is used for training.

### model training
In the training script [./train.py][3], the following parameters we need to assign

\`\`\`
Options:
  --train\_file\_list\_path TEXT  The path of the file which contains path list
	                           of train image files.  [required]
  --test\_file\_list\_path TEXT   The path of the file which contains path list
	                           of test image files.  [required]
  --label\_dict\_path TEXT       The path of label dictionary. If this parameter
	                           is set, but the file does not exist, label
	                           dictionay will be built from the training data
	                           automatically.  [required]
  --model\_save\_dir TEXT        The path to save the trained models (default:
	                           'models').
  --help                       Show this message and exit.

\`\`\`

- `train_file_list` ：The list file of training data, which contain file path and corresponding  text label for each row.
\`\`\`
word\_1.png, "PROPER"
word\_2.png, "FOOD"
\`\`\`
- `test_file_list` ：The list file of the test data .The format is the  same as above.
- `label_dict_path` ：The path of the lable  in the training data. If the dictionary file does not exist in the specified path, the program will automatically generate the markup dictionary using the tagged data from the training data.
- `model_save_dir` ：The saved path of model parameters .`./models` by default.

### The process of specific implementation:

1.Download the data [[2][4]] (Task 2.3: Word Recognition (2013 Edition) from the official website), and there will be three files: `Challenge2_Training_Task3_Images_GT.zip`, `Challenge2_Test_Task3_Images.zip`, and `Challenge2_Test_Task3_GT.txt`.
Corresponding to the training set, the corresponding words, the pictures of the test set and the words corresponding to the test data correspond to the pictures of the training set. Then perform the following commands to extract and move the data to the target folder.

\`\`\`bash
mkdir -p data/train\_data
mkdir -p data/test\_data
unzip Challenge2\_Training\_Task3\_Images\_GT.zip -d data/train\_data
unzip Challenge2\_Test\_Task3\_Images.zip -d data/test\_data
mv Challenge2\_Test\_Task3\_GT.txt data/test\_data
\`\`\`

2.Get the path of the `gt.txt` in the training data folder (data/train\_data) and the path of the `Challenge2_Test_Task3_GT.txt` in the test data folder (data/test\_data).

3.Carry out the following orders for training:
\`\`\`bash
python train.py \\
--train\_file\_list\_path 'data/train\_data/gt.txt' \\
--test\_file\_list\_path 'data/test\_data/Challenge2\_Test\_Task3\_GT.txt' \\
--label\_dict\_path 'label\_dict.txt'
\`\`\`
4.During training, the model parameters are automatically backed up to the specified directory, which is stored in the `./models` directory by default.


### Forecast
The prediction is partly done by `infer.py`, and it uses the best path decoding algorithm, that is to select a maximum probability character at every time step. In use, you need to specify a specific model save path, a picture fixed size, a batch\_size (default 10), a dictionary path, and a list file of the picture file in the `infer.py`. The following code is executed:
\`\`\`bash
python infer.py \\
--model\_path 'models/params\_pass\_00000.tar.gz' \\
--image\_shape '173,46' \\
--label\_dict\_path 'label\_dict.txt' \\
--infer\_file\_list\_path 'data/test\_data/Challenge2\_Test\_Task3\_GT.txt'
\`\`\`


### Other datasets

-   [SynthText in the Wild Dataset][5](41G)
-   [ICDAR 2003 Robust Reading Competitions][6]

### Matters needing attention

- Because the `warp CTC` of the model relies on the implementation of CUDA, this model only supports GPU operation.
- The parameters of the model. The occupied memory is relatively large, the actual implementation can be adjusted by `batch_size` to control the memory usage.
- The data set used in this example is small. If necessary, we can use another larger data set [[3][7]] to train the model.

## Reference

1. [Google Now Using ReCAPTCHA To Decode Street View Addresses][8]
2. [Focused Scene Text][9]
3. [SynthText in the Wild Dataset][10]

[1]:	http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html
[2]:	#%E5%8F%82%E8%80%83%E6%96%87%E7%8C%AE
[3]:	./train.py
[4]:	#%E5%8F%82%E8%80%83%E6%96%87%E7%8C%AE
[5]:	http://www.robots.ox.ac.uk/~vgg/data/scenetext/
[6]:	http://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2003_Robust_Reading_Competitions
[7]:	#%E5%8F%82%E8%80%83%E6%96%87%E7%8C%AE
[8]:	https://techcrunch.com/2012/03/29/google-now-using-recaptcha-to-decode-street-view-addresses/
[9]:	http://rrc.cvc.uab.es/?ch=2&com=introduction
[10]:	http://www.robots.ox.ac.uk/~vgg/data/scenetext/
Running sample code in this directory requires PaddelPaddle v0.10.0 and later. If the PaddlePaddle on your device is lower than this version, please follow the instructions in [installation document](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html) and make an update.
---

# Chinese Ancient Poetry Generation

## Introduction
On an encoder-decoder neural network model, we perform sequence-to-sequence training with The Complete Tang Poems. The model should generate the verse after the given input verse.

The encoders and decoders in the model all use a stacked bi-directional LSTM which, by default, has three layers and attention.

The following is a brief directory structure and description of this example：

```text
.
├── data                 # store training data and dictionary
│   ├── download.sh      # download raw data
├── README.md            # documentation
├── index.html           # document (html format)
├── preprocess.py        # raw data preprocessing
├── generate.py          # generate verse script
├── network_conf.py      # model definition
├── reader.py            # data reading interface
├── train.py             # training script
└── utils.py             # define utility functions
```

## Data Processing
### Raw Data Source
The training data of this example is The Complete Tang Poems in the [Chinese ancient poetry database](https://github.com/chinese-poetry/chinese-poetry). There are about 54,000 Tang poems.

### Downloading Raw Data
```bash
cd data && ./download.sh && cd ..
```
### Data Preprocessing
```bash
python preprocess.py --datadir data/raw --outfile data/poems.txt --dictfile data/dict.txt
```

After the above script is executed, the processed training data "poems.txt" and dictionary "dict.txt" will be generated. The dictionary's unit is word, and it is constructed by words with a frequency of at least 10.

Divided into three columns, each line in poems.txt contains the title, author, and content of a poem. Verses of a poem are separated by`.`.

Training data example:
```text
登鸛雀樓  王之渙  白日依山盡.黃河入海流.欲窮千里目.更上一層樓
觀獵      李白   太守耀清威.乘閑弄晚暉.江沙橫獵騎.山火遶行圍.箭逐雲鴻落.鷹隨月兔飛.不知白日暮.歡賞夜方歸
晦日重宴  陳嘉言  高門引冠蓋.下客抱支離.綺席珍羞滿.文場翰藻摛.蓂華彫上月.柳色藹春池.日斜歸戚里.連騎勒金羈
```

When the model is trained, each verse is used as a model input, and the next verse is used as a prediction target.


## Model Training
The command line arguments in the training script, ["train.py"](./train.py), can be viewed with `python train.py --help`. The main parameters are as follows:
- `num_passes`: number of passes
- `batch_size`: batch size
- `use_gpu`: whether to use GPU
- `trainer_count`: number of trainers, the default is 1
- `save_dir_path`: model storage path, the default is the current directory under the models directory
- `encoder_depth`: model encoder LSTM depth, default 3
- `decoder_depth`: model decoder LSTM depth, default 3
- `train_data_path`: training data path
- `word_dict_path`: data dictionary path
- `init_model_path`: initial model path, no need to specify at the start of training

### Training Execution
```bash
python train.py \
    --num_passes 50 \
    --batch_size 256 \
    --use_gpu True \
    --trainer_count 1 \
    --save_dir_path models \
    --train_data_path data/poems.txt \
    --word_dict_path data/dict.txt \
    2>&1 | tee train.log
```
After each pass training, the model parameters are saved under directory "models". Training logs are stored in "train.log".

### Optimal Model Parameters
Find the pass with the lowest cost and use the model parameters corresponding to the pass for subsequent prediction.
```bash
python -c 'import utils; utils.find_optiaml_pass("./train.log")'
```

## Generating Verses
Use the ["generate.py"](./generate.py) script to generate the next verse for the input verses. Command line arguments can be viewed with `python generate.py --help`.
The main parameters are described as follows:
- `model_path`: trained model parameter file
- `word_dict_path`: data dictionary path
- `test_data_path`: input data path
- `batch_size`: batch size, default is 1
- `beam_size`: search size in beam search, the default is 5
- `save_file`: output save path
- `use_gpu`: whether to use GPU

### Perform Generation
For example, save the verse `孤帆遠影碧空盡` in the file `input.txt` as input. To predict the next sentence, execute the command:
```bash
python generate.py \
    --model_path models/pass_00049.tar.gz \
    --word_dict_path data/dict.txt \
    --test_data_path input.txt \
    --save_file output.txt
```
The result will be saved in the file "output.txt". For the above example input, the generated verses are as follows:
```text
-9.6987     萬 壑 清 風 黃 葉 多
-10.0737    萬 里 遠 山 紅 葉 深
-10.4233    萬 壑 清 波 紅 一 流
-10.4802    萬 壑 清 風 黃 葉 深
-10.9060    萬 壑 清 風 紅 葉 多
```

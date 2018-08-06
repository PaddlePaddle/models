The minimum PaddlePaddle version needed for the code sample in this directory is v0.11.0. If you are on a version of PaddlePaddle earlier than v0.11.0, [please update your installation](http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html).

---

# Chinese Poem Generator

## Introduction
The Chinese poem generator is based on encoder-decoder sequence to sequence network model. Using full poems as training data, when given a poem sentence, the model could generate the next poem sentence.

Encoders and decoders in the model use a stacked bi-directional LSTM with attention mechanism. The default number of LSTM layers is 3.

The following is the directory structure of this example:

```text
.
├── data  
│   ├── download.sh  
├── README.md  
├── index.html           # doc(html)
├── preprocess.py        # preprocess original data
├── generate.py          # generate poem sentences
├── network_conf.py      # model config
├── reader.py            # data reader
├── train.py             # train model
└── utils.py             # utils
```

## Data Processing
### Original Data Source
The data used in the example is from [中华古诗词数据库](https://github.com/chinese-poetry/chinese-poetry). About 54,000 Tang poems in the database are used as training data.

### Download Original Data
```bash
cd data && ./download.sh && cd ..
```
### Preprocess Data
```bash
python preprocess.py --datadir data/raw --outfile data/poems.txt --dictfile data/dict.txt
```
The script above generates two data files: `poems.txt` and `dict.txt`. `poems.txt` contains preprocessed training data. `dict.txt` is a dictionary file, which contains Chinese characters which have appeared for at least 10 times.  

Each row of `poems.txt`, which contains information of a single poem, is split into 3 columns: title, author and poem content. Poem sentences are separated by `.`.

Data Sample:
```text
登鸛雀樓  王之渙  白日依山盡.黃河入海流.欲窮千里目.更上一層樓
觀獵      李白   太守耀清威.乘閑弄晚暉.江沙橫獵騎.山火遶行圍.箭逐雲鴻落.鷹隨月兔飛.不知白日暮.歡賞夜方歸
晦日重宴  陳嘉言  高門引冠蓋.下客抱支離.綺席珍羞滿.文場翰藻摛.蓂華彫上月.柳色藹春池.日斜歸戚里.連騎勒金羈
```

During the training process, the input of the model is a poem sentence and the prediction target is its following sentence.

## Model Training

Run `python train.py --help` to see [train.py](./train.py)'s command line arguments. Some important arguments are：
- `num_passes`: number of pass during training
- `batch_size`: batch size
- `use_gpu`: use GPU or not
- `trainer_count`: number of trainers, default: 1
- `save_dir_path`: path for saving trained model, default: `models` directory
- `encoder_depth`: LSTM depth of encoders, default: 3
- `decoder_depth`: LSTM depth of decoders, default: 3
- `train_data_path`: training data path
- `word_dict_path`: dictionary path
- `init_model_path`: model to start training with, unnecessary if train from scratch

### Start Training
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
After each training pass, model parameters will be saved in `models` directory. The training log will be saved in `train.log`.

### Optimized Model Parameters
Search the pass with the minimum cost. Use the corresponding model parameters for the prediction.
```bash
python -c 'import utils; utils.find_optiaml_pass("./train.log")'
```

## Generate Poem Sentences
Use [generate.py](./generate.py) script to generate the next sentence of the input poem sentence. Run `python generate.py --help` to see command line arguments.
Some important arguments are:
- `model_path`: trained model parameter file
- `word_dict_path`: dictionary path
- `test_data_path`: test data path
- `batch_size`: batch size
- `beam_size`: beam search range，default: 5
- `save_file`: output directory
- `use_gpu`: use GPU or not


### Generate Poem Sentences
For example, `input.txt` contains a poem sentence: `孤帆遠影碧空盡`. Use `input.txt` as the input file to generate the next poem sentence of `孤帆遠影碧空盡`.
```bash
python generate.py \
    --model_path models/pass_00049.tar.gz \
    --word_dict_path data/dict.txt \
    --test_data_path input.txt \
    --save_file output.txt

```
The result will be saved into `output.txt`. For the above input, generated poem sentences are as follows:
```text
-9.6987     萬 壑 清 風 黃 葉 多
-10.0737    萬 里 遠 山 紅 葉 深
-10.4233    萬 壑 清 波 紅 一 流
-10.4802    萬 壑 清 風 黃 葉 深
-10.9060    萬 壑 清 風 紅 葉 多
```

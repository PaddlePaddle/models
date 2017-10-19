# Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question Answering

This model implements the work in the following paper:

Peng Li, Wei Li, Zhengyan He, Xuguang Wang, Ying Cao, Jie Zhou, and Wei Xu. Dataset and Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question Answering. [arXiv:1607.06275](https://arxiv.org/abs/1607.06275).

If you use the dataset/code in your research, please cite the above paper:

```text
@article{li:2016:arxiv,
  author  = {Li, Peng and Li, Wei and He, Zhengyan and Wang, Xuguang and Cao, Ying and Zhou, Jie and Xu, Wei},
  title   = {Dataset and Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question Answering},
  journal = {arXiv:1607.06275v2},
  year    = {2016},
  url     = {https://arxiv.org/abs/1607.06275v2},
}
```


# Installation

1. Install PaddlePaddle v0.10.5 by the following commond. Note that v0.10.0 is not supported.
    ```bash
    # either one is OK
    # CPU
    pip install paddlepaddle
    # GPU
    pip install paddlepaddle-gpu
    ```
2. Download the [WebQA](http://idl.baidu.com/WebQA.html) dataset by running
   ```bash
   cd data && ./download.sh && cd ..
   ```

#Hyperparameters

All the hyperparameters are defined in `config.py`. The default values are aligned with the paper.

# Training

Training can be launched using the following command:

```bash
PYTHONPATH=data/evaluation:$PYTHONPATH python train.py 2>&1 | tee train.log
```
# Validation and Test

WebQA provoides two versions of validation and test sets.  Automatic valiation and test can be lauched by

```bash
PYTHONPATH=data/evaluation:$PYTHONPATH python val_and_test.py models [ann|ir]
```

where

* `models`: the directory where model files are stored. You can use `models` if `config.py` is not changed.
* `ann`: using the validation and test sets with annotated evidence.
* `ir`: using the validation and test sets with retrieved evidence.

Note that validation and test can run simultaneously with training. `val_and_test.py` will handle the synchronization related problems.

Intermediate results are stored in the directory `tmp`. You can delete them safely after validation and test.

The results should be comparable with those shown in Table 3 in the paper.

# Inferring using a Trained Model

Infer using a trained model by running:
```bash
PYTHONPATH=data/evaluation:$PYTHONPATH python infer.py \
  MODEL_FILE \
  INPUT_DATA \
  OUTPUT_FILE \
  2>&1 | tee infer.log
```

where

* `MODEL_FILE`: a trained model produced by `train.py`.
* `INPUT_DATA`: input data in the same format as the validation/test sets of the WebQA dataset.
* `OUTPUT_FILE`: results in the format specified in the WebQA dataset for the evaluation scripts.
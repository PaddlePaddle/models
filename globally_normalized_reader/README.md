The minimum PaddlePaddle version needed for the code sample in this directory is v0.11.0. If you are on a version of PaddlePaddle earlier than v0.11.0, [please update your installation](http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html).

---

# Globally Normalized Reader

This model implements the work in the following paper:

Jonathan Raiman and John Miller. Globally Normalized Reader. Empirical Methods in Natural Language Processing (EMNLP), 2017.

If you use the dataset/code in your research, please cite the above paper:

```text
@inproceedings{raiman2015gnr,
    author={Raiman, Jonathan and Miller, John},
    booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
    title={Globally Normalized Reader},
    year={2017},
}
```

You can also visit https://github.com/baidu-research/GloballyNormalizedReader to get more information.


# Installation

1. Please use [docker image](http://doc.paddlepaddle.org/develop/doc/getstarted/build_and_install/docker_install_en.html) to install the latest PaddlePaddle, by running:
    ```bash
    docker pull paddledev/paddle
    ```
2. Download all necessary data by running:
    ```bash
    cd data && ./download.sh && cd ..
    ```
3. Preprocess and featurizer data:
    ```bash
    python featurize.py --datadir data --outdir data/featurized  --glove-path data/glove.840B.300d.txt
    ```

# Training a Model

- Configurate the model by modifying `config.py` if needed, and then run:

    ```bash
    python train.py 2>&1 | tee train.log
    ```

# Inferring by a Trained Model

- Infer by a trained model by running:
   ```bash
   python infer.py \
     --model_path models/pass_00000.tar.gz \
     --data_dir data/featurized/ \
     --batch_size 2 \
     --use_gpu 0 \
     --trainer_count 1 \
     2>&1 | tee infer.log
   ```

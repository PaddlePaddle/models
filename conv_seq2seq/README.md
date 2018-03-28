The minimum PaddlePaddle version needed for the code sample in this directory is v0.11.0. If you are on a version of PaddlePaddle earlier than v0.11.0, [please update your installation](http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html).

---

# Convolutional Sequence to Sequence Learning
This model implements the work in the following paper:

Jonas Gehring, Micheal Auli, David Grangier, et al. Convolutional Sequence to Sequence Learning. Association for Computational Linguistics (ACL), 2017

# Data Preparation
- The data used in this tutorial can be downloaded by runing:

    ```bash
    sh download.sh
    ```

- Each line in the data file contains one sample and each sample consists of a source sentence and a target sentence. And the two sentences are seperated by '\t'. So, to use your own data, it should be organized as follows:

    ```
    <source sentence>\t<target sentence>
    ```

# Training a Model
- Modify the following script if needed and then run:

  ```bash
  python train.py \
      --train_data_path ./data/train \
      --test_data_path ./data/test \
      --src_dict_path ./data/src_dict \
      --trg_dict_path ./data/trg_dict \
      --enc_blocks "[(256, 3)] * 5" \
      --dec_blocks "[(256, 3)] * 3" \
      --emb_size 256 \
      --pos_size 200 \
      --drop_rate 0.2 \
      --use_bn False \
      --use_gpu False \
      --trainer_count 1 \
      --batch_size 32 \
      --num_passes 20 \
      >train.log 2>&1
  ```

# Inferring by a Trained Model
- Infer by a trained model by running:

  ```bash
  python infer.py \
      --infer_data_path ./data/dev \
      --src_dict_path ./data/src_dict \
      --trg_dict_path ./data/trg_dict \
      --enc_blocks "[(256, 3)] * 5" \
      --dec_blocks "[(256, 3)] * 3" \
      --emb_size 256 \
      --pos_size 200 \
      --drop_rate 0.2 \
      --use_bn False \
      --use_gpu False \
      --trainer_count 1 \
      --max_len 100 \
      --batch_size 256 \
      --beam_size 1 \
      --is_show_attention False \
      --model_path ./params.pass-0.tar.gz \
      1>infer_result 2>infer.log
    ```

# Notes
Since PaddlePaddle of current version doesn't support weight normalization, we use batch normalization instead to confirm convergence when the network is deep.

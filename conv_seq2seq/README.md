# Convolutional Sequence to Sequence Learning
This model implements the work in the following paper:

Jonas Gehring, Micheal Auli, David Grangier, et al. Convolutional Sequence to Sequence Learning. Association for Computational Linguistics (ACL), 2017

# Data Preparation

- In this tutorial, each line in a data file contains one sample and each sample consists of a source sentence and a target sentence. And the two sentences are seperated by '\t'. So, to use your own data, it should be organized as follows:

  ```
  <source sentence>\t<target sentence>
  ```

# Training a Model
- Modify the following script if needed and then run:

    ```bash
    python train.py \
      --train_data_path ./data/train_data \
      --test_data_path ./data/test_data \
      --src_dict_path ./data/src_dict \
      --trg_dict_path ./data/trg_dict \
      --enc_blocks "[(256, 3)] * 5" \
      --dec_blocks "[(256, 3)] * 3" \
      --emb_size 256 \
      --pos_size 200 \
      --drop_rate 0.1 \
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
      --infer_data_path ./data/infer_data \
      --src_dict_path ./data/src_dict \
      --trg_dict_path ./data/trg_dict \
      --enc_blocks "[(256, 3)] * 5" \
      --dec_blocks "[(256, 3)] * 3" \
      --emb_size 256 \
      --pos_size 200 \
      --drop_rate 0.1 \
      --use_gpu False \
      --trainer_count 1 \
      --max_len 100 \
      --beam_size 1 \
      --model_path ./params.pass-0.tar.gz \
      1>infer_result 2>infer.log
    ```

# Notes

Currently, beam search will forward the encoder multiple times when predicting each target word, which requires extra computations. And we will fix it later.

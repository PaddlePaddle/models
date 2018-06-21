export CUDA_VISIBLE_DEVICES=0
python eval.py \
--input_images_dir="/root/.cache/paddle/dataset/ctc_data/data/test_images/" \
--input_images_list="/root/.cache/paddle/dataset/ctc_data/data/test.list" \
--use_gpu=True \
--model_path="/home/work/workspace/models/fluid/ocr_recognition/models/model_60000"


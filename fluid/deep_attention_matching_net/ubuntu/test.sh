export CUDA_VISIBLE_DEVICES=0
python -u ../test_and_evaluate.py --use_cuda \
                --data_path ./data/data.pkl \
                --save_path ./step_3900 \
                --model_path ./models/step_3900 \
                --batch_size 200 \
                --vocab_size 434512 \
                --emb_size 200 \
                --_EOS_ 28270
                

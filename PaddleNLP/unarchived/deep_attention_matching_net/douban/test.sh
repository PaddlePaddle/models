export CUDA_VISIBLE_DEVICES=0
python -u ../test_and_evaluate.py --use_cuda \
                --ext_eval \
                --data_path ./data/data.pkl \
                --save_path ./eval_3900 \
                --model_path models/step_3900 \
                --channel1_num 16 \
                --batch_size 200 \
                --vocab_size 172130 \
                --emb_size 200 \
                --_EOS_ 1
                

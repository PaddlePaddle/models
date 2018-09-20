export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u ../test_and_evaluate.py --use_cuda \
                --data_path ./data/data.pkl \
                --save_path ./ \
                --model_path models/step_10000 \
                --batch_size 100 \
                --vocab_size 434512 \
                --emb_size 200 \
                --_EOS_ 28270
                

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u ../train_and_evaluate.py --use_cuda \
                --data_path ./data/data.pkl \
                --ext_eval \
                --word_emb_init ./data/word_embedding.pkl \
                --save_path ./models \
                --batch_size 100 \
                --vocab_size 172130 \
                --emb_size 200 \
                --_EOS_ 1
                

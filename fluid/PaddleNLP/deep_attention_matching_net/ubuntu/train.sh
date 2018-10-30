export CUDA_VISIBLE_DEVICES=0
export FLAGS_eager_delete_tensor_gb=0.0
python -u ../train_and_evaluate.py --use_cuda \
                --data_path ./data/data.pkl \
                --word_emb_init ./data/word_embedding.pkl \
                --save_path ./models \
                --use_pyreader \
                --batch_size 256 \
                --vocab_size 434512 \
                --emb_size 200 \
                --_EOS_ 28270
                

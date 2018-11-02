export CUDA_VISIBLE_DEVICES=2
python run.py   \
--trainset 'data/preprocessed/trainset/search.train.json' \
           'data/preprocessed/trainset/zhidao.train.json' \
--devset 'data/preprocessed/devset/search.dev.json' \
         'data/preprocessed/devset/zhidao.dev.json' \
--testset 'data/preprocessed/testset/search.test.json' \
          'data/preprocessed/testset/zhidao.test.json' \
--vocab_dir 'data/vocab' \
--use_gpu true \
--save_dir ./models \
--pass_num 10 \
--learning_rate 0.001 \
--batch_size 32 \
--embed_size 300 \
--hidden_size 150 \
--max_p_num 5 \
--max_p_len 500 \
--max_q_len 60 \
--max_a_len 200 \
--weight_decay 0.0001 \
--drop_rate 0.2 $@\

#! /bin/sh

path=$1

python ./src/id2word.py data/vocab.source.32000 < ${path} > ${path}_word
head -n 3003 ${path}_word > ${path}_word_tmp
mv ${path}_word_tmp ${path}_word
cat ${path}_word | sed 's/@@ //g' > ${path}.trans.post
python ./src/bleu_hook.py --reference wmt16_en_de/newstest2014.tok.de --translation ${path}.trans.post

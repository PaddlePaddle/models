The minimum PaddlePaddle version needed for the code sample in this directory is the lastest develop branch. If you are on a version of PaddlePaddle earlier than this, [please update your installation](http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html).

---

# Text Classification

## Data Preparation
```
wget http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz
tar zxf aclImdb_v1.tar.gz
```

## Training
```
python train.py --dict_path 'aclImdb/imdb.vocab'
```

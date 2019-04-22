#!/bin/bash

#The gdown.pl script comes from: https://github.com/circulosmeos/gdown.pl
./gdown.pl https://drive.google.com/open?id=0B7XZSACQf0KdenRmMk8yVUU5LWc dataset-train-diginetica.zip
unzip dataset-train-diginetica.zip "train-item-views.csv"
sed -i '1d' train-item-views.csv
sed -i '1i session_id;user_id;item_id;timeframe;eventdate' train-item-views.csv
mkdir diginetica

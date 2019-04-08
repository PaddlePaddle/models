#! /bin/bash

echo "begin download data"
mkdir raw_data && cd raw_data
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
gzip -d reviews_Electronics_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
gzip -d meta_Electronics.json.gz
echo "download data successful"

cd ..
python convert_pd.py
python remap_id.py

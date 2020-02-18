#!/bin/bash
mkdir data

pushd ./ && cd ./data
##downloads the 4 widely used KBC dataset
wget --no-check-certificate https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz -O fb15k.tgz
wget --no-check-certificate https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz -O wordnet-mlj12.tar.gz
wget --no-check-certificat https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip -O FB15K-237.2.zip
wget --no-check-certificat https://raw.githubusercontent.com/TimDettmers/ConvE/master/WN18RR.tar.gz -O WN18RR.tar.gz 

##downloads the path query dataset
wget --no-check-certificate https://worksheets.codalab.org/rest/bundles/0xdb6b691c2907435b974850e8eb9a5fc2/contents/blob/ -O freebase_paths.tar.gz
wget --no-check-certificate https://worksheets.codalab.org/rest/bundles/0xf91669f6c6d74987808aeb79bf716bd0/contents/blob/ -O wordnet_paths.tar.gz

## organize the train/valid/test files by renaming
#fb15k
tar -xvf fb15k.tgz 
mv FB15k fb15k
mv ./fb15k/freebase_mtr100_mte100-train.txt ./fb15k/train.txt
mv ./fb15k/freebase_mtr100_mte100-test.txt ./fb15k/test.txt
mv ./fb15k/freebase_mtr100_mte100-valid.txt ./fb15k/valid.txt

#wn18
tar -zxvf wordnet-mlj12.tar.gz && mv wordnet-mlj12 wn18
mv wn18/wordnet-mlj12-train.txt wn18/train.txt
mv wn18/wordnet-mlj12-test.txt wn18/test.txt
mv wn18/wordnet-mlj12-valid.txt wn18/valid.txt


#fb15k237
unzip FB15K-237.2.zip && mv Release fb15k237

#wn18rr
mkdir wn18rr && tar -zxvf WN18RR.tar.gz -C wn18rr

#pathqueryWN
mkdir pathqueryWN && tar -zxvf wordnet_paths.tar.gz -C pathqueryWN

#pathqueryFB
mkdir pathqueryFB && tar -zxvf freebase_paths.tar.gz -C pathqueryFB

##rm tmp zip files
# rm ./*.gz
# rm ./*.tgz
# rm ./*.zip

popd

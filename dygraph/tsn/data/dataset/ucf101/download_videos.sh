#! /usr/bin/bash env

wget --no-check-certificate "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
unrar x UCF101.rar
mv ./UCF-101 ./videos
rm -rf ./UCF101.rar


#!/usr/bin/env bash

set -x

echo "==============JOB BEGIN============"

# User configurations
HADOOP_FS_NAME=afs://cygnus.afs.baidu.com:9902
HADOOP_UGI=paddle,paddle
HDFS_PATH=/user/paddle/liuyi05/data/imagenet/resized
FILE_PREFIX=imagenet_resized.tar.part
UNTAR_DIRNAME=imagenet_resized

echo "Downloading data"
echo "Getting ImageNet dataset..."
for((i=0;i<=9;++i)); do
  hadoop fs -D fs.default.name=$HADOOP_FS_NAME -D hadoop.job.ugi=$HADOOP_UGI -get $HDFS_PATH/$FILE_PREFIX$i ./ &
done
wait
cat $FILE_PREFIX* > _joined.tar
echo "untar..."
tar xf _joined.tar
rm -f _joined.tar $FILE_PREFIX?
mkdir -p data
ln -snf $PWD/$UNTAR_DIRNAME data/ILSVRC2012

echo "Finish preparing dataset"

rm -rf mylog
tar xf thirdparty/image_classification.tar

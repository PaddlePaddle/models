#!/usr/bin/env bash

## upload model
#echo "tar model"
#save_name="`date +%y%m%d`_models.tar.gz"
#tar -zcvf ${save_name} output
#hadoop fs -Dfs.default.name=afs://cygnus.afs.baidu.com:9902 -Dhadoop.job.ugi=paddle,paddle -rmr /user/paddle/shenliang/transformer/${save_name}
#hadoop fs -Dfs.default.name=afs://cygnus.afs.baidu.com:9902 -Dhadoop.job.ugi=paddle,paddle -put ${save_name} /user/paddle/shenliang/transformer/.
echo "success train.."

echo "==============Job Done=============="


url=http://dam-data.cdn.bcebos.com/douban.tar.gz
md5=e07ca68f21c20e09efb3e8b247194405

if  [ ! -e douban.tar.gz ]; then
    wget -c $url
fi

echo "Checking md5 sum ..."
md5sum_tmp=`md5sum douban.tar.gz | cut -d ' ' -f1`

if [ $md5sum_tmp !=  $md5 ]; then
    echo "Md5sum check failed, please remove and redownload douban.tar.gz"
    exit 1
fi

echo "Untar douban.tar.gz ..."

tar -xzvf douban.tar.gz 


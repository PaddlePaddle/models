ubuntu_url=http://dam-data.cdn.bcebos.com/ubuntu.tar.gz
ubuntu_md5=9d7db116a040530a16f68dc0ab44e4b6

if  [ ! -e ubuntu.tar.gz ]; then
    wget -c $ubuntu_url
fi

echo "Checking md5 sum ..."
md5sum_tmp=`md5sum ubuntu.tar.gz | cut -d ' ' -f1`

if [ $md5sum_tmp !=  $ubuntu_md5 ]; then
    echo "Md5sum check failed, please remove and redownload ubuntu.tar.gz"
    exit 1
fi

echo "Untar ubuntu.tar.gz ..."

tar -xzvf ubuntu.tar.gz 
mv data ubuntu

douban_url=http://dam-data.cdn.bcebos.com/douban.tar.gz
douban_md5=e07ca68f21c20e09efb3e8b247194405

if  [ ! -e douban.tar.gz ]; then
    wget -c $douban_url
fi

echo "Checking md5 sum ..."
md5sum_tmp=`md5sum douban.tar.gz | cut -d ' ' -f1`

if [ $md5sum_tmp !=  $douban_md5 ]; then
    echo "Md5sum check failed, please remove and redownload douban.tar.gz"
    exit 1
fi

echo "Untar douban.tar.gz ..."

tar -xzvf douban.tar.gz 
mv data douban


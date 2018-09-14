url=http://dam-data.cdn.bcebos.com/ubuntu.tar.gz
md5=9d7db116a040530a16f68dc0ab44e4b6

if  [ ! -e ubuntu.tar.gz ]; then
    wget -c $url
fi

echo "Checking md5 sum ..."
md5sum_tmp=`md5sum ubuntu.tar.gz | cut -d ' ' -f1`

if [ $md5sum_tmp !=  $md5 ]; then
    echo "Md5sum check failed, please remove and redownload ubuntu.tar.gz"
    exit 1
fi

echo "Untar ubuntu.tar.gz ..."

tar -xzvf ubuntu.tar.gz 

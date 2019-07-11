url=http://deep-asr-data.gz.bcebos.com/aishell_pretrained_model.tar.gz
md5=7b51bde64e884f43901b7a3461ccbfa3

wget -c $url

echo "Checking md5 sum ..."
md5sum_tmp=`md5sum aishell_pretrained_model.tar.gz | cut -d ' ' -f1`

if [ $md5sum_tmp !=  $md5 ]; then
    echo "Md5sum check failed, please remove and redownload "
          "aishell_pretrained_model.tar.gz."
    exit 1
fi

tar xvf aishell_pretrained_model.tar.gz 

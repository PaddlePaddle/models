data_dir=~/.cache/paddle/dataset/speech/deep_asr_data/aishell
data_url='http://deep-asr-data.gz.bcebos.com/aishell_data.tar.gz'
lst_url='http://deep-asr-data.gz.bcebos.com/aishell_lst.tar.gz'
md5=e017d858d9e509c8a84b73f673f08b9a

if [ ! -e $data_dir ]; then
    mkdir -p $data_dir
fi

if [ ! -e $data_dir/aishell_data.tar.gz ]; then
    echo "Download $data_dir/aishell_data.tar.gz ..."
    wget -c  -P $data_dir $data_url
else
    echo "Skip downloading for $data_dir/aishell_data.tar.gz has already existed!"
fi

echo "Checking md5 sum ..."
md5sum_tmp=`md5sum $data_dir/aishell_data.tar.gz | cut -d ' ' -f1`

if [ $md5sum_tmp !=  $md5 ]; then
    echo "Md5sum check failed, please remove and redownload "
          "$data_dir/aishell_data.tar.gz"
    exit 1
fi

echo "Untar aishell_data.tar.gz ..."
tar xzf $data_dir/aishell_data.tar.gz -C $data_dir

if [ ! -e data ]; then
    mkdir data
fi

echo "Download and untar lst files ..."
wget -c -P data $lst_url
tar xvf data/aishell_lst.tar.gz -C data

ln -s $data_dir data/aishell

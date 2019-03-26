set -e
if [ "x${IMAGENET_USERNAME}" == x -o "x${IMAGENET_ACCESS_KEY}" == x ];then
  echo "Please create an account on image-net.org."
  echo "It will provide you a pair of username and accesskey to download imagenet data."
  read -p "Username: " IMAGENET_USERNAME
  read -p "Accesskey: " IMAGENET_ACCESS_KEY
fi

root_url=http://www.image-net.org/challenges/LSVRC/2012/nnoupb
valid_tar=ILSVRC2012_img_val.tar
train_tar=ILSVRC2012_img_train.tar
train_folder=train/
valid_folder=val/

echo "Download imagenet training data..."
mkdir -p ${train_folder}
wget -nd -c ${root_url}/${train_tar}
tar xf ${train_tar} -C ${train_folder}

cd ${train_folder}
for x in `ls *.tar`
do
  filename=`basename $x .tar`
  mkdir -p $filename
  tar -xf $x -C $filename
  rm -rf $x
done
cd -

echo "Download imagenet validation data..."
mkdir -p ${valid_folder}
wget -nd -c ${root_url}/${valid_tar}
tar xf ${valid_tar} -C ${valid_folder}

echo "Download imagenet label file: val_list.txt & train_list.txt"
label_file=ImageNet_label.tgz
label_url=http://paddle-imagenet-models.bj.bcebos.com/${label_file}
wget -nd -c ${label_url}
tar zxf ${label_file}


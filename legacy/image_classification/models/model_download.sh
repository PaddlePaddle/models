#! /usr/bin/env bash

function download() {
    URL=$1
    MD5=$2
    TARGET=$3

    if [ -e $TARGET ]; then
        md5_result=`md5sum $TARGET | awk -F[' '] '{print $1}'`
        if [ $MD5 == $md5_result ]; then
            echo "$TARGET already exists, download skipped."
            return 0
        fi
    fi

    wget -c $URL -O "$TARGET"
    if [ $? -ne 0 ]; then
        return 1
    fi

    md5_result=`md5sum $TARGET | awk -F[' '] '{print $1}'`
    if [ ! $MD5 == $md5_result ]; then
        return 1
    fi
}

case "$1" in
    "ResNet50")
		URL="http://cloud.dlnel.org/filepub/?uuid=f63f237a-698e-4a22-9782-baf5bb183019"
		MD5="eb4d7b5962c9954340207788af0d6967"	 
        ;;
    "ResNet101")
		URL="http://cloud.dlnel.org/filepub/?uuid=3d5fb996-83d0-4745-8adc-13ee960fc55c"
		MD5="7e71f24998aa8e434fa164a7c4fc9c02"
        ;;
    "Vgg16")
		URL="http://cloud.dlnel.org/filepub/?uuid=aa0e397e-474a-4cc1-bd8f-65a214039c2e"
		MD5="e73dc42507e6acd3a8b8087f66a9f395"
        ;;
    *)
        echo "The "$1" model is not provided currently."
		exit 1
        ;;
esac
TARGET="Paddle_"$1".tar.gz"

echo "Download "$1" model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download the model!"
    exit 1
fi


exit 0

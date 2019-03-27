root_url="http://paddle-imagenet-models-name.bj.bcebos.com"
MobileNetV1="MobileNetV1_pretrained.zip"
ResNet50="ResNet50_pretrained.zip"

wget ${root_url}/${MobileNetV1}
unzip ${MobileNetV1}

wget ${root_url}/${ResNet50}
unzip ${ResNet50}

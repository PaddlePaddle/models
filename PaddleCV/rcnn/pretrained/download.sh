DIR="$( cd "$(dirname "$0")" )"
cd "$DIR"

# Download the data.
echo "Downloading..."
wget http://paddlemodels.bj.bcebos.com/faster_rcnn/imagenet_resnet50_fusebn.tar.gz
echo "Extracting..."
tar -xf imagenet_resnet50_fusebn.tar.gz

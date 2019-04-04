DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

# Download the data.
echo "Downloading..."
wget http://paddlemodels.bj.bcebos.com/mobilenet_v1_imagenet.tar.gz
echo "Extractint..."
tar -xf mobilenet_v1_imagenet.tar.gz

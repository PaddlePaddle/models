DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

# Download the data.
echo "Downloading..."
wget http://paddlemodels.bj.bcebos.com/ssd_mobilenet_coco.tar.gz
echo "Extractint..."
tar -xf ssd_mobilenet_coco.tar.gz

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

# Download the pretrain weights.
echo "Downloading..."
wget https://paddlemodels.bj.bcebos.com/yolo/darknet53.tar.gz
wget https://paddlemodels.bj.bcebos.com/yolo/yolov3.tar.gz
echo "Extracting..."
tar -xf darknet53.tar.gz
tar -xf yolov3.tar.gz

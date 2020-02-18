DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

# Download the pretrain weights.
echo "Downloading..."
wget https://paddlemodels.bj.bcebos.com/yolo/darknet53.pdparams
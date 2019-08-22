DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

# Download the data.
echo "Downloading..."
wget https://paddlemodels.bj.bcebos.com/mushi.tar
# Extract the data.
echo "Extracting..."
tar xf mushi.tar

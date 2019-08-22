DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

# Download the data.
echo "Downloading..."
wget https://paddlemodels.bj.bcebos.com/insect.tar
# Extract the data.
echo "Extracting..."
tar xf insect.tar

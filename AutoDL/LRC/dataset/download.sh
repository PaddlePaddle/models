DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"
mkdir cifar
cd cifar
# Download the data.
echo "Downloading..."
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# Extract the data.
echo "Extracting..."
tar zvxf cifar-10-python.tar.gz

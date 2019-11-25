DIR="$( cd "$(dirname "$0")" ; pwd -P  )"
cd "$DIR"

echo "Downloading https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip

echo "Unzip modelnet40_ply_hdf5_2048.zip"
unzip modelnet40_ply_hdf5_2048.zip

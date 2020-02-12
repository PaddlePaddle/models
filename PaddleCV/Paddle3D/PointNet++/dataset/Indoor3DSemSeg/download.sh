DIR="$( cd "$(dirname "$0")" ; pwd -P  )"
cd "$DIR"

echo "Downloading https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip"
wget https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip

echo "Unzip indoor3d_sem_seg_hdf5_data.zip"
unzip indoor3d_sem_seg_hdf5_data.zip

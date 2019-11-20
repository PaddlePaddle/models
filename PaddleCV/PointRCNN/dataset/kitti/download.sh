DIR="$( cd "$(dirname "$0")" ; pwd -P  )"
cd "$DIR"

echo "Downloading https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip"
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip

echo "data_object_velodyne.zip"
unzip data_object_velodyne.zip

echo "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip

echo "data_object_image_2.zip"
unzip "data_object_image_2.zip"

echo "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip"
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip

echo "data_object_calib.zip"
unzip data_object_calib.zip

echo "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip

echo "data_object_label_2.zip"
unzip data_object_label_2.zip



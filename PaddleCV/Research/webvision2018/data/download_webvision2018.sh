wget https://data.vision.ee.ethz.ch/cvl/webvision2018/val_images_resized.tar
tar -xvf val_images_resized.tar
rm val_images_resized.tar
wget https://data.vision.ee.ethz.ch/cvl/webvision2018/val_filelist.txt
mv val_images_resized val
mv val_filelist.txt val_list.txt

# Download the dataset
echo "Downloading..."

wget https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar
wget https://www.crcv.ucf.edu/wp-content/uploads/2019/03/UCF101TrainTestSplits-RecognitionTask.zip

# Extract the data.
echo "Extracting..."
unrar x UCF101.rar
unzip UCF101TrainTestSplits-RecognitionTask.zip

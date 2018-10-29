# Download the dataset
echo "Downloading..."
wget http://crcv.ucf.edu/data/UCF101/UCF101.rar
wget http://crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip

# Extract the data.
echo "Extracting..."
unrar x UCF101.rar
unzip UCF101TrainTestSplits-RecognitionTask.zip

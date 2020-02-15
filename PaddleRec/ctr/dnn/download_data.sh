wget --no-check-certificate https://fleet.bj.bcebos.com/ctr_data.tar.gz
tar -zxvf ctr_data.tar.gz
mv ./raw_data ./train_data_full
mkdir train_data && cd train_data
cp ../train_data_full/part-0 ../train_data_full/part-1 ./ && cd ..
mv ./test_data ./test_data_full
mkdir test_data && cd test_data
cp ../test_data_full/part-220 ./  && cd ..
echo "Complete data download."
echo "Full Train data stored in ./train_data_full "
echo "Full Test data stored in ./test_data_full "
echo "Rapid Verification train data stored in ./train_data "
echo "Rapid Verification test data stored in ./test_data "
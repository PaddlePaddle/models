#!/bin/bash

#check data directory
cd ..
echo "Start download data and models.............."
if [ ! -d "data" ]; then
	echo "Directory data does not exist, make new data directory"
	mkdir data
fi
cd data

#check configure file
if [ ! -d "config" ]; then
	echo "config directory not exist........"
	exit 255
else
	if [ ! -f "config/ade.yaml" ]; then
		echo "config file dgu.yaml has been lost........"
		exit 255
	fi
fi

#check and download input data
if [ ! -d "input" ]; then
	echo "Directory input does not exist, make new input directory"
	mkdir input
fi
cd input
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/auto_dialogue_evaluation_dataset-1.0.0.tar.gz
tar -zxvf auto_dialogue_evaluation_dataset-1.0.0.tar.gz
rm auto_dialogue_evaluation_dataset-1.0.0.tar.gz
cd ..

#check and download pretrain model
if [ ! -d "pretrain_model" ]; then
	echo "Directory pretrain_model does not exist, make new pretrain_model directory"
	mkdir pretrain_model
fi

#check and download inferenece model
if [ ! -d "inference_models" ]; then
	echo "Directory inferenece_model does not exist, make new inferenece_model directory"
	mkdir inference_models
fi

#check output
if [ ! -d "output" ]; then
	echo "Directory output does not exist, make new output directory"
	mkdir output
fi

#check saved model
if [ ! -d "saved_models" ]; then
	echo "Directory saved_models does not exist, make new saved_models directory"
	mkdir saved_models
fi

cd saved_models
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/auto_dialogue_evaluation_models.2.0.0.tar.gz
tar -xvf auto_dialogue_evaluation_models.2.0.0.tar.gz
rm auto_dialogue_evaluation_models.2.0.0.tar.gz
echo "Finish.............."

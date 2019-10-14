cd bert_server
export CUDA_VISIBLE_DEVICES=1
sh start.sh
cd ../xlnet_server
export CUDA_VISIBLE_DEVICES=2
sh serve.sh
cd ..

sleep 60
python main_server.py 

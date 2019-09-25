cd bert_server
sh start.sh
cd ../xlnet_server
sh serve.sh
cd ..

sleep 60
python main_server.py 

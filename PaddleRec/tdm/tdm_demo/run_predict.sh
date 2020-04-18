DIRS=`pwd`
data_path="${DIRS}/data"
train_files_path="${data_path}/train"
test_files_path="${data_path}/test"

model_files_path="${DIRS}/model"

thirdparty_path="${DIRS}/thirdparty"
tree_travel_init_path="${thirdparty_path}/travel_list.txt"
tree_layer_init_path="${thirdparty_path}/layer_list.txt"
tree_info_init_path="${thirdparty_path}/tree_info.txt"

export GLOG_v=0

function main() {
    cmd="python predict.py \
        --is_local=1 \
        --cpu_num=1 \
        --batch_size=1 \
        --topK=1 \
        --is_test=1 \
        --save_init_model=1 \
        --train_files_path=${train_files_path} \
        --test_files_path=${test_files_path} \
        --model_files_path=${model_files_path} \
        --tree_travel_init_path=${tree_travel_init_path} \
        --tree_info_init_path=${tree_info_init_path} \
        --tree_layer_init_path=${tree_layer_init_path} " 
    ${cmd}
}

main "$@"

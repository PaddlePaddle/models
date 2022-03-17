source test_tipc/common_func.sh 

function add_check_function(){
    jit_replace='import numpy as np 
import paddle 

def verify_paddle_inference_correctness(layer, path, image_shape):
    from paddle import inference
    import numpy as np
    for i in range(len(image_shape)):
        if image_shape[i] < 0:
            image_shape[i] = 100
    model_file_path = path + ".pdmodel"
    params_file_path = path + ".pdiparams"
    config = inference.Config(model_file_path, params_file_path)
    predictor = inference.create_predictor(config)
    input_names = predictor.get_input_names()
    for name in input_names:
        input_tensor = predictor.get_input_handle(name)
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_handle(output_name)
        output_tensors.append(output_tensor)
    x = np.random.random(size=tuple([1]+image_shape)).astype("float32")
    input_tensor.copy_from_cpu(x)
    predictor.run()
    prob_out = output_tensors[0].copy_to_cpu()

    layer.eval()
    pred = layer(paddle.to_tensor(x))
    correct = np.allclose(pred, prob_out, rtol=1e-4, atol=1e-4)
    absolute_diff = np.abs(pred.numpy() - prob_out)
    max_absolute_diff = np.max(absolute_diff)
    print("max_absolute_diff:", max_absolute_diff)
    assert correct, "Result diff when load and inference:\nlayer max_absolute_diff:{}"\
                  .format(max_absolute_diff)
    print("Successful, dygraph and inference predictions are consistent.")'

    echo "${jit_replace}" > check_inference.py
}

function fun_run_check(){
    tab_num=$1
    dy_model=$2
    model_path=$3
    image_shape=$4

    space=`expr $tab_num \* 2`
    tab=$(printf "%-${space}s" " ")
    line0="${tab}from check_inference import verify_paddle_inference_correctness"
    line1="${tab}layer = ${dy_model}"
    line2="${tab}path = ${model_path}"
    line3="${tab}image_shape = ${image_shape}"
    line4="${tab}verify_paddle_inference_correctness(layer, path, image_shape)"
    echo "${line0}
${line1}
${line2}
${line3}
${line4}" > tmp_file.txt
}



echo "-------- Test inference result --------- "
add_check_function
model_type=$PWD

echo $model_type
## add_check_infer_result

if [[ $model_type =~ "PaddleClas" ]]; then
    echo "PaddleClas"
    # get export file
    export_file=${root_path}/ppcls/engine/engine.py
    # define layer path and img_shape
    layer="model"
    path="save_path"
    image_shape="self.config['Global']['image_shape']"
elif [[ $model_type =~ "PaddleOCR" ]]; then
    echo "PaddleOCR"
    # get export file
    export_file=${root_path}/tools/export_model.py
    # define layer path and img_shape
    layer="model"
    path="save_path"
    image_shape="infer_shape"
elif [[ $model_type =~ "PaddleDetection" ]]; then
    echo "PaddleDetection"
    # get export file
    export_file=${root_path}/ppdet/engine/trainer.py
    # define layer path and img_shape
    layer="static_model"
    path="os.path.join(save_dir, 'model')"
    image_shape="[3,512,512]"
elif [[ $model_type =~ "PaddleGAN" ]]; then
    echo "PaddleGAN"
    # get export file
    export_file=${root_path}/ppgan/models/base_model.py
    # define layer path and img_shape
    layer="static_model"
    path="os.path.join(output_dir, model_name)"
    image_shape="inputs_size[inputs_num]"
fi

echo $export_file

if [[ $model_type =~ "PaddleDetection" ]]; then
    tab_num="3"
    # get insert line
    line_number="670"
elif [[ $model_type =~ "PaddleGAN" ]]; then
    tab_num="3"
    line_number="205"
else 
    tab_num=`cat ${export_file} | grep "jit.save" | awk -F"  " '{print NF-1}'`
    # get insert line
    line_number=`cat ${export_file} | grep -n "jit.save" | awk -F ":" '{print $1}'`
fi

fun_run_check $tab_num $layer $path $image_shape
sed -i "${line_number} r tmp_file.txt" ${export_file}


#读取第一个参数

function run_all_config(){
for file in `ls $1` 
do
    if [ -d $1"/"$file ]; then
        read_dir $1"/"$file
    else
        export_model_cmd="python3.7 tools/export_model.py -c $1'/'$file -o Arch.pretrained=True"
        status_log="./check_clas.log"
        eval $export_model_cmd
        last_status=${PIPESTATUS[0]}
        echo $last_status
        status_check $last_status "${file}" "${status_log}"
    fi
done
} 

run_all_config "ppcls/configs/ImageNet/"



    


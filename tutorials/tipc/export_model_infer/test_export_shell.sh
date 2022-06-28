source test_tipc/common_func.sh
source test_tipc/utils_func.sh

echo "-------- Test inference result --------- "

FILENAME=$1
dataline=$(awk 'NR==52, NR==53{print}'  $FILENAME)
IFS=$'\n'
lines=(${dataline})
input_shape=$(func_parser_value "${lines[1]}")

function add_check_function(){
    jit_replace='import numpy as np 
import paddle 

def getdtype(dtype="float32"):
    if dtype == "float32" or dtype == "float":
        return np.float32
    if dtype == "float64":
        return np.float64
    if dtype == "int32":
        return np.int32
    if dtype == "int64":
        return np.int64

def randtool(dtype, low, high, shape):
    """
    np random tools
    """
    data = None
    if dtype.count("int"):
        data = np.random.randint(low, high, shape)
    elif dtype.count("float"):
        data = low + (high - low) * np.random.random(shape)
    elif dtype.count("bool"):
        data = np.random.randint(low, high, shape)
    return data.astype(getdtype(dtype))


def get_input_shape(data):
    config = {}
    data = data.split(";")
    shape_dict = {}
    for item in data:
        shape_list = item.strip("][").strip("{}").split("},{")
        for i, shape in enumerate(shape_list):
            if str(i) not in shape_dict:
                shape_dict[str(i)] = {}
                shape_dict[str(i)]["dtype"] = []
                shape_dict[str(i)]["shape"] = []
            arr = shape.strip("][").split(",[")
            dtype, shape = arr[0], list(map(int, arr[1].split(",")))
            shape.insert(0, -1)
            shape_dict[str(i)]["dtype"].append(dtype)
            shape_dict[str(i)]["shape"].append(shape)
    config["input_shape"] = shape_dict

    input_data = []
    for i, val in enumerate(config["input_shape"]):
        input_shape = config["input_shape"][val]
        shape = [1] + input_shape["shape"][0][1:]
        dtype = input_shape["dtype"][0]
        data = randtool(dtype, -1, 1, shape)
        input_data.append(data)
    return input_data

def verify_paddle_inference_correctness(layer, path):
    from paddle import inference
    import numpy as np

    model_file_path = path + ".pdmodel"
    params_file_path = path + ".pdiparams"
    config = inference.Config(model_file_path, params_file_path)
    predictor = inference.create_predictor(config)
    input_names = predictor.get_input_names()
    input_data = get_input_shape("xxxxxx")
    dygraph_input = {}
    if input_names == ["im_shape", "image", "scale_factor"]:
        input_names = ["image", "im_shape", "scale_factor"]
    for i,name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        fake_input = input_data[i]
        input_tensor.copy_from_cpu(fake_input)
        dygraph_input[name] = paddle.to_tensor(fake_input)
    predictor.run()
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_handle(output_name)
        output_tensors.append(output_tensor)
    prob_out = output_tensors[0].copy_to_cpu()

    layer.eval()
    pred = layer(dygraph_input)
    pred = list(pred.values())[0] if isinstance(pred, dict) else pred
    correct = np.allclose(pred, prob_out, rtol=1e-4, atol=1e-4)
    absolute_diff = np.abs(pred.numpy() - prob_out)
    max_absolute_diff = np.max(absolute_diff)
    # print("max_absolute_diff:", max_absolute_diff)
    assert correct, "Result diff when load and inference:\nlayer max_absolute_diff:{}"\
                  .format(max_absolute_diff)
    print("Successful, dygraph and inference predictions are consistent.")'

    echo "${jit_replace//xxxxxx/$input_shape}" > check_inference.py
}

add_check_function $FILENAME

function fun_run_check(){
    tab_num=$1
    dy_model=$2
    model_path=$3

    space=`expr $tab_num \* 2`
    tab=$(printf "%-${space}s" " ")
    line0="${tab}from check_inference import verify_paddle_inference_correctness"
    line1="${tab}layer = ${dy_model}"
    line2="${tab}path = ${model_path}"
    line4="${tab}verify_paddle_inference_correctness(layer, path)"
    echo "${line0}
${line1}
${line2}
${line4}" > tmp_file.txt
}

root_path=.
model_type=$PWD

echo $model_type

echo "-------- Add check infer code in jit.save ------"

if [[ $model_type =~ "PaddleClas" ]]; then
    echo "PaddleClas"
    # get export file
    export_file=${root_path}/ppcls/engine/engine.py
    # define layer path and img_shape
    layer="model"
    path="save_path"
elif [[ $model_type =~ "PaddleOCR" ]]; then
    echo "PaddleOCR"
    # get export file
    export_file=${root_path}/tools/export_model.py
    # define layer path and img_shape
    layer="model"
    path="save_path"
elif [[ $model_type =~ "PaddleDetection" ]]; then
    echo "PaddleDetection"
    # get export file
    export_file=${root_path}/ppdet/engine/trainer.py
    # define layer path and img_shape
    layer="static_model"
    path="os.path.join(save_dir, 'model')"
elif [[ $model_type =~ "PaddleGAN" ]]; then
    echo "PaddleGAN"
    # get export file
    export_file=${root_path}/ppgan/models/base_model.py
    # define layer path and img_shape
    layer="static_model"
    path="os.path.join(output_dir, model_name)"
elif [[ $model_type =~ "PaddleSeg" ]]; then
    echo "PaddleSeg"
    # get export file
    export_file=${root_path}/export.py
    # define layer path and img_shape
    layer="new_net"
    path="save_path"
elif [[ $model_type =~ "PaddleVideo" ]]; then
    echo "PaddleVideo"
    # get export file
    export_file=${root_path}/export.py
    # define layer path and img_shape
    layer="model"
    path="args.output_path"
fi

echo $export_file

if [[ $model_type =~ "PaddleDetection" ]]; then
    tab_num="6"
    # get insert line
    line_number="782"
elif [[ $model_type =~ "PaddleGAN" ]]; then
    tab_num="3"
    line_number="205"
else 
    tab_num=`cat ${export_file} | grep "jit.save" | awk -F"  " '{print NF-1}'`
    # get insert line
    line_number=`cat ${export_file} | grep -n "jit.save" | awk -F ":" '{print $1}'`
fi

fun_run_check $tab_num $layer $path

if [ `grep -c "check_inference" $export_file` -ne '0' ];then
    echo "The export file has add check code!"
else
    sed -i "${line_number} r tmp_file.txt" ${export_file}
fi 

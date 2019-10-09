## Quantization-aware training for SSD

### Introduction

The quantization-aware training used in this experiments is introduced in [fixed-point quantization desigin](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/design/quantization/fixed_point_quantization.md). Since quantization-aware training is still an active area of research and experimentation,
here, we just give an simple quantization training usage in Fluid based on MobileNet-SSD model, and more other exeperiments are still needed, like how to quantization traning by considering fusing batch normalization and convolution/fully-connected layers, channel-wise quantization of weights and so on.


A Python transpiler is used to rewrite Fluid training program or evaluation program for quantization-aware training:

```python

    #startup_prog = fluid.Program()
    #train_prog = fluid.Program()
    #loss = build_program(
    #    main_prog=train_prog,
    #    startup_prog=startup_prog,
    #    is_train=True)
    #build_program(
    #    main_prog=test_prog,
    #    startup_prog=startup_prog,
    #    is_train=False)
    #test_prog = test_prog.clone(for_test=True)
    # above is an pseudo code

    transpiler = fluid.contrib.QuantizeTranspiler(
        weight_bits=8,
        activation_bits=8,
        activation_quantize_type='abs_max', # or 'range_abs_max'
        weight_quantize_type='abs_max')
    # note, transpiler.training_transpile will rewrite train_prog
    # startup_prog is needed since it needs to insert and initialize
    # some state variable
    transpiler.training_transpile(train_prog, startup_prog)
    transpiler.training_transpile(test_prog, startup_prog)
```

  According to above design, this transpiler inserts fake quantization and de-quantization operation for each convolution operation (including depthwise convolution operation) and fully-connected operation. These quantizations take affect on weights and activations.

  In the design, we introduce dynamic quantization and static quantization strategies for different activation quantization methods. In the expriments, when set `activation_quantize_type` to `abs_max`, it is dynamic quantization. That is to say, the quantization scale (maximum of absolute value) of activation will be calculated each mini-batch during inference. When set `activation_quantize_type` to `range_abs_max`, a quantization scale for inference period will be calculated during training. Following part will introduce how to train.

### Quantization-aware training

  The training is fine-tuned on the well-trained MobileNet-SSD model. So download model at first:

  ```
  wget http://paddlemodels.bj.bcebos.com/ssd_mobilenet_v1_pascalvoc.tar.gz
  ```

- dynamic quantization:

  ```python
  python main_quant.py \
      --data_dir=$PascalVOC_DIR$ \
      --mode='train' \
      --init_model=ssd_mobilenet_v1_pascalvoc \
      --act_quant_type='abs_max' \
      --epoc_num=20 \
      --learning_rate=0.0001 \
      --batch_size=64 \
      --model_save_dir=$OUTPUT_DIR$
  ```
  Since fine-tuned on a well-trained model, we use a small start learnng rate 0.0001, and train 20 epocs.

- static quantization:
  ```python
  python main_quant.py \
      --data_dir=$PascalVOC_DIR$ \
      --mode='train' \
      --init_model=ssd_mobilenet_v1_pascalvoc \
      --act_quant_type='range_abs_max' \
      --epoc_num=80 \
      --learning_rate=0.001 \
      --lr_epochs=30,60 \
      --lr_decay_rates=1,0.1,0.01 \
      --batch_size=64 \
      --model_save_dir=$OUTPUT_DIR$
  ```
  Here, train 80 epocs, learning rate decays at 30 and 60 epocs by 0.1 every time. Users can adjust these hype-parameters.

### Convert to inference model

  As described in the design documentation, the inference graph is a little different from training, the difference is the de-quantization operation is before or after conv/fc. This is equivalent in training due to linear operation of conv/fc and de-quantization and functions' commutative law. But for inference, it needs to convert the graph, `fluid.contrib.QuantizeTranspiler.freeze_program` is used to do this:

  ```python
  #startup_prog = fluid.Program()
  #test_prog = fluid.Program()
  #test_py_reader, map_eval, nmsed_out, image = build_program(
  #    main_prog=test_prog,
  #    startup_prog=startup_prog,
  #    train_params=configs,
  #    is_train=False)
  #test_prog = test_prog.clone(for_test=True)
  #transpiler = fluid.contrib.QuantizeTranspiler(weight_bits=8,
  #    activation_bits=8,
  #    activation_quantize_type=act_quant_type,
  #    weight_quantize_type='abs_max')
  #transpiler.training_transpile(test_prog, startup_prog)
  #place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
  #exe = fluid.Executor(place)
  #exe.run(startup_prog)

  def if_exist(var):
      return os.path.exists(os.path.join(init_model, var.name))
  fluid.io.load_vars(exe, init_model, main_program=test_prog,
                     predicate=if_exist)
  # freeze the rewrited training program
  # freeze after load parameters, it will quantized weights
  transpiler.freeze_program(test_prog, place)
  ```

  Users can evaluate the converted model by:

  ```
  python main_quant.py \
      --data_dir=$PascalVOC_DIR$ \
      --mode='test' \
      --init_model=$MODLE_DIR$ \
      --model_save_dir=$MobileNet_SSD_8BIT_MODEL$
  ```

  You also can check the 8-bit model by the inference scripts

  ```
  python main_quant.py \
      --mode='infer' \
      --init_model=$MobileNet_SSD_8BIT_MODEL$ \
      --confs_threshold=0.5 \
      --image_path='/data/PascalVOC/VOCdevkit/VOC2007/JPEGImages/002271.jpg'
  ```
  See 002271.jpg for the visualized image with bbouding boxes.


  **Note**, if you want to convert model to 8-bit, you should call `fluid.contrib.QuantizeTranspiler.convert_to_int8` to do this. But, now Paddle can't load 8-bit model to do inference.

### Results

Results of MobileNet-v1-SSD 300x300 model on PascalVOC dataset.

| Model                                   | mAP                |
|:---------------------------------------:|:------------------:|
|Floating point: 32bit                    | 73.32%             |
|Fixed point: 8bit, dynamic quantization  | 72.77%             |
|Fixed point: 8bit, static quantization   | 72.45%             |

 As mentioned above, other experiments, like how to quantization traning by considering fusing batch normalization and convolution/fully-connected layers, channel-wise quantization of weights, quantizated weights type with uint8 instead of int8 and so on.

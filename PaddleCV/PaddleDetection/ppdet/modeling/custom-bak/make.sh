include_dir=$( python -c 'import paddle; print(paddle.sysconfig.get_include())' )
lib_dir=$( python -c 'import paddle; print(paddle.sysconfig.get_lib())' )

echo $include_dir
echo $lib_dir


#nvcc relu_op.cu -o relu_op.cu.o -ccbin cc -DPADDLE_WITH_MKLDNN -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO -Xcompiler -fPIC -std=c++11 -Xcompiler -fPIC -w --expt-relaxed-constexpr -O3 -DNVCC \
#  -I ${include_dir} \
#  -I ${include_dir}/third_party \
#  -I /usr/local/cuda/include \
#  -I /custom/cjt/custom-bak/custom/mkldnn/include/ \
#  -L ${lib_dir} -lpaddle_framework -lcudart 
  #-L /usr/lib64 -lstdc++ \
  #-L /lib64 -lm
  #-L /usr/local/cuda/lib64 \
  #-L ${lib_dir} -lpaddle_framework #-lcudart
  #-L /custom/cjt/custom-bak/custom/mkldnn/lib64 -lmkldnn


nvcc rrpn_generate_proposals_op.cu -c -o rrpn_generate_proposals_op.cu.o -ccbin cc -DPADDLE_WITH_MKLDNN -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO -Xcompiler -fPIC -std=c++11 -Xcompiler -fPIC -w --expt-relaxed-constexpr -O3 -DNVCC \
    -I ${include_dir} \
    -I ${include_dir}/third_party \
    -I /usr/local/cuda/include \
    -I /paddle/custom/mkldnn/include \
    -L ${lib_dir} -lpaddle_framework \
    -L /usr/local/cuda/targets/x86_64-linux/lib/ -lcudart


nvcc rotated_anchor_generator_op.cu -c -o rotated_anchor_generator_op.cu.o -ccbin cc -DPADDLE_WITH_MKLDNN -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO -Xcompiler -fPIC -std=c++11 -Xcompiler -fPIC -w --expt-relaxed-constexpr -O3 -DNVCC \
    -I ${include_dir} \
    -I ${include_dir}/third_party \
    -I /usr/local/cuda/include \
    -I /paddle/custom/mkldnn/include \
    -L ${lib_dir} -lpaddle_framework \
    -L /usr/local/cuda/targets/x86_64-linux/lib/ -lcudart

nvcc rrpn_box_coder_op.cu -c -o rrpn_box_coder_op.cu.o -ccbin cc -DPADDLE_WITH_MKLDNN -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO -Xcompiler -fPIC -std=c++11 -Xcompiler -fPIC -w --expt-relaxed-constexpr -O3 -DNVCC \
    -I ${include_dir} \
    -I ${include_dir}/third_party \
    -I /usr/local/cuda/include \
    -I /paddle/custom/mkldnn/include \
    -L ${lib_dir} -lpaddle_framework \
    -L /usr/local/cuda/targets/x86_64-linux/lib/ -lcudart

nvcc rrpn_rotated_roi_align_op.cu -c -o rrpn_rotated_roi_align_op.cu.o -ccbin cc -DPADDLE_WITH_MKLDNN -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO -Xcompiler -fPIC -std=c++11 -Xcompiler -fPIC -w --expt-relaxed-constexpr -O3 -DNVCC \
    -I ${include_dir} \
    -I ${include_dir}/third_party \
    -I /usr/local/cuda/include \
    -I /paddle/custom/mkldnn/include \
    -L ${lib_dir} -lpaddle_framework \
    -L /usr/local/cuda/targets/x86_64-linux/lib/ -lcudart


g++ rotated_anchor_generator_op.cc concat_and_split.cc rrpn_generate_proposal_labels_op.cc rrpn_generate_proposals_op.cc rrpn_target_assign_op.cc rrpn_box_coder_op.cc rrpn_rotated_roi_align_op.cc rrpn_rotated_roi_align_op.cu.o rrpn_box_coder_op.cu.o rotated_anchor_generator_op.cu.o rrpn_generate_proposals_op.cu.o -o lib.so -shared -fPIC -std=c++11 -O3 -DPADDLE_WITH_MKLDNN -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO \
  -I ${include_dir} \
  -I ${include_dir}/third_party \
  -I /usr/local/cuda/include \
  -I /paddle/custom/mkldnn/include \
  -L ${lib_dir} -lpaddle_framework \
  -L /usr/local/cuda/targets/x86_64-linux/lib/ -lcudart 
 #-L ${lib_dir} -lpaddle_framework #-lcuda

#g++ rrpn_box_coder_op.cc rrpn_box_coder_op.cu.o -o rrpn_box_coder_op.so -shared -fPIC -std=c++11 -O3 -DPADDLE_WITH_MKLDNN -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO \
#  -I ${include_dir} \
#  -I ${include_dir}/third_party \
#  -I /usr/local/cuda-9.0/include \
#  -I /custom/cjt/custom-bak/custom/mkldnn/include/ \
#  -L ${lib_dir} -lpaddle_framework \
#  -L /usr/local/cuda-9.0/targets/x86_64-linux/lib/ -lcudart
  #-L ${lib_dir} -lpaddle_framework #-lcuda


#g++ concat_and_split.cc rrpn_generate_proposal_labels_op.cc -o rotated_anchor_generator_op.so -shared -fPIC -std=c++11 -O3 -DPADDLE_WITH_MKLDNN -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO \
#  -I ${include_dir} \
#  -I ${include_dir}/third_party \
#  -I /usr/local/cuda-9.0/include \
#  -I /custom/cjt/custom-bak/custom/mkldnn/include/ \
#  -L ${lib_dir} -lpaddle_framework \
#  -L /usr/local/cuda-9.0/targets/x86_64-linux/lib/ -lcudart 
#  #-L ${lib_dir} -lpaddle_framework #-lcuda


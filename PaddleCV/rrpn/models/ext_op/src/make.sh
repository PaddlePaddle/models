include_dir=$( python -c 'import paddle; print(paddle.sysconfig.get_include())' )
lib_dir=$( python -c 'import paddle; print(paddle.sysconfig.get_lib())' )

echo $include_dir
echo $lib_dir

CUDA=$1
CUDNN=$2
NCCL=$3

if [ ! -d "$CUDA" ]; then
echo "Usage: sh make.sh \$CUDA_PATH \$CUDNN_PATH \$NCCL_PATH"
exit
fi

if [ ! -d "$CUDNN" ]; then
echo "Usage: sh make.sh \${CUDA_PATH} \${CUDNN_PATH} \${NCCL_PATH}"
exit
fi

if [ ! -d "$NCCL" ]; then
echo "Usage: sh make.sh \${CUDA_PATH} \${CUDNN_PATH} \${NCCL_PATH}"
exit
fi

git clone https://github.com/NVlabs/cub.git

nvcc rrpn_generate_proposals_op.cu -c -o rrpn_generate_proposals_op.cu.o -ccbin cc -DPADDLE_WITH_MKLDNN -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO -Xcompiler -fPIC -std=c++11 -Xcompiler -fPIC -w --expt-relaxed-constexpr -O3 -DNVCC \
    -I ${include_dir} \
   -I ${include_dir}/third_party \
    -I ${CUDA}/include \
    -I ${CUDNN}/include \
    -I ${NCCL}/include \
    -L ${lib_dir} -lpaddle_framework \
    -L ${CUDA}/lib64 -lcudart


nvcc rotated_anchor_generator_op.cu -c -o rotated_anchor_generator_op.cu.o -ccbin cc -DPADDLE_WITH_MKLDNN -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO -Xcompiler -fPIC -std=c++11 -Xcompiler -fPIC -w --expt-relaxed-constexpr -O3 -DNVCC \
    -I ${include_dir} \
    -I ${include_dir}/third_party \
    -I ${CUDA}/include \
    -I ${CUDNN}/include \
    -I ${NCCL}/include \
    -L ${lib_dir} -lpaddle_framework \
    -L ${CUDA}/lib64 -lcudart

nvcc rrpn_box_coder_op.cu -c -o rrpn_box_coder_op.cu.o -ccbin cc -DPADDLE_WITH_MKLDNN -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO -Xcompiler -fPIC -std=c++11 -Xcompiler -fPIC -w --expt-relaxed-constexpr -O3 -DNVCC \
    -I ${include_dir} \
    -I ${include_dir}/third_party \
    -I ${CUDA}/include \
    -I ${CUDNN}/include \
    -I ${NCCL}/include \
    -L ${lib_dir} -lpaddle_framework \
    -L ${CUDA}/lib64 -lcudart

nvcc rrpn_rotated_roi_align_op.cu -c -o rrpn_rotated_roi_align_op.cu.o -ccbin cc -DPADDLE_WITH_MKLDNN -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO -Xcompiler -fPIC -std=c++11 -Xcompiler -fPIC -w --expt-relaxed-constexpr -O3 -DNVCC \
    -I ${include_dir} \
    -I ${include_dir}/third_party \
    -I ${CUDA}/include \
    -I ${CUDNN}/include \
    -I ${NCCL}/include \
    -L ${lib_dir} -lpaddle_framework \
    -L ${CUDA}/lib64 -lcudart


g++ rotated_anchor_generator_op.cc concat_and_split.cc rrpn_generate_proposal_labels_op.cc rrpn_generate_proposals_op.cc rrpn_target_assign_op.cc rrpn_box_coder_op.cc rrpn_rotated_roi_align_op.cc rrpn_rotated_roi_align_op.cu.o rrpn_box_coder_op.cu.o rotated_anchor_generator_op.cu.o rrpn_generate_proposals_op.cu.o -o rrpn_lib.so -shared -fPIC -std=c++11 -O3 -DPADDLE_WITH_MKLDNN -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO \
  -I ${include_dir} \
  -I ${include_dir}/third_party \
  -I ${CUDA}/include \
  -I ${CUDNN}/include \
  -I ${NCCL}/include \
  -L ${lib_dir} -lpaddle_framework \
  -L ${CUDA}/lib64 -lcudart 

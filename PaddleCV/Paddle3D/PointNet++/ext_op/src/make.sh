include_dir=$( python -c 'import paddle; print(paddle.sysconfig.get_include())' )
lib_dir=$( python -c 'import paddle; print(paddle.sysconfig.get_lib())' )

echo $include_dir
echo $lib_dir

OPS='farthest_point_sampling_op gather_point_op group_points_op query_ball_op three_interp_op three_nn_op'
for op in ${OPS}
do
nvcc ${op}.cu -c -o ${op}.cu.o -ccbin cc -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO -DPADDLE_WITH_MKLDNN -Xcompiler -fPIC -std=c++11 -Xcompiler -fPIC -w --expt-relaxed-constexpr -O0 -g -DNVCC \
    -I ${include_dir}/third_party/ \
    -I ${include_dir}
done

g++ farthest_point_sampling_op.cc farthest_point_sampling_op.cu.o gather_point_op.cc gather_point_op.cu.o group_points_op.cc group_points_op.cu.o query_ball_op.cu.o query_ball_op.cc three_interp_op.cu.o three_interp_op.cc three_nn_op.cu.o three_nn_op.cc -o pointnet_lib.so -DPADDLE_WITH_MKLDNN -shared -fPIC -std=c++11 -O0 -g \
  -I ${include_dir}/third_party/ \
  -I ${include_dir} \
  -L ${lib_dir} \
  -L /usr/local/cuda/lib64 -lpaddle_framework -lcudart

rm *.cu.o

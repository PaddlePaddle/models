include_dir=$( python3.7 -c 'import paddle; print(paddle.sysconfig.get_include())' )
lib_dir=$( python3.7 -c 'import paddle; print(paddle.sysconfig.get_lib())' )

echo $include_dir
echo $lib_dir

OPS='correlation_op'
for op in ${OPS}
do
nvcc ${op}.cu -c -o ${op}.cu.o -ccbin cc -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO -DPADDLE_WITH_MKLDNN -Xcompiler -fPIC -std=c++11 -Xcompiler -fPIC -w --expt-relaxed-constexpr -O0 -g -DNVCC \
    -I ${include_dir}/third_party/ \
    -I ${include_dir}
done

##g++-4.8 correlation_op.cu.o correlation_op.cc -o correlation_lib.so -DPADDLE_WITH_MKLDNN -shared -fPIC -std=c++11 -O0 -g \
g++ correlation_op.cu.o correlation_op.cc -o correlation_lib.so -DPADDLE_WITH_MKLDNN -shared -fPIC -std=c++11 -O0 -g \
  -I ${include_dir}/third_party/ \
  -I ${include_dir} \
  -L ${lib_dir} \
  -L /usr/local/cuda/lib64 -lpaddle_framework -lcudart

rm *.cu.o

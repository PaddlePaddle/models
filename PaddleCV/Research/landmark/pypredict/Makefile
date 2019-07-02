.PHONY:clean

LDFLAGS = -fPIC -shared -Wl,--rpath=\$ORIGIN -Wl,--rpath=\$ORIGIN/../so -Wl,-z,origin

LIBS = ./so/libpaddle_fluid.so ./so/libmkldnn.so.0 ./so/libmklml_intel.so ./so/libiomp5.so -L $(PYTHONHOME)/lib -lpython2.7

CFLAGS= -fPIC -I pybind11/include -I ./fluid_inference/third_party/install/glog/include -I./fluid_inference/third_party/install/gflags/include -I ./fluid_inference/paddle/include/ -I. -I./fluid_inference/third_party/boost/ -I$(PYTHONHOME)/include/python2.7 -std=c++11  
OBJS = predictor.o\
       py_cnnpredict.o\
       conf_parser.o
       
PyCNNPredict.so: $(OBJS) 
	g++ $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS) 
%.o : %.cpp
	g++ $(CFLAGS) -c $^ -o $@

clean:
	rm -rf $(OBJS) PyCNNPredict.so
	
include_directories(${PY_INCLUDE_DIR})
include_directories(${PY_INCLUDE_DIR}\third_party)

if(EXISTS ${PY_LIB_DIR}/libpaddle_framework.so)
  set(lib_link
    -lpaddle_framework
  )
endif()
if(EXISTS ${FT_LIB_DIR}/libdecoding.a)
  list(APPEND lib_link -ldecoding)
endif()
if(EXISTS ${FT_LIB_DIR}/libdecoder.a)
  list(APPEND lib_link -ldecoder)
endif()

add_definitions(-DPADDLE_WITH_CUDA)
add_definitions(-DEIGEN_USE_GPU)
add_definitions(-DPADDLE_USE_DSO)
add_definitions(-DPADDLE_WITH_MKLDNN)

set(decoding_op_files fusion_decoding_op.cc fusion_decoding_op.cu)
add_library(decoding_op SHARED ${decoding_op_files})
target_link_libraries(decoding_op PRIVATE -lcublas -lcudart ${lib_link})

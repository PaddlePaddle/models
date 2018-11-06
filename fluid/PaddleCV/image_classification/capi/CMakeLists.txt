# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

cmake_minimum_required(VERSION 3.0)
project(image_classification CXX C)

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
        "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")


if(CMAKE_COMPILER_IS_GNUCXX)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    if (GCC_VERSION VERSION_LESS 4.8.2)
        message(FATAL_ERROR "A gcc compiler with a version >= 4.8.2 is needed.")
    endif()
else()
    message(FATAL_ERROR "A gcc compiler is needed.")
endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_CXX_FLAGS "-O4 -g -pipe -W -Wall -Wno-unused-parameter -fPIC -fpermissive -std=gnu++11")

##############################################################################

if (IS_ABSOLUTE ${PADDLE_ROOT})
    set(PADDLE_ABS_PATH ${PADDLE_ROOT})
else ()
    get_filename_component(PADDLE_ABS_PATH ${CMAKE_BINARY_DIR}/${PADDLE_ROOT} ABSOLUTE)
endif ()

if (EXISTS ${PADDLE_ABS_PATH})
    message(STATUS "\$PADDLE_ROOT is set to '${PADDLE_ABS_PATH}'")
    include_directories(${PADDLE_ABS_PATH})
    include_directories(${PADDLE_ABS_PATH}/third_party/install/gflags/include)
    include_directories(${PADDLE_ABS_PATH}/third_party/boost)
    include_directories(${PADDLE_ABS_PATH}/third_party/install/glog/include)
    include_directories(${PADDLE_ABS_PATH}/third_party/install/protobuf/include)
    include_directories(${PADDLE_ABS_PATH}/third_party/eigen3)

    link_directories(${PADDLE_ABS_PATH}/third_party/install/gflags/lib)
    link_directories(${PADDLE_ABS_PATH}/third_party/install/glog/lib)
else ()
    message(FATAL_ERROR "'${PADDLE_ABS_PATH}' does not exist.\nPlease set -DPADDLE_ROOT=/path/to/Paddle/build/fluid_install_dir/")
endif ()

set (fluid_lib_path ${PADDLE_ABS_PATH}/paddle/fluid/inference)
if (EXISTS ${fluid_lib_path})
    link_directories(${PADDLE_ABS_PATH}/paddle/fluid/inference)
    link_libraries(paddle_fluid)
else ()
    message(FATAL_ERROR "Fluid inference lib not found.\nPlease set -DPADDLE_ROOT=/path/to/Paddle/build/fluid_install_dir")
endif ()

# add mkldnn library if it exitsts
set (mkldnn_lib_path ${PADDLE_ABS_PATH}/third_party/install/mkldnn/lib)
if (EXISTS ${mkldnn_lib_path})
    message(STATUS "MKLDNN lib found.")
else ()
    message(STATUS "MKLDNN lib not found.")
endif()

# add mklml library if it exitsts
set (mklml_inc_path ${PADDLE_ABS_PATH}/third_party/install/mklml/include)
set (mklml_lib_path ${PADDLE_ABS_PATH}/third_party/install/mklml/lib)
if (EXISTS ${mklml_inc_path} AND EXISTS ${mklml_lib_path})
    message(STATUS "MKLML lib found.")
    include_directories(${PADDLE_ABS_PATH}/third_party/install/mklml/include)
    link_directories(${PADDLE_ABS_PATH}/third_party/install/mklml/lib)
    link_libraries(mklml_intel iomp5)
else ()
    message(STATUS "MKLML lib not found.")
endif()

link_libraries(rt dl pthread)

##############################################################################

# OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")

##############################################################################

set(TARGET_NAME infer_image_classification)
add_executable(${TARGET_NAME} infer_image_classification.cc data_reader.cc stats.cc)
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS} gflags glog ${mkldnn_lib_path}/libmkldnn.so.0)

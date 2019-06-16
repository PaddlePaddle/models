/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

enum class DataType : int {
  INT8 = 0,
  INT32 = 2,
  INT64 = 3,
  FLOAT32 = 4,
};

inline size_t get_type_size(DataType type) {
  switch (type) {
    case DataType::INT8:
      return sizeof(int8_t);
    case DataType::INT32:
      return sizeof(int32_t);
    case DataType::INT64:
      return sizeof(int64_t);
    case DataType::FLOAT32:
      return sizeof(float);
    default:
      return 0;
  }
}

struct DataBuf {
  std::size_t size;
  DataType type;
  std::shared_ptr<char> data;

  DataBuf() = default;

  DataBuf(DataType dtype, size_t dsize) { alloc(dtype, dsize); }

  DataBuf(const void *ddata, DataType dtype, size_t dsize) {
    alloc(dtype, dsize);
    copy(ddata, dsize);
  }

  DataBuf(const DataBuf &dbuf)
      : size(dbuf.size), type(dbuf.type), data(dbuf.data) {}

  DataBuf &operator=(const DataBuf &dbuf) {
    size = dbuf.size;
    type = dbuf.type;
    data = dbuf.data;
    return *this;
  }

  void reset(const void *ddata, size_t dsize) {
    clear();
    alloc(type, dsize);
    copy(ddata, dsize);
  }

  void clear() {
    size = 0;
    data.reset();
  }

  ~DataBuf() { clear(); }

 private:
  void alloc(DataType dtype, size_t dsize) {
    type = dtype;
    size = dsize;
    data.reset(new char[dsize * get_type_size(dtype)],
               std::default_delete<char[]>());
  }

  void copy(const void *ddata, size_t dsize) {
    const char *temp = reinterpret_cast<const char *>(ddata);
    std::copy(temp, temp + dsize * get_type_size(type), data.get());
  }
};

struct Tensor {
  std::string name;
  std::vector<int> shape;
  std::vector<std::vector<size_t>> lod;
  DataBuf data;
};

class ICNNPredict {
 public:
  ICNNPredict() {}
  virtual ~ICNNPredict() {}

  virtual ICNNPredict *clone() = 0;

  virtual bool predict(const std::vector<Tensor> &inputs,
                       const std::vector<std::string> &layers,
                       std::vector<Tensor> &outputs) = 0;

  virtual bool predict(const std::vector<std::vector<float>> &input_datas,
                       const std::vector<std::vector<int>> &input_shapes,
                       const std::vector<std::string> &layers,
                       std::vector<std::vector<float>> &output_datas,
                       std::vector<std::vector<int>> &output_shapes) = 0;

  virtual void destroy(std::vector<Tensor> &tensors) {
    std::vector<Tensor>().swap(tensors);
  }

  virtual void destroy(std::vector<std::vector<float>> &datas) {
    std::vector<std::vector<float>>().swap(datas);
  }

  virtual void destroy(std::vector<std::vector<int>> &shapes) {
    std::vector<std::vector<int>>().swap(shapes);
  }
};

ICNNPredict *create_cnnpredict(const std::string &conf_file,
                               const std::string &prefix);

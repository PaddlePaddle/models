#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <string>
#include <vector>

namespace py = pybind11;
using std::string;
using std::vector;

class PyCNNPredict {
 public:
  PyCNNPredict() : _predictor(NULL) {}

  ~PyCNNPredict();

  bool init(string conf_file, string prefix);

  py::list predict(py::list input_datas,
                   py::list input_shapes,
                   py::list layer_names);

 private:
  ICNNPredict *_predictor;
  py::list postprocess(const vector<vector<float>> &vdatas,
                       const vector<vector<int>> &vshapes);
};

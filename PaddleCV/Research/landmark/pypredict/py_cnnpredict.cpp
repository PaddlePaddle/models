#include "logger.h"

#include "cnnpredict_interface.h"
#include "common.h"
#include "py_cnnpredict.h"

template <class T>
vector<T> ndarray_to_vector(const py::array &nd) {
  py::dtype datatype = nd.dtype();
  size_t nd_dim = nd.ndim();
  const auto *shape = nd.shape();
  size_t data_num = nd.size();
  // py::buffer_info buf = nd.request();

  const T *nd_data = reinterpret_cast<const T *>(nd.data(0));
  vector<T> vec(data_num, 0);
  std::copy(nd_data, nd_data + data_num, vec.begin());
  return vec;
}

template <class T>
vector<T> list_to_vector(py::list &list) {
  vector<T> vec;
  for (size_t i = 0; i < py::len(list); i++) {
    T l = py::cast<T>(list[i]);
    vec.push_back(l);
  }

  return vec;
}

template <class T>
vector<vector<T>> ndlist_to_vectors(py::list &ndlist) {
  vector<vector<T>> vecs;
  for (unsigned int i = 0; i < py::len(ndlist); i++) {
    py::array nd = py::array(ndlist[i]);
    vector<T> vec = ndarray_to_vector<T>(nd);
    vecs.push_back(vec);
  }
  return vecs;
}

template <class T>
py::array vector_to_ndarray(const vector<T> &vec) {
  const std::vector<size_t> shape = {vec.size()};
  auto format = py::format_descriptor<T>::format();
  py::dtype dt(format);
  py::array nd(dt, shape, (const char *)vec.data());
  return nd;
}

template <class T>
py::list vectors_to_list(const vector<vector<T>> &vecs) {
  py::list ndlist;
  for (int i = 0; i < vecs.size(); i++) {
    py::array nd = vector_to_ndarray<T>(vecs[i]);
    ndlist.append(nd);
  }
  return ndlist;
}

PyCNNPredict::~PyCNNPredict() {
  if (_predictor != NULL) {
    delete _predictor;
    _predictor = NULL;
  }
}

bool PyCNNPredict::init(string conf_file, string prefix) {
  LOG(INFO) << "PyCNNPredict::init()";
  _predictor = create_cnnpredict(conf_file, prefix);
  if (_predictor == NULL) {
    LOG(FATAL) << "fail to call create_cnnpredict";
    return false;
  }
  return true;
}

py::list PyCNNPredict::postprocess(const vector<vector<float>> &vdatas,
                                   const vector<vector<int>> &vshapes) {
  LOG(INFO) << "PyCNNPredict::postprocess()";

  py::list result;
  if (vdatas.size() != vshapes.size()) {
    LOG(FATAL) << "datas and shapes size not equal";
    return result;
  }

  result.append(vectors_to_list(vdatas));
  result.append(vectors_to_list(vshapes));

  return result;
}

py::list PyCNNPredict::predict(py::list input_datas,
                               py::list input_shapes,
                               py::list layer_names) {
  LOG(INFO) << "PyCNNPredict::predict()";
  vector<vector<float>> inputdatas;
  vector<vector<int>> inputshapes;
  vector<string> layernames;
  vector<vector<float>> outputdatas;
  vector<vector<int>> outputshapes;

  py::list result;
  if (py::len(input_datas) != py::len(input_shapes)) {
    LOG(FATAL) << "datas and shapes size not equal";
    return result;
  }

  inputdatas = ndlist_to_vectors<float>(input_datas);
  inputshapes = ndlist_to_vectors<int>(input_shapes);
  layernames = list_to_vector<string>(layer_names);

  bool ret = _predictor->predict(
      inputdatas, inputshapes, layernames, outputdatas, outputshapes);
  if (!ret) {
    LOG(FATAL) << "fail to predict";
    return result;
  }

  return postprocess(outputdatas, outputshapes);
}

PYBIND11_MODULE(PyCNNPredict, m) {
  m.doc() = "pycnnpredict";
  py::class_<PyCNNPredict>(m, "PyCNNPredict")
      .def(py::init())
      .def("init", &PyCNNPredict::init)
      .def("predict", &PyCNNPredict::predict);
}

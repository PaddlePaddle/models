#include <algorithm>
#include <memory>
#include "logger.h"

#include "conf_parser.h"
#include "predictor.h"

Predictor::~Predictor() {}

bool feed(paddle::PaddlePredictor *predictor,
          const std::vector<Tensor> &tensors) {
  LOG(INFO) << "Predictor::feed";

  auto names = predictor->GetInputNames();
  if (names.size() != tensors.size()) {
    LOG(WARNING) << "The given size " << tensors.size()
                 << " is not equal to the required size " << names.size();
    return false;
  }

  for (size_t i = 0; i < names.size(); ++i) {
    auto i_t = predictor->GetInputTensor(names[i]);
    i_t->Reshape(tensors[i].shape);
    i_t->SetLoD(tensors[i].lod);

    if (tensors[i].data.type == DataType::FLOAT32) {
      const float *temp =
          reinterpret_cast<const float *>(tensors[i].data.data.get());
      i_t->copy_from_cpu(temp);
    } else if (tensors[i].data.type == DataType::INT32) {
      const int32_t *temp =
          reinterpret_cast<const int32_t *>(tensors[i].data.data.get());
      i_t->copy_from_cpu(temp);
    } else if (tensors[i].data.type == DataType::INT64) {
      const int64_t *temp =
          reinterpret_cast<const int64_t *>(tensors[i].data.data.get());
      i_t->copy_from_cpu(temp);
    } else {
      LOG(ERROR) << "do not support current datatype";
      return false;
    }
  }

  return true;
}

bool fetch(paddle::PaddlePredictor *predictor, std::vector<Tensor> &tensors) {
  LOG(INFO) << "Predictor::fetch";

  auto names = predictor->GetOutputNames();
  for (auto &name : names) {
    auto o_t = predictor->GetOutputTensor(name);
    std::vector<int> s = o_t->shape();

    Tensor out;
    out.shape = s;
    out.lod = o_t->lod();

    int num = std::accumulate(s.begin(), s.end(), 1, std::multiplies<int>());

    if (o_t->type() == paddle::PaddleDType::FLOAT32) {
      out.data = DataBuf(DataType::FLOAT32, size_t(num));
      float *p_data = reinterpret_cast<float *>(out.data.data.get());
      o_t->copy_to_cpu(p_data);
    } else if (o_t->type() == paddle::PaddleDType::INT32) {
      out.data = DataBuf(DataType::INT32, size_t(num));
      int32_t *p_data = reinterpret_cast<int32_t *>(out.data.data.get());
      o_t->copy_to_cpu(p_data);
    } else if (o_t->type() == paddle::PaddleDType::INT64) {
      out.data = DataBuf(DataType::INT64, size_t(num));
      int64_t *p_data = reinterpret_cast<int64_t *>(out.data.data.get());
      o_t->copy_to_cpu(p_data);
    } else {
      LOG(ERROR) << "do no support current datatype";
      return false;
    }

    tensors.push_back(out);
  }

  return true;
}

bool Predictor::predict(const std::vector<Tensor> &inputs,
                        const std::vector<std::string> &layers,
                        std::vector<Tensor> &outputs) {
  LOG(INFO) << "Predictor::predict";
  (void)layers;
  // 1. feed input
  if (!feed(_predictor.get(), inputs)) {
    return false;
  }

  // 2. execute inference
  if (!_predictor->ZeroCopyRun()) {
    LOG(WARNING) << "fail to execute predictor";
    return false;
  }

  // 3. fetch output
  if (!fetch(_predictor.get(), outputs)) {
    return false;
  }
  return true;
}

bool check_shape(const std::vector<std::vector<float>> &datas,
                 const std::vector<std::vector<int>> &shapes) {
  LOG(INFO) << "check_shape";
  if (datas.size() != shapes.size()) {
    LOG(ERROR) << "datas size: " << datas.size() << " != "
               << "shapes size(): " << shapes.size();
    return false;
  }
  for (size_t i = 0; i < datas.size(); ++i) {
    int count = 1;
    for (auto num : shapes[i]) {
      count *= num;
    }
    int data_size = static_cast<int>(datas[i].size());
    if (count != data_size) {
      LOG(ERROR) << "data[" << i << "] size " << data_size << " != "
                 << "shape [" << i << "] size " << count;
      return false;
    }
  }
  return true;
}

bool feed(paddle::PaddlePredictor *predictor,
          const std::vector<std::vector<float>> &datas,
          const std::vector<std::vector<int>> &shapes) {
  LOG(INFO) << "Predictor::feed";

  // 1. check input shape
  if (!check_shape(datas, shapes)) {
    return false;
  }

  // 2. check given input and required input
  auto names = predictor->GetInputNames();
  if (names.size() != datas.size()) {
    LOG(WARNING) << "The given size " << datas.size()
                 << " is not equal to the required size " << names.size();
    return false;
  }

  // 3. feed
  for (size_t i = 0; i < names.size(); ++i) {
    auto i_t = predictor->GetInputTensor(names[i]);
    i_t->Reshape(shapes[i]);
    i_t->copy_from_cpu(datas[i].data());
  }

  return true;
}

bool fetch(paddle::PaddlePredictor *predictor,
           std::vector<std::vector<float>> &datas,
           std::vector<std::vector<int>> &shapes) {
  LOG(INFO) << "Predictor::fetch";

  auto names = predictor->GetOutputNames();
  for (auto &name : names) {
    auto o_t = predictor->GetOutputTensor(name);
    std::vector<int> s = o_t->shape();
    shapes.push_back(s);

    int num = std::accumulate(s.begin(), s.end(), 1, std::multiplies<int>());

    std::vector<float> data(num);
    o_t->copy_to_cpu(data.data());
    datas.push_back(data);
  }

  return true;
}

bool Predictor::predict(const std::vector<std::vector<float>> &input_datas,
                        const std::vector<std::vector<int>> &input_shapes,
                        const std::vector<std::string> &layers,
                        std::vector<std::vector<float>> &output_datas,
                        std::vector<std::vector<int>> &output_shapes) {
  LOG(INFO) << "Predictor::predict";
  (void)layers;

  // 1. feed input
  if (!feed(_predictor.get(), input_datas, input_shapes)) {
    return false;
  }

  // 2. execute inference
  if (!_predictor->ZeroCopyRun()) {
    LOG(WARNING) << "fail to execute predictor";
    return false;
  }

  // 3. fetch output
  if (!fetch(_predictor.get(), output_datas, output_shapes)) {
    return false;
  }

  return true;
}

void init_tensorrt(const ConfParser *conf,
                   const std::string &prefix,
                   AnalysisConfig &config) {
  LOG(INFO) << "Predictor::init_tensorrt()";

  // 1. max_batch_size for tensorrt
  int max_batch_size = 1;
  if (!conf->get_int(prefix, "max_batch_size", max_batch_size)) {
    LOG(WARNING) << "fail to get max_batch_size from conf, set as 1";
  }
  max_batch_size = std::max(1, max_batch_size);

  // 2. workspace_size for tensorrt
  int workspace_size = 0;
  if (!conf->get_int(prefix, "workspace_size", workspace_size)) {
    LOG(WARNING) << "fail to get workspace_size from conf, set as 0";
  }
  workspace_size = std::max(0, workspace_size);

  // 3. min_subgraph_size for tensorrt
  int min_subgraph_size = 3;
  if (!conf->get_int(prefix, "min_subgraph_size", min_subgraph_size)) {
    LOG(WARNING) << "fail to get min_subgraph_size from conf, set as 3";
  }
  min_subgraph_size = std::max(0, min_subgraph_size);

  config.EnableTensorRtEngine(
      workspace_size, max_batch_size, min_subgraph_size);
}

void init_anakin(const ConfParser *conf,
                 const std::string &prefix,
                 AnalysisConfig &config) {
  LOG(INFO) << "Predictor::init_anakin()";

  // 1. max_batch_size for tensorrt
  int max_batch_size = 1;
  if (!conf->get_int(prefix, "max_batch_size", max_batch_size)) {
    LOG(WARNING) << "fail to get max_batch_size from conf, set as 1";
  }
  max_batch_size = std::max(1, max_batch_size);

  std::map<std::string, std::vector<int>> anakin_max_input_dict;
  std::vector<std::string> input_names;
  if (!conf->get_strings(prefix, "input_names", input_names)) {
    LOG(WARNING) << "fail to get input_names from conf";
  }
  for (auto &n : input_names) {
    std::vector<int> shape;
    if (!conf->get_ints(prefix, n, shape)) {
      LOG(WARNING) << "fail to get the shape of " + n;
    } else {
      anakin_max_input_dict[n] = shape;
    }
  }

  config.EnableAnakinEngine(max_batch_size, anakin_max_input_dict);
  config.pass_builder()->TurnOnDebug();
}

void init_gpu(const ConfParser *conf,
              const std::string &prefix,
              int device,
              AnalysisConfig &config) {
  LOG(INFO) << "Predictor::init_gpu()";

  // 1. GPU memeroy
  uint32_t gpu_memory_mb = 1024;
  if (!conf->get_uint(prefix, "gpu_memory_mb", gpu_memory_mb)) {
    LOG(WARNING) << "fail to get gpu_memory_mb from conf, set as 1024";
  }
  config.EnableUseGpu(gpu_memory_mb, device);

  // 2. use_tensorrt
  std::string infer_engine;
  if (!conf->get_string(prefix, "infer_engine", infer_engine)) {
    LOG(WARNING) << "disable infer engine";
    return;
  } else if (infer_engine == "tensorrt") {
    init_tensorrt(conf, prefix + "tensorrt_", config);
  } else if (infer_engine == "anakin") {
    init_anakin(conf, prefix + "anakin_", config);
  } else {
    LOG(WARNING) << "unknwon infer engine";
    return;
  }
}

void init_cpu(const ConfParser *conf,
              const std::string &prefix,
              AnalysisConfig &config) {
  LOG(INFO) << "Predictor::init_cpu()";

  config.DisableGpu();

  // 1. cpu_math_library (such as mkl/openblas) num_threads
  int num_threads = 1;
  if (!conf->get_int(prefix, "num_threads", num_threads)) {
    LOG(WARNING) << "fail to get num_threads conf, set as 1";
  }
  num_threads = std::max(1, num_threads);
  config.SetCpuMathLibraryNumThreads(num_threads);

  // 2. use_mkldnn
  int use_mkldnn = -1;
  if (conf->get_int(prefix, "use_mkldnn", use_mkldnn) && use_mkldnn > 0) {
    config.EnableMKLDNN();
  }
}

bool init_model(const ConfParser *conf,
                const std::string &prefix,
                AnalysisConfig &config) {
  LOG(INFO) << "Predictor::init_model()";

  std::string prog_file;
  if (!conf->get_string(prefix, "prog_file", prog_file)) {
    LOG(WARNING) << "fail to get prog_file from conf";
  }

  std::string param_file;
  if (!conf->get_string(prefix, "param_file", param_file)) {
    LOG(WARNING) << "fail to get param_file from conf";
  }

  if (!prog_file.empty() && !param_file.empty()) {
    if (!file_exist(prog_file)) {
      LOG(FATAL) << "file: " << prog_file << " is not exist";
      return false;
    }
    if (!file_exist(param_file)) {
      LOG(FATAL) << "file: " << param_file << " is not exist";
      return false;
    }
    config.SetModel(prog_file, param_file);
    return true;
  }

  std::string model_path;
  if (!conf->get_string(prefix, "model_path", model_path)) {
    LOG(FATAL) << "fail to get model_path from conf";
    return false;
  }
  config.SetModel(model_path);

  return true;
}

void show_version_info() {
  static bool initialized = false;
  if (initialized) {
    return;
  }

  LOG(INFO) << "[date:" << __DATE__ << "]"
            << "[time:" << __TIME__ << "]";
  LOG(INFO) << "paddle " << paddle::get_version();

  initialized = true;
}

bool Predictor::init(const std::string &conf_file, const std::string &prefix) {
  LOG(INFO) << "Predictor::init()";

  show_version_info();

  std::unique_ptr<AnalysisConfig> config(new AnalysisConfig());

  std::unique_ptr<ConfParser> conf(new ConfParser());
  if (!conf->init(conf_file)) {
    LOG(FATAL) << "fail to load conf file: " << conf_file;
    return false;
  }

  // 1. Debug
  if (!conf->get_int(prefix, "debug", _debug)) {
    _debug = -1;
    LOG(WARNING) << "fail to get debug from conf, set as -1";
  }

  // 2. init model
  if (!init_model(conf.get(), prefix, *config.get())) {
    LOG(FATAL) << "fail to init model";
    return false;
  }

  // 3. enable_ir_optim
  int ir_optim = -1;
  if (!conf->get_int(prefix, "enable_ir_optim", ir_optim)) {
    LOG(WARNING) << "fail to get enable_ir_optim from conf, set as false";
  }
  config->SwitchIrOptim(ir_optim > 0);

  // 4. specify_input_name
  int sp_input = -1;
  if (!conf->get_int(prefix, "specify_input_name", sp_input)) {
    LOG(WARNING) << "fail to get specify_input_name from conf, set as false";
  }
  config->SwitchSpecifyInputNames(sp_input > 0);

  // 5. use zerocopy
  config->SwitchUseFeedFetchOps(false);

  // 6. Device
  int device = -1;
  if (!conf->get_int(prefix, "device", device)) {
    LOG(WARNING) << "fail to get device from conf";
    return false;
  }
  if (device < 0) {
    LOG(INFO) << "use cpu!";
    init_cpu(conf.get(), prefix, *config.get());
  } else {
    LOG(INFO) << "use gpu!";
    init_gpu(conf.get(), prefix, device, *config.get());
  }

  // 7. delete unused pass
  std::vector<std::string> passes;
  if (conf->get_strings(prefix, "delete_pass", passes)) {
    for (auto &p : passes) {
      LOG(INFO) << "delete pass: " << p;
      config->pass_builder()->DeletePass(p);
    }
  }

  // 8. create predictor
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(*config.get());
  if (NULL == predictor) {
    LOG(ERROR) << "fail to create paddle predictor";
    return false;
  }
  _predictor = std::move(predictor);

  return true;
}

bool Predictor::init_shared(Predictor *cls) {
  LOG(INFO) << "Predictor::init_shared";

  this->_predictor = std::move(cls->_predictor->Clone());
  if (NULL == this->_predictor) {
    LOG(ERROR) << "fail to clone paddle predictor";
    return false;
  }

  return true;
}

ICNNPredict *Predictor::clone() {
  LOG(INFO) << "Predictor::clone";
  Predictor *cls = new Predictor();

  if (!cls->init_shared(this)) {
    LOG(FATAL) << "fail to call cls->init_shared";
    delete cls;
    return NULL;
  }
  return cls;
}

ICNNPredict *create_cnnpredict(const std::string &conf_file,
                               const std::string &prefix) {
  LOG(INFO) << "create_cnnpredict";
  Predictor *predictor = new Predictor();

  if (!predictor->init(conf_file, prefix)) {
    delete predictor;
    return NULL;
  }

  return predictor;
}

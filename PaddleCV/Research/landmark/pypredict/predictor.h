#pragma once

#include <iostream>
#include <vector>
#include "cnnpredict_interface.h"
#include "common.h"
#include "paddle_inference_api.h"

using paddle::CreatePaddlePredictor;
using paddle::AnalysisConfig;
using paddle::PaddleEngineKind;

class Predictor : public ICNNPredict {
 public:
  Predictor() : _debug(0) {}

  virtual ~Predictor();

  ICNNPredict *clone();

  /**
   * [init predict from conf]
   * @param  conf_file [conf file]
   * @param  prefix [prefix before every key]
   * @return      [true of fasle]
   */
  bool init(const std::string &conf_file, const std::string &prefix);

  bool predict(const std::vector<Tensor> &inputs,
               const std::vector<std::string> &layers,
               std::vector<Tensor> &outputs);

  bool predict(const std::vector<std::vector<float>> &input_datas,
               const std::vector<std::vector<int>> &input_shapes,
               const std::vector<std::string> &layers,
               std::vector<std::vector<float>> &output_datas,
               std::vector<std::vector<int>> &output_shapes);

 private:
  bool init_shared(Predictor *cls);

  int _debug;
  std::unique_ptr<paddle::PaddlePredictor> _predictor;
};

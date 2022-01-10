// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle_api.h" // NOLINT
#include <arm_neon.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <vector>

using namespace paddle::lite_api; // NOLINT
using namespace std;

struct RESULT {
  std::string class_name;
  int class_id;
  float score;
};

std::vector<RESULT> PostProcess(const float *output_data, int output_size,
                                const std::vector<std::string> &word_labels,
                                cv::Mat &output_image) {
  const int TOPK = 5;
  int max_indices[TOPK];
  double max_scores[TOPK];
  for (int i = 0; i < TOPK; i++) {
    max_indices[i] = 0;
    max_scores[i] = 0;
  }
  for (int i = 0; i < output_size; i++) {
    float score = output_data[i];
    int index = i;
    for (int j = 0; j < TOPK; j++) {
      if (score > max_scores[j]) {
        index += max_indices[j];
        max_indices[j] = index - max_indices[j];
        index -= max_indices[j];
        score += max_scores[j];
        max_scores[j] = score - max_scores[j];
        score -= max_scores[j];
      }
    }
  }

  std::vector<RESULT> results(TOPK);
  for (int i = 0; i < results.size(); i++) {
    results[i].class_name = "Unknown";
    if (max_indices[i] >= 0 && max_indices[i] < word_labels.size()) {
      results[i].class_name = word_labels[max_indices[i]];
    }
    results[i].score = max_scores[i];
    results[i].class_id = max_indices[i];
    cv::putText(output_image,
                "Top" + std::to_string(i + 1) + "." + results[i].class_name +
                    ":" + std::to_string(results[i].score),
                cv::Point2d(5, i * 18 + 20), cv::FONT_HERSHEY_PLAIN, 1,
                cv::Scalar(51, 255, 255));
  }
  return results;
}

// fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
void NeonMeanScale(const float *din, float *dout, int size,
                   const std::vector<float> mean,
                   const std::vector<float> scale) {
  if (mean.size() != 3 || scale.size() != 3) {
    std::cerr << "[ERROR] mean or scale size must equal to 3\n";
    exit(1);
  }
  float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(scale[0]);
  float32x4_t vscale1 = vdupq_n_f32(scale[1]);
  float32x4_t vscale2 = vdupq_n_f32(scale[2]);

  float *dout_c0 = dout;
  float *dout_c1 = dout + size;
  float *dout_c2 = dout + size * 2;

  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(din);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(dout_c0, vs0);
    vst1q_f32(dout_c1, vs1);
    vst1q_f32(dout_c2, vs2);

    din += 12;
    dout_c0 += 4;
    dout_c1 += 4;
    dout_c2 += 4;
  }
  for (; i < size; i++) {
    *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
    *(dout_c1++) = (*(din++) - mean[1]) * scale[1];
    *(dout_c2++) = (*(din++) - mean[2]) * scale[2];
  }
}

cv::Mat ResizeImage(const cv::Mat &img, const int &resize_short_size) {
  int w = img.cols;
  int h = img.rows;

  cv::Mat resize_img;

  float ratio = 1.f;
  if (h < w) {
    ratio = float(resize_short_size) / float(h);
  } else {
    ratio = float(resize_short_size) / float(w);
  }
  int resize_h = round(float(h) * ratio);
  int resize_w = round(float(w) * ratio);

  cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
  return resize_img;
}

cv::Mat CenterCropImg(const cv::Mat &img, const int &crop_size) {
  int resize_w = img.cols;
  int resize_h = img.rows;
  int w_start = int((resize_w - crop_size) / 2);
  int h_start = int((resize_h - crop_size) / 2);
  cv::Rect rect(w_start, h_start, crop_size, crop_size);
  cv::Mat crop_img = img(rect);
  return crop_img;
}

std::vector<RESULT>
RunClasModel(std::shared_ptr<PaddlePredictor> predictor, const cv::Mat &img,
             const std::map<std::string, std::string> &config,
             const std::vector<std::string> &word_labels, double &cost_time) {
  // Read img
  int resize_short_size = stoi(config.at("resize_short_size"));
  int crop_size = stoi(config.at("crop_size"));
  int visualize = stoi(config.at("visualize"));

  cv::Mat resize_image = ResizeImage(img, resize_short_size);

  cv::Mat crop_image = CenterCropImg(resize_image, crop_size);

  cv::Mat img_fp;
  double e = 1.0 / 255.0;
  crop_image.convertTo(img_fp, CV_32FC3, e);

  // Prepare input data from image
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize({1, 3, img_fp.rows, img_fp.cols});
  auto *data0 = input_tensor->mutable_data<float>();

  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
  const float *dimg = reinterpret_cast<const float *>(img_fp.data);
  NeonMeanScale(dimg, data0, img_fp.rows * img_fp.cols, mean, scale);

  auto start = std::chrono::system_clock::now();
  // Run predictor
  predictor->Run();

  // Get output and post process
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  auto *output_data = output_tensor->data<float>();
  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  cost_time = double(duration.count()) *
              std::chrono::microseconds::period::num /
              std::chrono::microseconds::period::den;

  int output_size = 1;
  for (auto dim : output_tensor->shape()) {
    output_size *= dim;
  }

  cv::Mat output_image;
  auto results =
      PostProcess(output_data, output_size, word_labels, output_image);

  if (visualize) {
    std::string output_image_path = "./clas_result.png";
    cv::imwrite(output_image_path, output_image);
    std::cout << "save output image into " << output_image_path << std::endl;
  }

  return results;
}

std::shared_ptr<PaddlePredictor> LoadModel(std::string model_file) {
  MobileConfig config;
  config.set_model_from_file(model_file);

  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);
  return predictor;
}

std::vector<std::string> split(const std::string &str,
                               const std::string &delim) {
  std::vector<std::string> res;
  if ("" == str)
    return res;
  char *strs = new char[str.length() + 1];
  std::strcpy(strs, str.c_str());

  char *d = new char[delim.length() + 1];
  std::strcpy(d, delim.c_str());

  char *p = std::strtok(strs, d);
  while (p) {
    string s = p;
    res.push_back(s);
    p = std::strtok(NULL, d);
  }

  return res;
}

std::vector<std::string> ReadDict(std::string path) {
  std::ifstream in(path);
  std::string filename;
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(line);
    }
  } else {
    std::cout << "no such file" << std::endl;
  }
  return m_vec;
}

std::map<std::string, std::string> LoadConfigTxt(std::string config_path) {
  auto config = ReadDict(config_path);

  std::map<std::string, std::string> dict;
  for (int i = 0; i < config.size(); i++) {
    std::vector<std::string> res = split(config[i], " ");
    dict[res[0]] = res[1];
  }
  return dict;
}

void PrintConfig(const std::map<std::string, std::string> &config) {
  std::cout << "=======PaddleClas lite demo config======" << std::endl;
  for (auto iter = config.begin(); iter != config.end(); iter++) {
    std::cout << iter->first << " : " << iter->second << std::endl;
  }
  std::cout << "=======End of PaddleClas lite demo config======" << std::endl;
}

std::vector<std::string> LoadLabels(const std::string &path) {
  std::ifstream file;
  std::vector<std::string> labels;
  file.open(path);
  while (file) {
    std::string line;
    std::getline(file, line);
    std::string::size_type pos = line.find(" ");
    if (pos != std::string::npos) {
      line = line.substr(pos);
    }
    labels.push_back(line);
  }
  file.clear();
  file.close();
  return labels;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: " << argv[0] << " config_path img_path\n";
    exit(1);
  }

  std::string config_path = argv[1];
  std::string img_path = argv[2];

  // load config
  auto config = LoadConfigTxt(config_path);
  PrintConfig(config);

  double elapsed_time = 0.0;
  int warmup_iter = 10;

  bool enable_benchmark = bool(stoi(config.at("enable_benchmark")));
  int total_cnt = enable_benchmark ? 1000 : 1;

  std::string clas_model_file = config.at("clas_model_file");
  std::string label_path = config.at("label_path");

  // Load Labels
  std::vector<std::string> word_labels = LoadLabels(label_path);

  auto clas_predictor = LoadModel(clas_model_file);
  for (int j = 0; j < total_cnt; ++j) {
    cv::Mat srcimg = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::cvtColor(srcimg, srcimg, cv::COLOR_BGR2RGB);

    double run_time = 0;
    std::vector<RESULT> results =
        RunClasModel(clas_predictor, srcimg, config, word_labels, run_time);

    std::cout << "===clas result for image: " << img_path << "===" << std::endl;
    for (int i = 0; i < results.size(); i++) {
      std::cout << "\t"
                << "Top-" << i + 1 << ", class_id: " << results[i].class_id
                << ", class_name: " << results[i].class_name
                << ", score: " << results[i].score << std::endl;
    }
    if (j >= warmup_iter) {
      elapsed_time += run_time;
      std::cout << "Current image path: " << img_path << std::endl;
      std::cout << "Current time cost: " << run_time << " s, "
                << "average time cost in all: "
                << elapsed_time / (j + 1 - warmup_iter) << " s." << std::endl;
    } else {
      std::cout << "Current time cost: " << run_time << " s." << std::endl;
    }
  }

  return 0;
}

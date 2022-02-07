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

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/core/utils/filesystem.hpp>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include <include/cls.h>
#include <include/cls_config.h>

using namespace std;
using namespace cv;
using namespace PaddleClas;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: " << argv[0]
              << " configure_filepath image_path\n";
    exit(1);
  }

  ClsConfig config(argv[1]);

  config.PrintConfigInfo();

  std::string path(argv[2]);

  std::vector<std::string> img_files_list;
  if (cv::utils::fs::isDirectory(path)) {
    std::vector<cv::String> filenames;
    cv::glob(path, filenames);
    for (auto f : filenames) {
      img_files_list.push_back(f);
    }
  } else {
    img_files_list.push_back(path);
  }

  std::cout << "img_file_list length: " << img_files_list.size() << std::endl;

  Classifier classifier(config.cls_model_path, config.cls_params_path,
                        config.use_gpu, config.gpu_id, config.gpu_mem,
                        config.cpu_math_library_num_threads, config.use_mkldnn,
                        config.use_tensorrt, config.use_fp16,
                        config.resize_short_size, config.crop_size);

  double elapsed_time = 0.0;
  int warmup_iter = img_files_list.size() > 5 ? 5 : 0;
  for (int idx = 0; idx < img_files_list.size(); ++idx) {
    std::string img_path = img_files_list[idx];
    cv::Mat srcimg = cv::imread(img_path, cv::IMREAD_COLOR);
    if (!srcimg.data) {
      std::cerr << "[ERROR] image read failed! image path: " << img_path
                << "\n";
      exit(-1);
    }

    cv::cvtColor(srcimg, srcimg, cv::COLOR_BGR2RGB);

    double run_time = classifier.Run(srcimg);
    if (idx >= warmup_iter) {
      elapsed_time += run_time;
      std::cout << "Current image path: " << img_path << std::endl;
      std::cout << "Current time cost: " << run_time << " s, "
                << "average time cost in all: "
                << elapsed_time / (idx + 1 - warmup_iter) << " s." << std::endl;
    } else {
      std::cout << "Current time cost: " << run_time << " s." << std::endl;
    }
  }

  return 0;
}

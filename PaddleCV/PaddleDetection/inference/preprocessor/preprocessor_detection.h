#pragma once

#include "preprocessor.h"

namespace PaddleSolution {

    class DetectionPreProcessor : public ImagePreProcessor {

    public:
        DetectionPreProcessor() : _config(nullptr) {
        };

        bool init(std::shared_ptr<PaddleSolution::PaddleModelConfigPaser> config);
         
        bool single_process(const std::string& fname, std::vector<float> &data, int* ori_w, int* ori_h, int* resize_w, int* resize_h, float* scale_ratio);

        bool batch_process(const std::vector<std::string>& imgs, std::vector<std::vector<float>> &data, int* ori_w, int* ori_h, int* resize_w, int* resize_h, float* scale_ratio);
    private:
        std::shared_ptr<PaddleSolution::PaddleModelConfigPaser> _config;
    };

}

#pragma once

#include "preprocessor.h"

namespace PaddleSolution {

    class ClassifyPreProcessor : public ImagePreProcessor {

    public:
        ClassifyPreProcessor() : _config(nullptr) {
        };

        bool init(std::shared_ptr<PaddleSolution::PaddleSegModelConfigPaser> config);

        bool single_process(const std::string& fname, float* data);

        bool batch_process(const std::vector<std::string>& imgs, float* data);

    private:
        std::shared_ptr<PaddleSolution::PaddleSegModelConfigPaser> _config;
    };

}

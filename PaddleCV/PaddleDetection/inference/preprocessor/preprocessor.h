#pragma once
#include <vector>
#include <string>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils/conf_parser.h"

namespace  PaddleSolution {

class ImagePreProcessor {
protected:
    ImagePreProcessor() {};
    
public:
    virtual ~ImagePreProcessor() {}

    virtual bool single_process(const std::string& fname, float* data, int* ori_w, int* ori_h) {
        return true;
    }

    virtual bool batch_process(const std::vector<std::string>& imgs, float* data, int* ori_w, int* ori_h) {
        return true;
    }

    virtual bool single_process(const std::string& fname, float* data) {
        return true;
    }
    
    virtual bool batch_process(const std::vector<std::string>& imgs, float* data) {
        return true;
    }
    
    virtual bool single_process(const std::string& fname, std::vector<float> &data, int* ori_w, int* ori_h, int* resize_w, int* resize_h, float* scale_ratio) {
	return true;
    }

    virtual bool batch_process(const std::vector<std::string>& imgs, std::vector<std::vector<float>> &data, int* ori_w, int* ori_h, int* resize_w, int* resize_h, float* scale_ratio) {
	return true;
    }

}; // end of class ImagePreProcessor

std::shared_ptr<ImagePreProcessor> create_processor(const std::string &config_file);

} // end of namespace paddle_solution


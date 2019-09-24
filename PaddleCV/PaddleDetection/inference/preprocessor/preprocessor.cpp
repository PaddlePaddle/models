
#include <glog/logging.h>

#include "preprocessor.h"
#include "preprocessor_seg.h"
#include "preprocessor_classify.h"
#include "preprocessor_detection.h"

namespace PaddleSolution {

    std::shared_ptr<ImagePreProcessor> create_processor(const std::string& conf_file) {

        auto config = std::make_shared<PaddleSolution::PaddleSegModelConfigPaser>();
        if (!config->load_config(conf_file)) {
            LOG(FATAL) << "fail to laod conf file [" << conf_file << "]";
            return nullptr;
        }

        if (config->_pre_processor == "SegPreProcessor") {
            auto p = std::make_shared<SegPreProcessor>();
            if (!p->init(config)) {
                return nullptr;
            }
            return p;
        }
        
        if (config->_pre_processor == "ClassifyPreProcessor") {
            auto p = std::make_shared<ClassifyPreProcessor>();
            if (!p->init(config)) {
                return nullptr;
            }
            return p;
        }

        if (config->_pre_processor == "DetectionPreProcessor") {
            auto p = std::make_shared<DetectionPreProcessor>();
            if (!p->init(config)) {
                return nullptr;
            }
            return p;
        }
	

        LOG(FATAL) << "unknown processor_name [" << config->_pre_processor << "]";

        return nullptr;
    }
}

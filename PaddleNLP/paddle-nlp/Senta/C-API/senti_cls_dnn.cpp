#include <vector>
#include <iostream>
#include <map>
#include <fstream>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include "senti_cls_dnn.h"

using std::string;
using std::vector;
using std::map;
using std::ifstream;
using std::cout;
using std::cerr;
using std::endl;

namespace senti_cls_dnn {
    
    paddle::contrib::AnalysisConfig SentiClsDnn::_s_config;
    std::unique_ptr<paddle::PaddlePredictor> SentiClsDnn::_s_main_predictor;
    const string g_unk_word = "<unk>";
    map<std::string, int64_t> SentiClsDnn::_s_vocab_dict;
    
    int SentiClsDnn::global_init(const string& config_path) {
        
        _s_config.model_dir = config_path + "/model/epoch0/";
        _s_config.use_gpu = false;
        _s_config.device = 0;
        _s_config.enable_ir_optim = false;
        _s_main_predictor = paddle::CreatePaddlePredictor<paddle::contrib::AnalysisConfig>(_s_config);
        
        string network_vocab = config_path + "/network.vob";
        int ret = init_vocab_dict(network_vocab);
        if (ret != 0) {
            cerr << "[ERROR] Init vocab_dict failed!" << endl;
            return -1;
        }
        return 0;
    }

    void SentiClsDnn::global_destroy() {
        _s_vocab_dict.clear();
        return;
    }

    int SentiClsDnn::thread_init(const int thread_id) {
        _thread_id = thread_id;
        _predictor = _s_main_predictor->Clone();
        return 0;
    }

    void SentiClsDnn::thread_destroy() {
        return;
    } 

    int SentiClsDnn::init_vocab_dict(const std::string& dict_path) {
        ifstream fin(dict_path.c_str());
        if (fin.is_open() == false) {
            return -1;
        }
        string line;
        int64_t total_count = 0;
        while (getline(fin, line)) {
            vector<string> line_vec;
            boost::split(line_vec, line, boost::is_any_of("\t"));
            if (line_vec.size() != 1) {
                cerr << "[WARNING] Bad line format:\t" << line
                    << endl;
                continue;
            }
            string word = line_vec[0];
            _s_vocab_dict[word] = total_count;
            total_count += 1;
        }
        cerr << "[NOTICE] Total " << total_count
            << " words in vocab(include oov)" << endl;
        _s_vocab_dict[g_unk_word] = total_count;
        return 0;
    }

    int SentiClsDnn::trans_word_to_id(const vector<string>& word_list,
        vector<int64_t>& id_list) {
        for (size_t i = 0; i < word_list.size(); i++) {
            const string& cur_word_str = word_list[i];
            if (_s_vocab_dict.find(cur_word_str) != _s_vocab_dict.end()) {
                id_list.push_back(_s_vocab_dict[cur_word_str]);
            }
            else {
                continue;
            }
        }
        if (id_list.size() <= 0) {
            cerr << "[ERROR] Failed to trans word to id!" << endl;
            return -1;
        }
        return 0;
    }

    void SentiClsDnn::normalize_result(SentiClsRes& senti_cls_res) {
        const float neu_threshold = 0.55; // should be (1, 0.5)
        float prob_0 = senti_cls_res._neg_prob;
        float prob_2 = senti_cls_res._pos_prob;
        if (prob_0 > neu_threshold) {
            // if negative probability > threshold, then the classification
            // label is negative
            senti_cls_res._label = 0;
            senti_cls_res._confidence_val = (prob_0 - neu_threshold)
                 / (1 - neu_threshold);
        }
        else if (prob_2 > neu_threshold) {
            // if positive probability > threshold, then the classification
            // label is positive
            senti_cls_res._label = 2;
            senti_cls_res._confidence_val = (prob_2 - neu_threshold)
                / (1 - neu_threshold);
        }
        else {
            // else the classification label is neural
            senti_cls_res._label = 1;
            senti_cls_res._confidence_val = 1.0 - (fabs(prob_2 - 0.5)
                / (neu_threshold - 0.5));
        }
    }

    int SentiClsDnn::predict(const string& input_str, SentiClsRes& senti_cls_res) {
        // do wordsegment
        string raw_input_str = boost::trim_copy(input_str);
        vector<string> word_list;
        boost::split(word_list, raw_input_str, boost::is_any_of(" "));
        /*
        for (size_t i = 0; i < word_list.size(); i++) {
            cout << word_list[i] << " ";
        }
        cout << endl;
        */

        // trans words to ids
        vector<int64_t> id_list;
        int ret = trans_word_to_id(word_list, id_list);
        if (ret != 0) {
            cerr << "[ERROR] Failed in word_to_id!" << endl;
            return -1;
        }

        // do paddle inference
        paddle::PaddleTensor input_basic_words;
        input_basic_words.shape = std::vector<int>({int(id_list.size()), 1});
        input_basic_words.data = paddle::PaddleBuf(id_list.data(),
            id_list.size() * sizeof(int64_t));
        input_basic_words.dtype = paddle::PaddleDType::INT64;
        input_basic_words.name = "words";
        vector<size_t> cur_basic_lod;
        cur_basic_lod.push_back(0);
        cur_basic_lod.push_back(id_list.size());
        input_basic_words.lod.push_back(cur_basic_lod);
        std::vector<paddle::PaddleTensor> slots({input_basic_words});

        std::vector<paddle::PaddleTensor> outputs;
        _predictor->Run(slots, &outputs);

        const size_t num_elements = outputs.front().data.length() / sizeof(float);
        std::vector<float> prob_res;
        for (size_t i = 0; i < num_elements; i++) {
            float cur_prob = static_cast<float*>(outputs.front().data.data())[i];
            prob_res.push_back(cur_prob);
        }
        // normalize result and re-calculate the pval from prob
        senti_cls_res._pos_prob = prob_res[1];
        senti_cls_res._neg_prob = prob_res[0];
        normalize_result(senti_cls_res);
        return 0;
    } 
}

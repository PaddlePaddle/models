#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>
#include "senti_cls_dnn.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::cerr;

namespace senti_cls_dnn {
    pthread_mutex_t g_thread_mutex;
    // Current thread id
    int g_thread_id = 0;

    void* run_thread(void* pdict) {
        SentiClsDnn senti_cls_dnn_tool;
        pthread_mutex_lock(&g_thread_mutex);
        g_thread_id += 1;
        // Thread Resources Init
        int ret = senti_cls_dnn_tool.thread_init(g_thread_id);
        if (ret != 0) {
            cerr << "[ERROR] thread init failed!" << endl;
            return NULL;
        }
        pthread_mutex_unlock(&g_thread_mutex);
        while (true) {
            string line;
            pthread_mutex_lock(&g_thread_mutex);
            getline(std::cin, line, '\n');
            if (!(std::cin)) {
                pthread_mutex_unlock(&g_thread_mutex);
                break;
            }
            pthread_mutex_unlock(&g_thread_mutex);
            // Sentiment Classification
            SentiClsRes senti_cls_res;
            int ret = senti_cls_dnn_tool.predict(line, senti_cls_res);
            pthread_mutex_lock(&g_thread_mutex);
            cout << line << "\t" << senti_cls_res._pos_prob << "\t"
                << senti_cls_res._neg_prob << "\t"
                << senti_cls_res._confidence_val
                << "\t" << senti_cls_res._label << endl;
            pthread_mutex_unlock(&g_thread_mutex);
        } 
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "./senti_cls_dnn model_path thread_num" << endl;
        return -1;
    }
    // The config path
    string model_path = argv[1];
    // Number of threads
    int thread_num = atoi(argv[2]);

    // Global Resourses Init
    int ret = senti_cls_dnn::SentiClsDnn::global_init(model_path);
    if (ret != 0) {
        cerr << "[ERROR] SentiClsDnn global init failed!" << endl;
        return -1;
    }
    
    // Multi Thread
    pthread_t pids[256] = {0};
    void* (*pf) (void*) = senti_cls_dnn::run_thread;
    for (int i = 0; i < thread_num; i++) {
        int ret = pthread_create(&pids[i], 0, pf, (void*)NULL);
        if (ret != 0) {
            cerr << "[ERROR] create pthread failed!" << endl;
            continue;
        }
    }

    for (int i = 0; i < thread_num; i++) {
        pthread_join(pids[i], NULL);
    }

    // Global Resources Destroy
    senti_cls_dnn::SentiClsDnn::global_destroy();

}

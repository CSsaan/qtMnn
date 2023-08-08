#ifndef STYLETRANSFERMODEL_H
#define STYLETRANSFERMODEL_H
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <QDebug>
#include "Backend.hpp"
#include "MNNDefine.h"
#include "Interpreter.hpp"
#include "Tensor.hpp"
#include <math.h>
#include <iostream>
#include <stdio.h>
using namespace MNN;
using namespace cv;
using namespace std;

class StyleTransferModel
{
public:
    StyleTransferModel(int num_thread_ = 4);
    ~StyleTransferModel();
    // 初始化模型
    void init_model(const char* model_path);
    // 进行转换
    void transfer(cv::Mat &src_mat, cv::Mat &dst_mat);

private:
    // 均值和方差
    float means[3] = {103.94f, 116.78f, 123.68f};
    float norms[3] = {0.017f, 0.017f, 0.017f};
    // mnn 模型
    std::shared_ptr<MNN::Interpreter> interpreter;
    // 推断时用到的变量
    MNN::Session*  session;
    MNN::Tensor* input_tensor;
    MNN::Tensor* output_Tensor;
    // 配置
    MNN::ScheduleConfig config;
    MNN::BackendConfig backendConfig;
};

#endif // STYLETRANSFERMODEL_H

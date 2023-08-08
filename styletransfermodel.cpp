#include "styletransfermodel.h"
#include <chrono>
using namespace std;
using namespace chrono;
#include <ImageProcess.hpp> // MNN::CV

#define MODEL_INPUT_IMG_W 400
#define MODEL_INPUT_IMG_H 400

StyleTransferModel::StyleTransferModel(int num_thread_)
{
    config.numThread = num_thread_;
    config.type      = static_cast<MNNForwardType>(MNN_FORWARD_OPENCL);
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
    config.backendConfig = &backendConfig;
}

StyleTransferModel::~StyleTransferModel()
{
    qDebug() << "pass" ;
}

void StyleTransferModel::init_model(const char *model_path)
{
    // 1. 检查是否非空
    if(model_path == NULL) {
        qDebug() << "open model is NULL!";
        return;
    }
    // 2. 加载模型
    interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path));
    if(interpreter == NULL)
    {
        qDebug() << model_path << " not exit! ";
        return;
    }
    // 3. 创建会话
    session = interpreter->createSession(config);
//    interpreter->releaseModel();
    input_tensor = interpreter->getSessionInput(session, "img_placeholder");
}

void StyleTransferModel::transfer(Mat &src_mat, Mat &dst_mat)
{
    // resize 图片
    cv::Mat resize_src_mat;
    size_t rows = src_mat.rows, cols = src_mat.cols;
    cv::resize(src_mat, resize_src_mat, cv::Size(0, 0), MODEL_INPUT_IMG_W / (float )cols, MODEL_INPUT_IMG_H / (float)rows);
    // 做后处理和拷贝
    interpreter->resizeTensor(input_tensor, {1, 3, MODEL_INPUT_IMG_H, MODEL_INPUT_IMG_W});
    interpreter->resizeSession(session);
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, means, 3, norms, 3));
    pretreat->convert(resize_src_mat.data, MODEL_INPUT_IMG_W, MODEL_INPUT_IMG_H, resize_src_mat.step[0], input_tensor);
    // 运行推理
    system_clock::time_point t1 = system_clock::now();
    interpreter->runSession(session);
    system_clock::time_point t2 = system_clock::now();
    // 获取输出
    output_Tensor = interpreter->getSessionOutput(session, "add_37");
    if(output_Tensor == NULL) {
        qDebug() << "[StyleTransferModel] output tensor is NULL";
        return;
    }
//    MNN::Tensor tensor_scores_host(output_Tensor, output_Tensor->getDimensionType());
//    // 拷贝数据
//    output_Tensor->copyToHostTensor(&tensor_scores_host);
//    // post processing steps
//    auto scores_dataPtr  = tensor_scores_host.host<float>();
    // 进行转换
    const float* buffer = output_Tensor->host<float>();
    cv::Mat tmp(output_Tensor->height(), output_Tensor->width(), CV_32FC3, (void*)buffer);
    tmp.convertTo(dst_mat, CV_8UC3);
    auto minus = t2 - t1;
    std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(minus);
    printf("[StyleTransferModel] shape: (%d, %d, %d, %d), time spend: %lld ms",
           output_Tensor->batch(), output_Tensor->channel(), output_Tensor->height(), output_Tensor->width(), ms);
}

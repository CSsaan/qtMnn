/// 人脸检测

#ifndef ULTRAFACE_H
#define ULTRAFACE_H

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#include "Backend.hpp"
#include "MNNDefine.h"
#include "Interpreter.hpp"
#include "Tensor.hpp"
#include <ImageProcess.hpp> // MNN::CV
using namespace MNN;

#define hard_nms  1
#define blending_nms  2
// 人脸结果信息
typedef struct FaceInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
} FaceInfo;

class UltraFace
{
public:
    UltraFace(int input_width, int input_height, int num_thread_ = 4, float score_threshold_ = 0.7, float iou_threshold_ = 0.2);
    ~UltraFace();
    // 初始化模型
    void init_model(const char* mnn_path);
    int detect(cv::Mat &img, vector<FaceInfo> &face_list);
    cv::Mat Get_Resize_Croped_Img(cv::Mat frame, cv::Point pt1, cv::Point pt2, cv::Point &s_point, cv::Size &croped_wh, Size dst_size);
    int in_w; // 模型输入尺寸
    int in_h;

private:
    int num_thread;
    float score_threshold;
    float iou_threshold;

    int num_anchors;
    int image_w; // 原图像尺寸
    int image_h;

    int num_featuremap = 4; // 特征图数量

    void generateBBox(std::vector<FaceInfo> &bbox_collection, MNN::Tensor *scores, MNN::Tensor *boxes); // 生成所有人脸信息
    void nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type = blending_nms); // IOU筛选人脸

    // MNN
    std::shared_ptr<MNN::Interpreter> interpreter;
    MNN::Session *session = nullptr;
    MNN::Tensor *input_tensor = nullptr;

    const float mean_vals[3] = {127, 127, 127};
    const float norm_vals[3] = {1.0 / 128, 1.0 / 128, 1.0 / 128};
    const float center_variance = 0.1;
    const float size_variance = 0.2;
    const std::vector<std::vector<float>> min_boxes = {
            {10.0f,  16.0f,  24.0f},
            {32.0f,  48.0f},
            {64.0f,  96.0f},
            {128.0f, 192.0f, 256.0f}};
    const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;
    std::vector<std::vector<float>> priors = {};

    // 配置
    MNN::ScheduleConfig config;
    MNN::BackendConfig backendConfig;
};

#endif // ULTRAFACE_H

/// 人脸检测
///

#include "ultraface.h"
#include <QDebug>

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

UltraFace::UltraFace(int input_width, int input_height, int num_thread_, float score_threshold_, float iou_threshold_)
{
    num_thread = num_thread_;
    score_threshold = score_threshold_;
    iou_threshold = iou_threshold_;
    in_w = input_width;
    in_h = input_height;
    std::vector<int> w_h_list = {in_w, in_h};

    // 生成anchors
    for(auto size : w_h_list)
    {
        std::vector<float> fm_item;
        for(float stride : strides)
        {
            fm_item.push_back(size/stride);
        }
        featuremap_size.push_back(fm_item); // std::vector(30, 15, 7.5, 3.75), std::vector(45, 22.5, 11.25, 5.625)
    }

    for (auto size : w_h_list)
    {
        shrinkage_size.push_back(strides);
    }
    /* generate prior anchors */
    for (int index = 0; index < num_featuremap; index++)
    {
        float scale_w = in_w / shrinkage_size[0][index];
        float scale_h = in_h / shrinkage_size[1][index];
        for (int j = 0; j < featuremap_size[1][index]; j++)
        {
            for (int i = 0; i < featuremap_size[0][index]; i++)
            {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;

                for (float k : min_boxes[index])
                {
                    float w = k / in_w;
                    float h = k / in_h;
                    priors.push_back({clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1)});
                }
            }
        }
    }
    /* generate prior anchors finished */
    num_anchors = priors.size();
    qDebug() << "生成anchors完成. num_anchors:" << num_anchors;
}

UltraFace::~UltraFace()
{
    qDebug() << "end UltraFace.";
}

int UltraFace::detect(Mat &raw_image, vector<FaceInfo> &face_list)
{
    if (raw_image.empty())
    {
        std::cout << "image is empty ,please check!" << std::endl;
        return -1;
    }
    image_h = raw_image.rows;
    image_w = raw_image.cols;

    cv::Mat resize_image;
    cv::resize(raw_image, resize_image, cv::Size(0, 0), in_w / (float)image_w, in_h / (float)image_h); // 原图像rezie到模型输入大小
    interpreter->resizeTensor(input_tensor, {1, 3, in_h, in_w});
    interpreter->resizeSession(session);
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, mean_vals, 3, norm_vals, 3));
    pretreat->convert(resize_image.data, in_w, in_h, resize_image.step[0], input_tensor);

    // run network
    interpreter->runSession(session);

    // get output data
    MNN::Tensor *tensor_scores = interpreter->getSessionOutput(session, "scores");
    if(tensor_scores == NULL) {
        qDebug() << "[UltraFaceModel] output tensor_scores is NULL";
        return -1;
    }
    MNN::Tensor *tensor_boxes = interpreter->getSessionOutput(session, "boxes");
    if(tensor_boxes == NULL) {
        qDebug() << "[UltraFaceModel] output tensor_boxes is NULL";
        return -1;
    }

    MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
    tensor_scores->copyToHostTensor(&tensor_scores_host);

    MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());
    tensor_boxes->copyToHostTensor(&tensor_boxes_host);

    std::vector<FaceInfo> bbox_collection;
    generateBBox(bbox_collection, tensor_scores, tensor_boxes);
    nms(bbox_collection, face_list);
    return 0;
}

Mat UltraFace::Get_Resize_Croped_Img(Mat frame, Point pt1, Point pt2, Point &s_point, Size &croped_wh, Size dst_size)
{
    float cx, cy, halfw;
    cv::Mat resize_img, croped_img;
    cv::Point_<float> center_point;

    try{
        center_point = (pt1 + pt2) / 2;
        cx = center_point.x;
        cy = center_point.y;
        halfw = max((pt2.x - pt1.x)/2, (pt2.y - pt1.y)/2);
        float min_x = (cx-halfw) > 0 ? cx-halfw:0;
        float min_y = (cy-halfw) > 0 ? cy-halfw:0;
        halfw = min({halfw, (frame.rows - min_y) / 2, (frame.cols - min_x) / 2});
        croped_img = frame(cv::Rect(min_x, min_y, 2*halfw, 2*halfw));
        croped_wh = cv::Size(2*halfw, 2*halfw);
        qDebug() << "[INFO]>>> croped_wh: " << croped_wh.width << croped_wh.height;

        s_point = cv::Point(min_x, min_y);
        if(halfw > 20)
        {
            cv::resize(croped_img, resize_img, cv::Size(0, 0), (float)dst_size.width / croped_img.cols, (float)dst_size.height / croped_img.rows);
            resize_img.convertTo(resize_img, CV_32FC3);
//            resize_img = (resize_img - 123.0) / 58.0;
        }
    }
    catch(exception e){
        qDebug() << "[INFO]>>> No face was detected!!!";
    }

    return resize_img;
}

void UltraFace::generateBBox(std::vector<FaceInfo> &bbox_collection, Tensor *scores, Tensor *boxes)
{
    for (int i = 0; i < num_anchors; i++)
    {
        if (scores->host<float>()[i * 2 + 1] > score_threshold)
        {
            FaceInfo rects;
            float x_center = boxes->host<float>()[i * 4] * center_variance * priors[i][2] + priors[i][0];
            float y_center = boxes->host<float>()[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
            float w = exp(boxes->host<float>()[i * 4 + 2] * size_variance) * priors[i][2];
            float h = exp(boxes->host<float>()[i * 4 + 3] * size_variance) * priors[i][3];

            rects.x1 = clip(x_center - w / 2.0, 1) * image_w;
            rects.y1 = clip(y_center - h / 2.0, 1) * image_h;
            rects.x2 = clip(x_center + w / 2.0, 1) * image_w;
            rects.y2 = clip(y_center + h / 2.0, 1) * image_h;
            rects.score = clip(scores->host<float>()[i * 2 + 1], 1);
            bbox_collection.push_back(rects);
        }
    }
    qDebug() << "生成BBox完成. BBox_size:" << bbox_collection.size();
}

void UltraFace::nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type)
{
    std::sort(input.begin(), input.end(), [](const FaceInfo &a, const FaceInfo &b) { return a.score > b.score; });
    int box_num = input.size();
    std::vector<int> merged(box_num, 0);

    for (int i = 0; i < box_num; i++) {
        if (merged[i])
            continue;
        std::vector<FaceInfo> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        float h0 = input[i].y2 - input[i].y1 + 1;
        float w0 = input[i].x2 - input[i].x1 + 1;

        float area0 = h0 * w0;

        for (int j = i + 1; j < box_num; j++) {
            if (merged[j])
                continue;

            float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
            float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

            float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
            float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if (inner_h <= 0 || inner_w <= 0)
                continue;

            float inner_area = inner_h * inner_w;

            float h1 = input[j].y2 - input[j].y1 + 1;
            float w1 = input[j].x2 - input[j].x1 + 1;

            float area1 = h1 * w1;

            float score;

            score = inner_area / (area0 + area1 - inner_area);

            if (score > iou_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        switch (type) {
            case hard_nms: {
                output.push_back(buf[0]);
                break;
            }
            case blending_nms: {
                float total = 0;
                for (int i = 0; i < buf.size(); i++) {
                    total += exp(buf[i].score);
                }
                FaceInfo rects;
                memset(&rects, 0, sizeof(rects));
                for (int i = 0; i < buf.size(); i++) {
                    float rate = exp(buf[i].score) / total;
                    rects.x1 += buf[i].x1 * rate;
                    rects.y1 += buf[i].y1 * rate;
                    rects.x2 += buf[i].x2 * rate;
                    rects.y2 += buf[i].y2 * rate;
                    rects.score += buf[i].score * rate;
                }
                output.push_back(rects);
                break;
            }
            default: {
                printf("wrong type of nms.");
                exit(-1);
            }
        }
    }
}

void UltraFace::init_model(const char* mnn_path)
{
    // 1. 检查是否非空
    if(mnn_path == NULL)
    {
        qDebug() << "open model is NULL!";
        return;
    }
    // 2. 加载模型
    interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path));
    if(interpreter == NULL)
    {
        qDebug() << mnn_path << " not exit! ";
        return;
    }
    // 3. 创建会话
    // 配置模型参数
    config.numThread = num_thread;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
    config.backendConfig = &backendConfig;
    config.type = static_cast<MNNForwardType>(MNN_FORWARD_OPENCL);
//    config.type = MNN_FORWARD_OPENGL;  // 设置使用OpenGL进行后端推理
    qDebug() << "模型配置完成. numThread:" << num_thread << "type:MNN_FORWARD_OPENGL";

    session = interpreter->createSession(config);
    input_tensor = interpreter->getSessionInput(session, nullptr);
    qDebug() << "模型初始化完成.";
}

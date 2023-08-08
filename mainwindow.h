#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "styletransfermodel.h"
#include "ultraface.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    void MyFunc(); // 风格迁移模型初始化
    void MyFunc2(); // 人脸检测模型初始化
    ~MainWindow();

    vector<FaceInfo> face_info; // 所有人脸位置信息

private:
    Ui::MainWindow *ui;
    void setUI();
    // 风格迁移
    QString image_name = "/home/liao/sda2/myProject/qtMnn/face5.png";
    const char* model_name = "/home/liao/sda2/myProject/qtMnn/model/rain.mnn";
    std::shared_ptr<StyleTransferModel> model;
    cv::Mat dst_mat; // 结果图片

    // 人脸检测模型
    QString image_name_ultraface = "/home/liao/sda2/myProject/qtMnn/face5.png";
    char model_name_ultraface[50] = "/home/liao/sda2/myProject/qtMnn/model/RFB-320.mnn";
    std::shared_ptr<UltraFace> model_ultraface;
    bool has_face;
    cv::Mat face_mat; // 人脸区域


    // UI
private slots:
    // [风格] tab1
    void slot_push_chosePng();
    void slot_push_infer();
    void slot_push_save();
    void slot_push_ClosePNG();

    // [人脸检测] tab2
    void slot_push_chosePng_ultraface();
    void slot_push_infer_ultraface();
    void slot_push_save_ultraface();
    void slot_push_ClosePNG_ultraface();

    void exit();
    void about();
};
#endif // MAINWINDOW_H

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
using namespace std;
using namespace chrono;
using namespace cv;

#include "Backend.hpp"
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Interpreter.hpp"
#include "Tensor.hpp"
#include <math.h>
#include <iostream>
#include <stdio.h>
using namespace MNN;

#include <QDebug>
#include <QFileDialog>
#include <QFile>
#include <QDebug>
#include <QMessageBox>
#include <QFontDialog>
#include <QShortcut>
#include <QPushButton>
#include <QDateTime>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setUI();
}

void MainWindow::MyFunc()
{
    // 初始化风格迁移模型
    model = std::shared_ptr<StyleTransferModel>(new StyleTransferModel(4));
    model->init_model(model_name);
    // 读取图像数据
    cv::Mat raw_image = cv::imread(image_name.toStdString());
    // 进行推理转换
    model->transfer(raw_image, dst_mat);
    if(dst_mat.empty())
    {
        qDebug() << "推理结果图像为空！";
        return;
    }
}

void MainWindow::MyFunc2()
{
    // 初始化人脸检测模型
    model_ultraface = std::make_shared<UltraFace>(240, 360, 4, 0.95, 0.4);
    model_ultraface->init_model(model_name_ultraface);
    // 读取图像数据
    cv::Mat raw_image = cv::imread(image_name_ultraface.toStdString());
    qDebug()<<"image_name_ultraface:"<<image_name_ultraface;
    // 进行推理
    model_ultraface->detect(raw_image, face_info);
    // 检测所有的人脸
    if(face_info.empty())
    {
        has_face = false;
        return ;
    }
    else
    {
        has_face = true;
        qDebug() << "has faces: " << face_info.size();
    }
    // 查找面积最大的人脸图像
    int max_area_id = 0;
    float max_area = 0;
    for(int i = 0; i < face_info.size(); ++i)
    {
        float area = (face_info[i].x1 - face_info[i].x2) * (face_info[i].y1 - face_info[i].y2);
        if(area > max_area)
        {
            max_area_id = i;
            max_area = area;
        }
    }
    if(max_area < 1000)
    {
        return;
    }

    cv::Size croped_wh;
    cv::Point s_point;
    // *裁剪一个最大的人脸部分*
    cv::Point pt1(face_info[max_area_id].x1, face_info[max_area_id].y1);
    cv::Point pt2(face_info[max_area_id].x2, face_info[max_area_id].y2);
    face_mat = model_ultraface->Get_Resize_Croped_Img(raw_image, pt1, pt2, s_point, croped_wh, cv::Size(96,96)); // 裁剪人脸部分
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::setUI()
{
    // 如果不在ui文件里链接，则在这里手动槽链接
    connect(ui->push_Open_Image, SIGNAL(clicked()), this, SLOT(slot_push_chosePng())); // 打开图片
    connect(ui->push_infer, SIGNAL(clicked()), this, SLOT(slot_push_infer())); // [风格]进行推理
    connect(ui->push_save, SIGNAL(clicked()), this, SLOT(slot_push_save())); // [风格]保存结果
    connect(ui->push_Close, SIGNAL(clicked()), this, SLOT(slot_push_ClosePNG())); // [风格]关闭推理结果

    connect(ui->push_Open_Image2, SIGNAL(clicked()), this, SLOT(slot_push_chosePng_ultraface())); // [人脸检测]打开图片
    connect(ui->push_infer2, SIGNAL(clicked()), this, SLOT(slot_push_infer_ultraface())); // [人脸检测]进行推理
    // 窗口栏按键
    connect(ui->actionExit, &QAction::triggered, this, &MainWindow::exit);
    connect(ui->actionAbout, &QAction::triggered, this, &MainWindow::about);
}

void MainWindow::slot_push_chosePng()
{
    QString OpenFile;
    OpenFile = QFileDialog::getOpenFileName(this, "Choose image", "", "Image Files(*.jpg *.png *.bmp *.pgm *.pbm);;All(*.*)");
    image_name = OpenFile;
    qDebug() << image_name;

    // 显示原始图片
    QImage qimg;
    qimg.load(image_name);
    QImage resizeImg = qimg.scaled(ui->label_orig->width(), ui->label_orig->height());
    ui->label_orig->setPixmap(QPixmap::fromImage(resizeImg));
    ui->label_orig->show();
}

void MainWindow::slot_push_infer()
{
    system_clock::time_point t1 = system_clock::now();
    // 开始推理
    MyFunc();
    system_clock::time_point t2 = system_clock::now();
    auto minus = t2 - t1;
    std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(minus);
    qDebug() <<"[time spend]:" << ms.count() << "ms";

    // 显示结果图片
    if(!dst_mat.empty())
    {
        cvtColor(dst_mat, dst_mat, COLOR_BGR2RGB);
        QImage qimg((const unsigned char*)dst_mat.data, dst_mat.cols, dst_mat.rows, dst_mat.step, QImage::Format_RGB888);
        QImage resizeImg = qimg.scaled(ui->label_result->width(), ui->label_result->height());
        ui->label_result->setPixmap(QPixmap::fromImage(resizeImg));
        ui->label_result->show();
    }
    else
    {
        QMessageBox::warning(this, "Warning", "推理结果为空！");
        return;
    }
}

void MainWindow::slot_push_save()
{
    //获取保存路径
    if(dst_mat.empty())
        return;
    QString _TempPath  = QCoreApplication::applicationDirPath();//.exe项目文件所在目录
    QDateTime dtCurtime = QDateTime::currentDateTime();  //设置当前时间为保存时间
    _TempPath = _TempPath + "//Image//" + dtCurtime.toString("yyyyMMddhhmmss");  //保存名称为path + image（） + 日期年y月M日d时h分m秒s
    QString strFileName = QFileDialog::getSaveFileName(this,tr("Save Picture"),  //类函数QFileDiaLog:获取文件路径//getSaveFileName:获取保存文件名字
                                               _TempPath,
                                               "PNG(*.png);;JPG(*.jpg);;BMP(*.bmp);;TIF(*.tif)");
    if(!strFileName.isNull())
    {
        cv::imwrite(strFileName.toStdString(), dst_mat);
        qDebug() << strFileName;
    }
}

void MainWindow::slot_push_ClosePNG()
{
    ui->label_orig->clear();
    ui->label_result->clear();
}

void MainWindow::slot_push_chosePng_ultraface()
{
    QString OpenFile;
    OpenFile = QFileDialog::getOpenFileName(this, "Choose image", "", "Image Files(*.jpg *.png *.bmp *.pgm *.pbm);;All(*.*)");
    image_name_ultraface = OpenFile;
    qDebug() << image_name_ultraface;

    // 显示原始图片
    QImage qimg;
    qimg.load(image_name_ultraface);
    QImage resizeImg = qimg.scaled(ui->label_orig2->width(), ui->label_orig2->height());
    ui->label_orig2->setPixmap(QPixmap::fromImage(resizeImg));
    ui->label_orig2->show();
}

void MainWindow::slot_push_infer_ultraface()
{
    system_clock::time_point t1 = system_clock::now();
    // 开始推理
    MyFunc2();
    system_clock::time_point t2 = system_clock::now();
    auto minus = t2 - t1;
    std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(minus);
    qDebug() <<"[time spend]:" << ms.count() << "ms";

    // 显示一个人最大脸结果图片
    if(!face_mat.empty())
    {
        face_mat.convertTo(face_mat, CV_8UC3);
        cvtColor(face_mat, face_mat, COLOR_BGR2RGB);
        QImage qimg((const unsigned char*)face_mat.data, face_mat.cols, face_mat.rows, face_mat.step, QImage::Format_RGB888);
        QImage resizeImg = qimg.scaled(ui->label_result2->width(), ui->label_result2->height());
        ui->label_result2->setPixmap(QPixmap::fromImage(resizeImg));
        ui->label_result2->show();
    }
    else
    {
        QMessageBox::warning(this, "Warning", "推理结果为空！");
        return;
    }

    // 更新原始图像为所有人脸方框图
    cv::Point s_point;
    cv::Size croped_wh;
    Mat src_img = cv::imread(image_name_ultraface.toStdString());
    Mat faces_img = src_img.clone();
    for(int i = 0; i < face_info.size(); ++i)
    {
        cv::Point pt1(face_info[i].x1, face_info[i].y1);
        cv::Point pt2(face_info[i].x2, face_info[i].y2);
        Mat _face_mat = model_ultraface->Get_Resize_Croped_Img(src_img, pt1, pt2, s_point, croped_wh, cv::Size(96,96)); // 裁剪人脸部分
//        rectangle(faces_img, s_point, Point(s_point.x+croped_wh.width, s_point.y+croped_wh.height), Scalar(0, 0, 255), 1, 8, 0);//黄色矩形框
        qDebug() << s_point.x << s_point.y << croped_wh.width << croped_wh.height;
        cv::rectangle(faces_img, cv::Point(s_point.x, s_point.y), cv::Point(s_point.x+croped_wh.width, s_point.y+croped_wh.height), cv::Scalar(0, 0, 255), 4, 8, 0);
    }
    cvtColor(faces_img, faces_img, COLOR_BGR2RGB);
    QImage qim((const unsigned char*)faces_img.data, faces_img.cols, faces_img.rows, faces_img.step, QImage::Format_RGB888);
    QImage resizeImg = qim.scaled(ui->label_orig2->width(), ui->label_orig2->height());
    ui->label_orig2->setPixmap(QPixmap::fromImage(resizeImg));
    ui->label_orig2->show();

}

void MainWindow::slot_push_save_ultraface()
{
    qDebug() << "slot_push_save_ultraface pass";
}

void MainWindow::slot_push_ClosePNG_ultraface()
{
    qDebug() << "slot_push_ClosePNG_ultraface pass";
}

void MainWindow::exit()
{
    QCoreApplication::quit();
}

void MainWindow::about()
{
    QMessageBox msgBox;
    msgBox.setBaseSize(200,200);
    msgBox.setText("AI by MNN model test.");
    msgBox.setInformativeText("A AI model infer test by CS.");
//    QIcon *icon = new QIcon("/home/liao/sda2/myProject/NotepadCS/cs.jpg");
//    msgBox.setWindowIcon(*icon);
//    msgBox.setStyleSheet("image:url(:/home/liao/sda2/myProject/NotepadCS/cs.ico)");
    msgBox.setStandardButtons(QMessageBox::Cancel);
    msgBox.exec();
}




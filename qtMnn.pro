QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11



TARGET = styletransfermodel
CONFIG += console
CONFIG -= app_bundle
TEMPLATE = app
#opencv
INCLUDEPATH += /usr/local/include/opencv \
               /usr/local/include \

LIBS += -L/home/liao/sda2/opencv/opencv-3.4.15/build/lib \
 -lopencv_stitching -lopencv_objdetect \
-lopencv_superres -lopencv_videostab  \#-lippicv -lopencv_shape -lopencv_videoio
-lopencv_imgcodecs\
-lopencv_calib3d -lopencv_features2d -lopencv_highgui \
-lopencv_video \
-lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_core
#mnn
INCLUDEPATH +=/home/liao/sda2/mnn/MNN/include \
/home/liao/sda2/mnn/MNN/include/MNN \
/home/liao/sda2/mnn/MNN/schema/current \
/home/liao/sda2/mnn/MNN/tools \
/home/liao/sda2/mnn/MNN/tools/cpp \
/home/liao/sda2/mnn/MNN/source \
/home/liao/sda2/mnn/MNN/source/backend \
/home/liao/sda2/mnn/MNN/source/core \
/home/liao/sda2/mnn/MNN/source/cv \
/home/liao/sda2/mnn/MNN/source/math \
/home/liao/sda2/mnn/MNN/source/shape \
/home/liao/sda2/mnn/MNN/3rd_party \
/home/liao/sda2/mnn/MNN/3rd_party/imageHelper
LIBS += -L/home/liao/sda2/mnn/MNN/build
LIBS += -lMNN



# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    styletransfermodel.cpp \
    ultraface.cpp

HEADERS += \
    mainwindow.h \
    styletransfermodel.h \
    ultraface.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

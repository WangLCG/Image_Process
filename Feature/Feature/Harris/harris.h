#pragma once

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
using namespace std;

/*
RGB转换成灰度图像的一个常用公式是：
Gray = R*0.299 + G*0.587 + B*0.114
*/
//******************灰度转换函数*************************  
//第一个参数image输入的彩色RGB图像的引用；  
//第二个参数imageGray是转换后输出的灰度图像的引用；  
//*******************************************************
void ConvertRGB2GRAY(const Mat &image, Mat &imageGray);

//******************Sobel卷积因子计算X、Y方向梯度和梯度方向角********************  
//第一个参数imageSourc原始灰度图像；  
//第二个参数imageSobelX是X方向梯度图像；  
//第三个参数imageSobelY是Y方向梯度图像；  
//第四个参数pointDrection是梯度方向角数组指针  
//*************************************************************  
void SobelGradDirction(Mat &imageSource, Mat &imageSobelX, Mat &imageSobelY);

//******************计算Sobel的X方向梯度幅值的平方*************************  
//第一个参数imageGradX是X方向梯度图像；    
//第二个参数SobelAmpXX是输出的X方向梯度图像的平方  
//*************************************************************  
void SobelXX(const Mat imageGradX, Mat_<float> &SobelAmpXX);

//******************计算Sobel的Y方向梯度幅值的平方*************************    
//第一个参数imageGradY是Y方向梯度图像；  
//第二个参数SobelAmpXX是输出的Y方向梯度图像的平方  
//*************************************************************  
void SobelYY(const Mat imageGradY, Mat_<float> &SobelAmpYY);

//******************计算Sobel的XY方向梯度幅值的乘积*************************    
//第一个参数imageGradX是X方向梯度图像；
//第二个参数imageGradY是Y方向梯度图像；
//第二个参数SobelAmpXY是输出的XY方向梯度图像 
//*************************************************************  
void SobelXY(const Mat imageGradX, const Mat imageGradY, Mat_<float> &SobelAmpXY);

//****************计算一维高斯的权值数组*****************
//第一个参数size是代表的卷积核的边长的大小
//第二个参数sigma表示的是sigma的大小
//*******************************************************
double *getOneGuassionArray(int size, double sigma);

//****************高斯滤波函数的实现*****************
//第一个参数srcImage是代表的输入的原图
//第二个参数dst表示的是输出的图
//第三个参数size表示的是卷积核的边长的大小
//*******************************************************
void MyGaussianBlur(Mat_<float> &srcImage, Mat_<float> &dst, int size);

//****计算局部特涨结果矩阵M的特征值和响应函数H = (A*B - C) - k*(A+B)^2******
//M
//A  C
//C  B
//Tr(M)=a+b=A+B
//Det(M)=a*b=A*B-C^2
//计算输出响应函数的值得矩阵
//****************************************************************************
void harrisResponse(Mat_<float> &GaussXX, Mat_<float> &GaussYY, Mat_<float> &GaussXY, Mat_<float> &resultData, float k);


//***********非极大值抑制和满足阈值及某邻域内的局部极大值为角点**************
//第一个参数是响应函数的矩阵
//第二个参数是输入的灰度图像
//第三个参数表示的是输出的角点检测到的结果图
void LocalMaxValue(Mat_<float> &resultData, Mat &srcGray, Mat &ResultImage, int kSize);


//********************* CStyle ************************************
//////////////////////////////////////////////////////////////////////////
/// \brief  计算局部特涨结果矩阵M的特征值和响应函数H = (A*B - C) - k*(A+B)^2
/// \remark 
/// \param[in]  GaussXX  高斯滤波后X方向梯度平方
/// \param[in]  GaussYY  高斯滤波后Y方向梯度平方
/// \param[in/out]  resultData   输出响应
/// \param[in]  width   图像宽
/// \param[in]  height  图像高
//////////////////////////////////////////////////////////////////////////
/*****************************************
        M
        A  C
        C  B
        Tr(M)=a+b=A+B
        Det(M)=a*b=A*B-C^2
        计算输出响应函数的值得矩阵
*****************************************/
void harrisResponse_CStyle(float* GaussXX, float* GaussYY, float* GaussXY, float* resultData, int width, int height, float k);

//////////////////////////////////////////////////////////////////////////
/// \brief  梯度图高斯滤波
/// \remark 
/// \param[in]  srcImage  梯度图
/// \param[in]  width   图像宽
/// \param[in]  height  图像高
/// \param[in/out]  dst  滤波输出
/// \param[in]  k_size   高斯滤波尺寸
/// \param[in]  channels   图像通道数，暂时支持单通道
//////////////////////////////////////////////////////////////////////////
int MyGaussianBlur_CStyle(float *srcImage, int width, int height, float* dst, int k_size, int channels);

//////////////////////////////////////////////////////////////////////////
/// \brief  计算X,Y梯度乘积
/// \remark 
/// \param[in]  imageGradX  X方向梯度图
/// \param[in]  imageGradY  Y方向梯度图
/// \param[in]  width   图像宽
/// \param[in]  height  图像高
/// \param[in/out]  SobelAmpXY  输出
//////////////////////////////////////////////////////////////////////////
void SobelXY_CStyle(float* imageGradX, float* imageGradY, int width, int height, float* SobelAmpXY);

//////////////////////////////////////////////////////////////////////////
/// \brief  计算Y梯度平方
/// \remark 
/// \param[in]  imageGradY  Y方向梯度图
/// \param[in]  width   图像宽
/// \param[in]  height  图像高
/// \param[in/out]  SobelAmpYY  输出
//////////////////////////////////////////////////////////////////////////
void SobelYY_CStyle(float* imageGradY, int width, int height, float* SobelAmpYY);

//////////////////////////////////////////////////////////////////////////
/// \brief  计算X梯度平方
/// \remark 
/// \param[in]  imageGradX  X方向梯度图
/// \param[in]  width   图像宽
/// \param[in]  height  图像高
/// \param[in/out]  SobelAmpXX  输出
//////////////////////////////////////////////////////////////////////////
void SobelXX_CStyle(float* imageGradX, int width, int height, float* SobelAmpXX);

//////////////////////////////////////////////////////////////////////////
/// \brief  计算图像梯度
/// \remark 
/// \param[in]  imageSource  源图（灰度）
/// \param[in]  width   图像宽
/// \param[in]  height  图像高
/// \param[in/out]  imageGradX  X方向梯度图
/// \param[in/out]  imageGradY  Y方向梯度图
//////////////////////////////////////////////////////////////////////////
void SobelGradDirction_CStyle(unsigned char *imageSource, int width, int height, float* imageSobelX, float* imageSobelY);

//////////////////////////////////////////////////////////////////////////
/// \brief  3X3 角点响应图非极大值抑制
/// \remark 
/// \param[in]  resultData  角点响应图
/// \param[in]  srcGray     源灰度应图
/// \param[in/out]  ResultImage  输出--极大值角点处画点
/// \param[in]  width   图像宽
/// \param[in]  height  图像高
//////////////////////////////////////////////////////////////////////////
void Local_3x3_MaxValue_CStyle(float *resultData, unsigned char* srcGray, unsigned char* ResultImage, int width, int height);

//使用opencv实现的harris角点响应测试
int harris_main();

//使用C语言实现的harris角点响应测试
int harris_main_CStyle();
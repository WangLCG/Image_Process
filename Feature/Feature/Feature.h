//////////////////////////////////////////////////////////////////////////
/// \file       Feature.h
/// \brief      LBP特征提取头文件
/// \author     
/// \version    
///             [V1.0] 2018/11/12, draft
/// \copyright  
///             
//////////////////////////////////////////////////////////////////////////

#ifndef  _FEATURE_H

#define _FEATURE_H
//for opencv
#include "cvhead.h"

using namespace cv;
using namespace std;

//////////////////////////////////////////////////////////////////////////
/// \brief  原始LBP特征提取
/// \remark 
/// \param[in]  src_img  源灰度图
/// \param[in]  width    源图宽
/// \param[in]  height   源图高
/// \param[in/out]  lbp_img  输出lbp特征图
//////////////////////////////////////////////////////////////////////////
void LBP_3X3_Ori(unsigned char* src_img, int width, int height, unsigned char* lbp_img);

//////////////////////////////////////////////////////////////////////////
/// \brief  旋转不变LBP特征
/// \remark 
/// \param[in]  src_img  源灰度图
/// \param[in]  width    源图宽
/// \param[in]  height   源图高
/// \param[in/out]  lbp_img  输出lbp特征图
//////////////////////////////////////////////////////////////////////////
void LBP_3X3_Ratio(unsigned char* src_img, int width, int height, unsigned char* lbp_img);

//////////////////////////////////////////////////////////////////////////
/// \brief  圆形LBP特征
/// \remark 
/// \param[in]  src_img  源灰度图
/// \param[in]  width    源图宽
/// \param[in]  height   源图高
/// \param[in]  radius    采样半径
/// \param[in]  neighbors   采样点数，默认为8
/// \param[in/out]  lbp_img  输出lbp特征图
//////////////////////////////////////////////////////////////////////////
void LBP_Circle(unsigned char* src_img, int width, int height, int radius, int neighbors, unsigned char* lbp_img);

//////////////////////////////////////////////////////////////////////////
/// \brief  获取8bit数内的跳变次数
/// \remark 
/// \param[in]  num
/// \return  num的跳变次数
//////////////////////////////////////////////////////////////////////////
int GetHopCount(unsigned char num);

//////////////////////////////////////////////////////////////////////////
/// \brief  Uniform LBP特征
/// \remark 
/// \param[in]  src_img  源灰度图
/// \param[in]  width    源图宽
/// \param[in]  height   源图高
/// \param[in/out]  lbp_img  输出lbp特征图
//////////////////////////////////////////////////////////////////////////
void Unifom_LBP(unsigned char* src_img, int width, int height, unsigned char* lbp_img);

//////////////////////////////////////////////////////////////////////////
/// \brief  MB-LBP特征
/// \remark 
/// \param[in]  src_img  源灰度图
/// \param[in]  width    源图宽
/// \param[in]  height   源图高
/// \param[in]  Block_size   分块尺寸,默认使用9
/// \param[in/out]  lbp_img  输出lbp特征图
//////////////////////////////////////////////////////////////////////////
void MB_LBP(unsigned char* src_img, int width, int height, int Block_size, unsigned char* lbp_img);

//////////////////////////////////////////////////////////////////////////
/// \brief  获取lbp图像的LBPH特征
/// \remark 
/// \param[in]  src  lbp特征图
/// \param[in]  numPatterns    lbp特征图的模式种类（uniform lbp为59）
/// \param[in]  grid_x   分块的宽
/// \param[in]  grid_y   分块的高
/// \param[in]  normed   是否归一化直方图，默认true
/// \return     输出全图的lbph特征
//////////////////////////////////////////////////////////////////////////
Mat getLBPH(Mat src, int numPatterns, int grid_x, int grid_y, bool normed);

//////////////////////////////////////////////////////////////////////////
/// \brief  获取lbp图像分块的LBPH特征
/// \remark 
/// \param[in]  src  lbp特征图
/// \param[in]  minValue   图像最小灰度值
/// \param[in]  maxValue   图像最大灰度值
/// \param[in]  normed   是否归一化直方图，默认true
/// \return     输出块的lbph特征
//////////////////////////////////////////////////////////////////////////
Mat getLocalRegionLBPH(const Mat& src, int minValue, int maxValue, bool normed);

int LBP();


#define ENABLE_INTEGRAL   (1)
//////////////////////////////////////////////////////////////////////////
/// \brief  积分图的快速实现算法
/// \remark 
/// \param[in]  src    源灰度图
/// \param[in]  width  源灰度图宽
/// \param[in]  height  源灰度图高
/// \param[in/out]  integral   积分图输出,为了用opencv显示出来才定义为float型的，实际可使用int*
/// \return     success/fail
//////////////////////////////////////////////////////////////////////////
int integral_image_cacul(const unsigned char*  src, int width, int height, float *  integral);

#endif
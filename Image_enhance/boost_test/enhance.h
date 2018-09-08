/************************************************************************/
/*     一些图像增强的C实现 只保证QCIF/CIF大小下的正确性，灰度图
*/
/************************************************************************/
#ifndef _ENHANCE_H_
#define _ENHANCE_H_

typedef struct _hv_enhance_process
{
    int  enable_hist_equlized;
    int  enable_box_filter;
    int  enable_gaussian_filter;
    int  enable_lapalce_sharpen;
    int  enable_gamma_core;
    int  enable_white_balance;
}
hv_enhance_process;

/* 
    公式参考：
        https://en.wikipedia.org/wiki/YUV#Converting_between_Y%E2%80%B2UV_and_RGB
        https://zh.wikipedia.org/wiki/YUV#YUV%E8%BD%89RGB  
*/
//rgb24 --> uyvy422
//y=0.257*r+0.504*g+0.098*b+16
//u = -0.148*r - 0.291*g + 0.439*b + 128
//v=-0.439*r - 0.368*g - 0.071*b + 128

//256倍扩大  
#define GetY_256(R,G,B) (66*(R) + 129 * (G) + 25*(B) )
#define GetU_256(R,G,B) (-38*(R) - 74*(G) + 112*(B) )
#define GetV_256(R,G,B) (112*(R) - 94*(G) - 18*(B) )

#define GetY(R,G,B) (0.257*(R)+0.504*(G)+0.098*(B)+16)
#define GetU(R,G,B) (-0.148*(R)-0.291*(G)+0.439*(B)+128)
#define GetV(R,G,B) (0.439*(R)-0.368*(G)-0.071*(B)+128)

//uyvy --> rgb 24
/*
    R = Y + 1.403(V' - 128)  //v == cr

    G = Y - 0.344(U' - 128) - 0.714(V'- 128)

    B = Y + 1.770(U' - 128)  // u == cb 

*/
#define GET_R_256(Y, V)     ((Y) + 359 * ((V)))
#define GET_G_256(Y, U, V)  ((Y) - 88 * ((U)) - 183 * ((V)))
#define GET_B_256(Y, U)     ((Y) + 453 * ((U)))

#define GET_R(Y, V)     ((Y) + 1.403 * ((V) - 128))
#define GET_G(Y, U, V)  ((Y) - 0.344 * ((U) - 128) - 0.714 * ((V) - 128))
#define GET_B(Y, U)     ((Y) + 1.770 * ((U) - 128))

#define TUNE(r) ( (r) < 0 ? 0 : ((r) > 255 ? 255 : (r)) )

///////////////////////////////////////////////////////////////////////////////////////////
/// \brief 直方图均衡化
/// \detail 
/// \param[in] src      
/// \param[in] width    
/// \param[in] pitch    
/// \param[in] height   
/// \param[in/out] dst      
/// \param[in] scratchBuf  内部用临时缓存 4 * 256 * sizeof(int)
////////////////////////////////////////////////////////////////////////////////////////////
int HistEqulizedInt(
    unsigned char * src,
    int width, int height,
    unsigned char * dst,
    int * scratchBuf);

int HistEqulizedF32(
    unsigned char * src,
    int width, int height,
    unsigned char * dst,
    int * scratchBuf);


//////////////////////////////////////////////////////////////////////////
/// \brief Box filter used 5x5 kernel to process QCIF char image.
/// \details    
/// \param[in]  srcImg      The input source image data.
/// \param[in]  srcImgW     The width of input source image data.
/// \param[in]  srcImgH     The height of input source image data.
/// \param[out] dstImg      The output destination image data, the 5x5 neighbor summer value,
/// \                       it need to divide 25 to get the filter value  
/// \param[in]  scratchBuf  The iram buffer, size is  176 * sizeof(short) .
/// \return     The error code -1. 0 if succeeds.
//////////////////////////////////////////////////////////////////////////
int IFBoxFilter8UCHAR5x5_QCIF(
    const unsigned char*   srcImg,
    const int srcImgW,
    const int srcImgH,
    short*   dstImg,
    unsigned char*   scratchBuf);

int IFBoxFilter8UCHAR3x3_QCIF(
    const unsigned char*   srcImg,
    const int srcImgW,
    const int srcImgH,
    short*   dstImg,
    unsigned char*   scratchBuf);


//////////////////////////////////////////////////////////////////////////
/// \brief Box filter used 5x5 kernel to process  CIF char image.
/// \details    
/// \param[in]  srcImg      The input source image data.
/// \param[in]  srcImgW     The width of input source image data.
/// \param[in]  srcImgH     The height of input source image data.
/// \param[in]  filterW     The width of filter.
/// \param[in]  filterH     The height of filter.
/// \param[out] dstImg      The output destination image data, the 5x5 neighbor summer value,
/// \                       it need to divide 25 to get the filter value  
/// \param[in]  scratchBuf   352 * sizeof(short) .
/// \return     The error code -1. 0 if succeeds.
//////////////////////////////////////////////////////////////////////////
int IFBoxFilter8UCHAR5x5_CIF(
    const unsigned char*   srcImg,
    const int srcImgW,
    const int srcImgH,
    short*   dstImg,
    unsigned char*   scratchBuf);

int IFBoxFilter8UCHAR3x3_CIF(
    const unsigned char*   srcImg,
    const int srcImgW,
    const int srcImgH,
    short*   dstImg,
    unsigned char*   scratchBuf);

//////////////////////////////////////////////////////////////////////////
/// \brief Box filter used 3x3 kernel to process char image(cif or qcif )
/// \details    通用的处理CIF或QCIF分辨率的均值滤波实现   
/// \param[in]  srcImg      The input source image data.
/// \param[in]  srcImgW     The width of input source image data.
/// \param[in]  srcImgH     The height of input source image data.
/// \param[in]  filterW     The width of filter.
/// \param[in]  filterH     The height of filter.
/// \param[out] dstImg      The output destination image data, the 5x5 neighbor summer value,
/// \                       it need to divide 25 to get the filter value  
/// \param[in]  scratchBuf  The iram buffer, size is srcImgW * srcImgH * sizeof(unsigned char) + 176 * sizeof(short) .
/// \return     The error code -1. 0 if succeeds.
//////////////////////////////////////////////////////////////////////////

int IFBoxFilter8UCHAR3x3(
    const unsigned char*   srcImg,
    const int srcImgW,
    const int srcImgH,
    short*   dstImg,
    unsigned char*   scratchBuf);

//////////////////////////////////////////////////////////////////////////
/// \brief Box filter used 5x5 kernel to process float image.
/// \details    
/// \param[in]  srcImg      The input source image data.
/// \param[in]  srcImgW     The width of input source image data.
/// \param[in]  srcImgH     The height of input source image data.
/// \param[out] dstImg      The output destination image data that after blur.
/// \return     The error code -1. 0 if succeeds.
//////////////////////////////////////////////////////////////////////////
int IFBoxFilter8UC5x5(
    const unsigned char*   srcImg,
    const int srcImgW,
    const int srcImgH,
    float*   dstImg);

///////////////////////////////////////////////////////////////////////////////////////////
/// \brief 高斯3x3滤波 （定点化）
/// \detail 
/// \param[in] src      
/// \param[in] width    
/// \param[in] height   
/// \param[in/out] dst      
////////////////////////////////////////////////////////////////////////////////////////////
int GaussianKernelFilterIq23(unsigned char * src, unsigned char * dst, int width, int height);


///////////////////////////////////////////////////////////////////////////////////////////
/// \brief 图像锐化
/// \detail 
/// \param[in] src      
/// \param[in] width    
/// \param[in] pitch    
/// \param[in] height   
/// \param[in] dst      
////////////////////////////////////////////////////////////////////////////////////////////
int Laplacian8U_3x3(const unsigned char *   src, unsigned char *   dst, int width, int height);
int Laplacian8U_5x5(const unsigned char *   src, unsigned char *   dst, int nWidth, int nHeight);
int GnPC_Laplacian8U(const unsigned char *   src, unsigned char *   dst, int nWidth, int nHeight, int ksize);
int GnPC_CFLaplacian(unsigned char* src, unsigned char* dst, int width, int height, int ksize);

///////////////////////////////////////////////////////////////////////////////////////////
/// \brief Gamma矫正
/// \detail 
/// \param[in] src      
/// \param[in] width    
/// \param[in] height   
/// \param[in] gamma      gamma矫正系数 
/// \param[out] dst      
/// \param[tnt] updat_LUT  更新查找表标志位  
////////////////////////////////////////////////////////////////////////////////////////////
void GammaCorrectiom(unsigned char*   src, int src_w, int src_h, 
                            float gamma, unsigned char*   dst, int updat_LUT);
//更新gamma矫正的查找表  
void updat_gamma_lut(float gamma);

///////////////////////////////////////////////////////////////////////////////////////////
/// \brief uyvy 转RGB24
/// \detail 
/// \param[in] uyvydata      
/// \param[in] src_w    
/// \param[in] src_h   
/// \param[in] rgbdata     
/// \param[out] buf      rgb图像实际存储位置  
/// \param[out] rgb_sum      rgb图像所有通道的像素值统计，rgb_sum[0] - b通道, 1 - g, 2 - r .白平衡使用  
////////////////////////////////////////////////////////////////////////////////////////////

void uyvy2rgb24( unsigned char *   uyvydata, int width, int height, 
            unsigned char *  rgbdata, unsigned char *  buf , unsigned int*   rgb_sum);

///////////////////////////////////////////////////////////////////////////////////////////
/// \brief RGB图像白平衡--灰度世界法
/// \detail 
/// \param[in/out] src_img      
/// \param[in] src_w    
/// \param[in] src_h   
////////////////////////////////////////////////////////////////////////////////////////////
void white_balance_gray_world(unsigned char *  rgb_img, int w, int h);

///////////////////////////////////////////////////////////////////////////////////////////
/// \brief RGB24 转 YUV422Packed格式(uyvy)
/// \detail 
/// \param[in] rgbData      
/// \param[in] src_w    
/// \param[in] src_h   
/// \param[in/out] yuvData   
////////////////////////////////////////////////////////////////////////////////////////////
void  rgb2yuv422Packed(unsigned char *   rgbData, int width, int height,  unsigned char *   yuvData);

///////////////////////////////////////////////////////////////////////////////////////////
/// \brief uyvy图像的白平衡实现--灰度世界法
/// \detail 
/// \param[in/out] uyvybuf      原始图像
/// \param[in] w    
/// \param[in] h   
/// \param[in] rgbbuf    暂存rgb图像，至少3*w*h
/// \param[in] buf       临时计算用，至少3*w*h
////////////////////////////////////////////////////////////////////////////////////////////

int  uyvy_white_balance(unsigned char *  uyvybuf, int w, int h, 
        unsigned char *  rgbbuf, unsigned char *  buf);

///////////////////////////////////////////////////////////////////////////////////
/// \brief yuv420p格式转uyvy
/// \detail 
/// \param[in] y_plane    指向I420格式中的Y平面
/// \param[in] u_plane    指向I420格式中的U平面
/// \param[in] v_plane    指向I420格式中的v平面
/// \param[in] y_stride   I420格式中Y平面的步长，以像素为单位;
/// \param[in] uv_stride  I420格式中U和V平面的步长，以像素为单位;
/// \param[in] image      UYVY图形输出
/// \param[in] width      图像的宽
/// \param[in] height     图像的高
/////////////////////////////////////////////////////////////////////////////////// 
void transform_I420_to_uyvy(
    unsigned char *y_plane,
    unsigned char *u_plane,
    unsigned char *v_plane,
    int y_stride, int uv_stride,
    unsigned char *image,
    int width, int height);

//// des_buffer RGB24是按照 bgr存储的。 
int UYVY422ToRGB888(const unsigned char *src_buffer, int w, int h, const unsigned char *des_buffer);
#endif  ///< _HV_ENHANCE_H_


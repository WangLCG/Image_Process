#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include<math.h>

//for opencv
#include "cvhead.h"

using namespace cv;
using namespace std;

#define PI    (3.1415926)

//////////////////////////////////////////////////////////////////////////
//\brief 生成 N x N 高斯算子  
//\param[in] sigma 二维高斯函数的sigma因子  
//\param[in] N 算子的尺寸(NxN) 
//\param[out]  定点化的算子 (2^23倍)
//////////////////////////////////////////////////////////////////////////
void Create_NxN_GaussIq23(double sigma, int N, int* Gaus)
{
    double sum = 0;   //权重之和必须等于1，所以公式计算得到的矩阵的每个元素除以总和  
    double *tmp_res = new double[N*N];
    int center = N / 2;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            tmp_res[i * N + j] = (1 / (2 * PI * sigma*sigma))*exp(-((i - center)*(i - center) + (j - center)*(j - center)) / (2 * sigma*sigma));
            sum += tmp_res[i * N + j];
            //printf("%f\n", tmp_res[i * N + j]);
        }
        //printf("\n");
    }

   // printf("######   in   ####\n");

    //权重之和为1
    for (int i = 0; i < 9; ++i)
    {
        Gaus[i] = (tmp_res[i] / sum) * (1 << 23);
        //printf("%d\t",  Gaus[i]);
    }

    delete[] tmp_res;
}

#define   LEFT_SHIT_NUM  (12)
/*
    LoG = ( -2 * sigma^2 + x^2 +  y^2) * exp( -( x^2 + y^2 ) / (2 * sigma * sigma) ) / (2 * PI * sigma^6)
*/
//////////////////////////////////////////////////////////////////////////
//\brief 生成N x N 高斯-拉普拉斯边缘检测算子(LOG算子)  
//\param[in] sigma 二维高斯函数的sigma因子  
//\param[in] N LoG算子的尺寸(NxN) 
//\param[out]  定点化的LOG算子 (4096倍)
//////////////////////////////////////////////////////////////////////////
void Create_NxN_LoG(double sigma, int N, int* LoG)
{
    int R = N / 2;
    double sum = 0;   //用于确保权重之和等于1, 以便在均匀亮度区域不会检测到边缘
    double *tmp_res = new double[N*N];

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            double double_sigma = sigma * sigma;

            double r = (i - R) * (i - R) + (j - R) * (j - R);  //x^2 + y^2
            double res = exp(-r / (2 * double_sigma));       //exp(-(x^2 + y^2) / 2 * sigma * sigma)
            double r_sigm = r - 2 * double_sigma;

            res = res * r_sigm / ( 2 * PI * double_sigma *double_sigma * double_sigma);
            tmp_res[i * N + j] = res;
            //printf("%f\t", res);
            sum += tmp_res[i * N + j];
            //tmp_res[i * N + j] = res * 4096;
        }
        //printf("\n");
    }

    //归一化 
    for (int i = 0; i < N * N; ++i)
    {
        LoG[i] = (tmp_res[i] / sum)  * (1 << LEFT_SHIT_NUM);
    }

    delete[] tmp_res;

}

#define  USE_DEFAULE_LOG_MAT  (0)
//////////////////////////////////////////////////////////////////////////
//\brief 生成5 x 5 高斯-拉普拉斯边缘检测算子 
//\param[in] sigma 二维高斯函数的sigma因子  
//\param[in] N LoG算子的尺寸(NxN) 4096倍
//\param[out]  定点化的LOG算子 
//////////////////////////////////////////////////////////////////////////
void LOG_5X5_Filter(unsigned char* src_img, int width, int height, unsigned char* dst)
{
    if (!src_img || !dst
        || width < 0
        || height < 0)
    {
        printf("Invalid Params %s:%s:%d\n", __FILE__, __FUNCTION__, __LINE__);
        return ;
    }

#if USE_DEFAULE_LOG_MAT
    int LoG[25] =
    {
        0, 0,  1,  0, 0,
        0, 1,  2,  1, 0,
        1, 2, -16, 2, 1,
        0, 1,  2,  1, 0,
        0, 0,  1,  0, 0
    };
   /* {
        -2, -4, -4, -4, -2,
        -4,  0,  8,  0, -4,
        -4,  8, 24,  8, -4,
        -4,  0,  8,  0, -4,
        -2, -4, -4, -4, -2
    };*/
#else
    int N = 5;
    int* LoG = new int[N * N];
    memset(LoG, 0, sizeof(int) * N * N);
    double sigma = 0.8;
    Create_NxN_LoG(sigma, N, LoG);
#endif

    int i = 0, j = 0 , k = 0, l = 0;
    int tmp_data = 0;
    for (j = 2; j < height - 2; ++j)
    {
        int y_idx = j * width;
        for (i = 2; i < width - 2; ++i)
        {
            for (k = 0; k < 5; ++k)
            {
                int k_idx = j + k - 2 ;
                int kernel_idx = 5 * k;
                for (l = 0; l < 5; ++l)
                {
                    tmp_data += LoG[kernel_idx + l] * src_img[k_idx * width + l + i - 2];
                }
            }

#if USE_DEFAULE_LOG_MAT
            tmp_data = abs(tmp_data);
#else
            tmp_data = (tmp_data) >> LEFT_SHIT_NUM;
#endif
            dst[y_idx + i] = tmp_data > 255 ? 255 : tmp_data ;
            
            tmp_data = 0;
        }
    }

#if (USE_DEFAULE_LOG_MAT == 0)
    delete[] LoG;
#endif
}

//extern int gGaussianKernelIq23_3x3[9] =
//{
//    719239,  1017823,  719239,
//    1017823, 1440349, 1017823,
//    719239,  1017823,  719239
//};

//////////////////////////////////////////////////////////////////////////
//\brief 3 x 3 高斯滤波 
//\param[in] src 输入图像 
//\param[in/out] dst 处理输出
//\param[in] width 输入图像宽
//\param[in] height 输入图像高 
//\param[in] sigma 高斯核sigma因子  
//\return   
//////////////////////////////////////////////////////////////////////////
int Gaussian_3x3_KernelFilter(unsigned char *  src, unsigned char *  dst, int width, int height, double sigma)
{
    if (!src || !dst
        || width < 0
        || height < 0)
    {
        printf("Invalid Params %s:%s:%d\n", __FILE__, __FUNCTION__, __LINE__);
        return -1;
    }

    int *gGaussianKernelIq23_3x3 = new int[3 * 3];
    Create_NxN_GaussIq23(sigma, 3, gGaussianKernelIq23_3x3);

    {
        int i, j;
        int temp = 0;
        int halfIq23 = 0;//(1 << 22);//0.5 * (1 << 23)  
        unsigned char *   srcPtr = src;
        unsigned char *   srcPtr0 = NULL;
        unsigned char *   srcPtr1 = NULL;
        unsigned char *   srcPtr2 = NULL;
        unsigned char *   dstPtr = dst;

        //第一行///  
        srcPtr0 = srcPtr;
        srcPtr1 = srcPtr;
        srcPtr2 = srcPtr + width;
        //memcpy(dstPtr, srcPtr, width * sizeof(unsigned char));

        //第一个
        for (j = 1; j < width - 1; j++)
        {
            temp = srcPtr0[0] * gGaussianKernelIq23_3x3[0]
                + srcPtr0[1] * gGaussianKernelIq23_3x3[1]
                + srcPtr0[2] * gGaussianKernelIq23_3x3[2]
                + srcPtr1[0] * gGaussianKernelIq23_3x3[3]
                + srcPtr1[1] * gGaussianKernelIq23_3x3[4]
                + srcPtr1[2] * gGaussianKernelIq23_3x3[5]
                + srcPtr2[0] * gGaussianKernelIq23_3x3[6]
                + srcPtr2[1] * gGaussianKernelIq23_3x3[7]
                + srcPtr2[2] * gGaussianKernelIq23_3x3[8];

            temp = ((temp + halfIq23) >> 23) > 255 ? 255 : ((temp + halfIq23) >> 23);
            dstPtr[j] = temp;

            srcPtr1++;
            srcPtr2++;
        }
        dstPtr[0] = dstPtr[1];
        dstPtr[width - 1] = dstPtr[width - 2];

        for (i = 1; i < height - 1; i++)
        {
            int offset = i * width;
            srcPtr = src + offset;
            dstPtr = dst + offset;
            srcPtr0 = srcPtr - width;
            srcPtr1 = srcPtr;
            srcPtr2 = srcPtr + width;
            for (j = 1; j < width - 1; j++)
            {
                temp = srcPtr0[0] * gGaussianKernelIq23_3x3[0]
                    + srcPtr0[1] * gGaussianKernelIq23_3x3[1]
                    + srcPtr0[2] * gGaussianKernelIq23_3x3[2]
                    + srcPtr1[0] * gGaussianKernelIq23_3x3[3]
                    + srcPtr1[1] * gGaussianKernelIq23_3x3[4]
                    + srcPtr1[2] * gGaussianKernelIq23_3x3[5]
                    + srcPtr2[0] * gGaussianKernelIq23_3x3[6]
                    + srcPtr2[1] * gGaussianKernelIq23_3x3[7]
                    + srcPtr2[2] * gGaussianKernelIq23_3x3[8];

                temp = ((temp + halfIq23) >> 23) > 255 ? 255 : ((temp + halfIq23) >> 23);
                dstPtr[j] = temp;

                srcPtr0++;
                srcPtr1++;
                srcPtr2++;
            }
            dstPtr[0] = dstPtr[1];
            dstPtr[width - 1] = dstPtr[width - 2];
        }

        //最后一行///  
        srcPtr = src + width * (height - 1);
        dstPtr = dst + width * (height - 1);
        //memcpy(dstPtr, srcPtr, width * sizeof(unsigned char));
        srcPtr0 = srcPtr - width;
        srcPtr1 = srcPtr;
        srcPtr2 = srcPtr;
        for (j = 1; j < width - 1; j++)
        {
            temp = srcPtr0[0] * gGaussianKernelIq23_3x3[0]
                + srcPtr0[1] * gGaussianKernelIq23_3x3[1]
                + srcPtr0[2] * gGaussianKernelIq23_3x3[2]
                + srcPtr1[0] * gGaussianKernelIq23_3x3[3]
                + srcPtr1[1] * gGaussianKernelIq23_3x3[4]
                + srcPtr1[2] * gGaussianKernelIq23_3x3[5]
                + srcPtr2[0] * gGaussianKernelIq23_3x3[6]
                + srcPtr2[1] * gGaussianKernelIq23_3x3[7]
                + srcPtr2[2] * gGaussianKernelIq23_3x3[8];

            temp = ((temp + halfIq23) >> 23) > 255 ? 255 : ((temp + halfIq23) >> 23);
            dstPtr[j] = temp;

            srcPtr0++;
            srcPtr1++;
        }
        dstPtr[0] = dstPtr[1];
        dstPtr[width - 1] = dstPtr[width - 2];
    }

    delete[] gGaussianKernelIq23_3x3;

    return 0;
}

//////////////////////////////////////////////////////////////////////////
//\brief 3x3 DoG算子 
//\param[in] src 输入图像 
//\param[in/out] dst 处理输出
//\param[in] width 输入图像宽
//\param[in] height 输入图像高 
//\param[in] sigma1 高斯核sigma因子  
//\param[in] sigma2 高斯核sigma因子  
//\return   
//////////////////////////////////////////////////////////////////////////
int  DoG_3x3_Filter(unsigned char *src, unsigned char *dst, int width, int height, double sigma1, double sigma2)
{
    if (!src || !dst
        || width < 0
        || height < 0)
    {
        printf("Invalid Params %s:%s:%d\n", __FILE__, __FUNCTION__, __LINE__);
        return -1;
    }

    unsigned char * tmp = new unsigned char[width * height];
    memset(tmp, 0, sizeof(unsigned char) * width * height);

    Gaussian_3x3_KernelFilter(src,  dst,  width, height, sigma1);
    Gaussian_3x3_KernelFilter(src,  tmp,   width, height, sigma2);

    Mat Gaus_img1(height, width, CV_8UC1);
    Mat Gaus_img2(height, width, CV_8UC1);
    memcpy(Gaus_img1.data, dst,  width * height );
    memcpy(Gaus_img2.data, tmp, width * height);
    imshow("Gaus_img1", Gaus_img1);
    imshow("Gaus_img2", Gaus_img2);

    int size = sizeof(unsigned char) * width * height;
    int i = 0;
    for (i = 0; i < size; ++i)
    {

        dst[i] = (dst[i] - tmp[i]);
        //dst[i] = dst[i] > 255 ? 255 : dst[i];
    }

    delete[] tmp;
    return 0;
}

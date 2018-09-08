#include <iostream>
#include<boost/thread/thread.hpp>
#include <map>
#include <vector>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//for opencv
#include "cvhead.h"

#define IMAGE_ENHANCE  (1)
//直方图均衡化  
#define HistEquEnabl  (0)   

//3x3高斯滤波 
#define GaussianFilterEnable  (0)   

//BoxFilter
#define BoxFilterEnable  (0) 

//拉普拉斯锐化
#define LaplacEnable  (0) 

//gama矫正   
#define GammaCorreEnable (0)
//白平衡    
#define WhiteBalanceEnable (1)

using namespace std;
using namespace cv;

#ifdef __cplusplus
extern "C"
{
#endif

#include"enhance.h"

#ifdef __cplusplus
}
#endif

void hello()
{
    std::cout << "hello" << std::endl;
}

//仅做boost测试 
void boost_test()
{
    boost::thread thrd(&hello);
    thrd.join();
}

int main()
{
    string vName;
    cout << "input image : " << endl;
    cin >> vName;

    Mat src_img = imread(vName);
    if (src_img.empty())
    {
        cout << "read image fail\n" << endl;
        getchar();
        return 0;
    }

    Mat y_img;
    cvtColor(src_img, y_img, CV_BGR2GRAY);

#if  IMAGE_ENHANCE
    unsigned char* src = new unsigned char[352 * 288];
    unsigned char* dst = new unsigned char[352 * 288];
    int width = y_img.cols;
    int height = y_img.rows;
    int image_size = width * height;
#endif

#if WhiteBalanceEnable
    //RGB转uyvy 
    unsigned char * cif_uyvy_buf = new unsigned char[352 * 288 * 2];
    unsigned char * rgb_buf = new unsigned char[352 * 288 * 3];
    unsigned char* tmp_buf = new unsigned char[352 * 288 * 3];

    Mat yuv_img;
    Mat rgb_img(288, 352, CV_8UC3 );

    cvtColor(src_img, yuv_img, CV_BGR2YUV_I420);
    unsigned char * yuv_img_ptr = yuv_img.data;
    {
        //转换数据格式I420 -> UYVY   
        int src_img_w = src_img.cols;
        int src_img_h = src_img.rows;
        unsigned char *y_plane = yuv_img_ptr;
        unsigned char *u_plane = y_plane + src_img_w * src_img_h;
        unsigned char *v_plane = u_plane + (src_img_w * src_img_h >> 2);

        transform_I420_to_uyvy(y_plane, u_plane, v_plane, src_img_w, src_img_w / 2, cif_uyvy_buf, src_img_w, src_img_h);
        
        FILE* wfd1 = fopen("white_uyvy_process_in", "wb+");
        if (wfd1)
        {
            fwrite(cif_uyvy_buf, 1, 352 * 288 * 2, wfd1);
            fclose(wfd1);
        }

        uyvy_white_balance(cif_uyvy_buf, src_img_w, src_img_h, rgb_buf, tmp_buf);

        //UYVY422ToRGB888(cif_uyvy_buf, src_img_w, src_img_h, rgb_buf);
        memcpy(rgb_img.data, rgb_buf, src_img_w * src_img_h * 3);

        FILE* wfd = fopen("white_uyvy_process_out","wb+");
        if (wfd)
        {
            fwrite(cif_uyvy_buf,1, 352*288*2,wfd);
            fclose(wfd);
        }
        imshow("rgb_img", rgb_img);
    }

    delete[] cif_uyvy_buf;
    delete[] rgb_buf;
    delete[] tmp_buf;
#endif

//直方图均衡化 CIF大小
#if HistEquEnabl
    int * scratchBuf = new int[4 * 256];
    memcpy(src, y_img.data, width * height);
    Mat HistEQ_img(height, width, CV_8UC1);

    HistEqulizedInt(src, width,  height, dst, scratchBuf);
    memcpy(HistEQ_img.data, dst, width * height);
    delete[] scratchBuf;

    imshow("Y_img", y_img);
    imshow("HistEQ_img", HistEQ_img);
#endif

#if GaussianFilterEnable
    memcpy(src, y_img.data, width * height);
    Mat Gass_img(height, width, CV_8UC1);
    GaussianKernelFilterIq23(src, dst, width, height);
    memcpy(Gass_img.data, dst, width * height);

    imshow("Y_img", y_img);
    imshow("Gass_img", Gass_img);
#endif

#if BoxFilterEnable
    
    memcpy(src, y_img.data, image_size);
    Mat Box_img(height, width, CV_8UC1);

    unsigned char *scratchBuf = new unsigned char[352 * sizeof(short)];
    short* process_dst = new short[image_size];

     IFBoxFilter8UCHAR3x3_CIF(src, width, height , process_dst, scratchBuf);

     //处理结果还原为灰度图片  
     for (int i = 0; i < image_size; i++)
     {
         dst[i] = unsigned char (process_dst[i] / 9);
     }
     memcpy(Box_img.data, dst, image_size);

     delete[] scratchBuf;
     delete[] process_dst;
    imshow("Y_img", y_img);
    imshow("Box_img", Box_img);
#endif

#if LaplacEnable
    memcpy(src, y_img.data, image_size);
    Mat Laplac_img(height, width, CV_8UC1);

    //3x3核 
    GnPC_Laplacian8U( src,  dst, width, height, 5);
    memcpy(Laplac_img.data, dst, image_size);
    imshow("Y_img", y_img);
    imshow("Laplac_img", Laplac_img);
#endif

#if  GammaCorreEnable
    memcpy(src, y_img.data, image_size);
    Mat Gamma_img(height, width, CV_8UC1);

    float gamma = 1.8;
    int updat_lut = 1;
    GammaCorrectiom( src,  width,  height,  gamma, dst, updat_lut);

    memcpy(Gamma_img.data, dst, image_size);
    imshow("Y_img", y_img);
    imshow("Gamma_img", Gamma_img);
#endif

#if  IMAGE_ENHANCE
    delete[] src;
    delete[] dst;
#endif
    imshow("src_img", src_img);
    imshow("Y_img", y_img);
    waitKey(0);

    //getchar();
    return 0;
}
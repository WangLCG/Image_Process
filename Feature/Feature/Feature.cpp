#include <stdio.h>
#include <string.h>
#include <iostream>
#include "Feature.h"
#include<math.h>

//for opencv
#include "cvhead.h"

using namespace cv;
using namespace std;

#define PI   (3.1415926535897932384626433832795)
#define USER_EPSILON     (10e-9)

//原始LBP  
void LBP_3X3_Ori(unsigned char* src_img, int width, int height, unsigned char* lbp_img)
{
    memset(lbp_img, 0, width * height);
    int i = 0, j = 0;
    unsigned char code = 0;
    for (i = 1; i < height - 1; ++i)
    {
        int index_sub_1 = (i - 1) * width;
        int index = i * width;
        int index_plus_1 = (i + 1) * width;

        for (j = 1; j < width - 1; j++)
        {
            code = 0;
            unsigned char center = src_img[index + j];
            code |= (src_img[index_sub_1 + j - 1] >= center) << 7;
            code |= (src_img[index_sub_1 + j ] >= center) << 6;
            code |= (src_img[index_sub_1 + j + 1] >= center) << 5;
            code |= (src_img[index + j - 1] >= center) << 4;
            code |= (src_img[index + j + 1] >= center) << 3;
            code |= (src_img[index_plus_1 + j - 1] >= center) << 2;
            code |= (src_img[index_plus_1 + j] >= center) << 1;
            code |= (src_img[index_plus_1 + j + 1] >= center) << 0;

            lbp_img[index_sub_1 + j - 1] = code;
        }
    }
}

//旋转不变LBP
void LBP_3X3_Ratio(unsigned char* src_img, int width, int height, unsigned char* lbp_img)
{
    memset(lbp_img, 0, width * height);

    unsigned char RITable[256];
    int tmp = 0;
    int val = 0;

    //建立查找表
    for (int i = 0; i < 256; i++)
    {
        val = i;
        tmp = i;
        for (int j = 0; j < 8; ++j)
        {
            //旋转这一特征领域 
            if (tmp & 0x1)
            {
                tmp = tmp >> 1;
                tmp = tmp | 0x80;//1000 0000
            }
            else
            {
                tmp = tmp >> 1;
            }
            //printf("tmp = %d \t",tmp);
            //tmp = (i >> (8 - j) | (i << j));
            val = val > tmp ? tmp : val;  //找所有旋转模式中最小的
        }
        //printf("\n");
        RITable[i] = val;
    }

    int i = 0, j = 0;
    unsigned char code = 0;
    for (i = 1; i < height - 1; ++i)
    {
        int index_sub_1 = (i - 1) * width;
        int index = i * width;
        int index_plus_1 = (i + 1) * width;

        for (j = 1; j < width - 1; j++)
        {
            code = 0;
            unsigned char center = src_img[index + j];
            code |= (src_img[index_sub_1 + j - 1] >= center) << 7;
            code |= (src_img[index_sub_1 + j] >= center) << 6;
            code |= (src_img[index_sub_1 + j + 1] >= center) << 5;
            code |= (src_img[index + j - 1] >= center) << 4;
            code |= (src_img[index + j + 1] >= center) << 3;
            code |= (src_img[index_plus_1 + j - 1] >= center) << 2;
            code |= (src_img[index_plus_1 + j] >= center) << 1;
            code |= (src_img[index_plus_1 + j + 1] >= center) << 0;

            lbp_img[index_sub_1 + j - 1] = RITable[code];
        }
    }
}

//圆形LBP
void LBP_Circle(unsigned char* src_img, int width, int height, int radius, int neighbors, unsigned char* lbp_img)
{
    memset(lbp_img, 0, width * height);
    int n = 0;
    for (n = 0; n < neighbors; ++n)
    {
       //采样点坐标 
        float fx = (float)(radius * cos(2.0 * PI * n / neighbors));
        float fy = (float)(-radius * cos(2.0 * PI * n / neighbors));

        int floor_x = (int)floor(fx);  //下取整
        int floor_y = (int)floor(fy);
        int ceil_x = (int)ceil(fx);  //上取整
        int ceil_y = (int)ceil(fy);

        //小数部分
        float ty = fy - floor_y;
        float tx = fx - floor_x;

        //双线性插值权重设置
        float w1 = (1 - tx) * (1 - ty);
        float w2 = tx * (1 - ty);
        float w3 = (1 - tx) * ty;
        float w4 = tx * ty;

        int i = 0, j = 0;
        for (i = radius; i < height - radius; ++i)
        {
            int index_0 = (i + floor_y) * width;
            int index_1 = (i + ceil_y) * width;

            int src_idx = i * width;
            int dst_idx = (i - radius) * width;

            for (j = radius; j < width - radius; ++j)
            {
                float t = (float)(
                    w1 * src_img[index_0 + j + floor_x] +
                    w2 * src_img[index_0 + j + ceil_x] +
                    w3 * src_img[index_1 + j + floor_x] +
                    w4 * src_img[index_1 + j + ceil_x]
                    );

                //大于等于为1，然后移位 
                lbp_img[dst_idx + j - radius] += (((t > src_img[src_idx + j])
                    || (abs(t - src_img[src_idx + j]) < USER_EPSILON)) << n);
            }
        }
    }

    return ;
}

//获取8bit数内的跳变次数
int GetHopCount(unsigned char num)
{
    unsigned char flag[8] = { 0 };
    int ret = 0;
    int i = 7;

    for (; i >= 0; --i)
    {
        flag[i] = num & 1;
        num = num >> 1;
    }

    for (i = 0; i < 7; i++)
    {
        if (flag[i] != flag[i + 1])
            ++ret;
    }

    if(flag[0] != flag[7])
        ++ret;

    return ret;
}

//uniform LBP
void Unifom_LBP(unsigned char* src_img, int width, int height, unsigned char* lbp_img)
{
    unsigned char UTable[256] = { 0 };

    memset(lbp_img, 0, width*height);

    //建立查找表  
    unsigned char tmp = 1;  //共58种模式  
    for (int i = 0; i < 256; ++i)
    {
        if (GetHopCount(i) <= 2)
        {
            UTable[i] = tmp;
            ++tmp;
        }
    }
    //printf("tmp = %d\n", tmp);

    int i = 0, j = 0;
    unsigned char code = 0;
    for (i = 1; i < height - 1; ++i)
    {
        int index_sub_1 = (i - 1) * width;
        int index = i * width;
        int index_plus_1 = (i + 1) * width;

        for (j = 1; j < width - 1; j++)
        {
            code = 0;
            unsigned char center = src_img[index + j];
            code |= (src_img[index_sub_1 + j - 1] >= center) << 7;
            code |= (src_img[index_sub_1 + j] >= center) << 6;
            code |= (src_img[index_sub_1 + j + 1] >= center) << 5;
            code |= (src_img[index + j - 1] >= center) << 4;
            code |= (src_img[index + j + 1] >= center) << 3;
            code |= (src_img[index_plus_1 + j - 1] >= center) << 2;
            code |= (src_img[index_plus_1 + j] >= center) << 1;
            code |= (src_img[index_plus_1 + j + 1] >= center) << 0;

            lbp_img[index_sub_1 + j - 1] = UTable[code];
        }
    }

}

//MB_LBP特征 
void MB_LBP(unsigned char* src_img, int width, int height, int Block_size, unsigned char* lbp_img)
{
    int cellsize = Block_size / 3;
    int offset = cellsize / 2;

    unsigned char *tmp_img = new unsigned char[width * height];
    memset(tmp_img, 0 , width * height);

    //计算Blok块内子领域的平均灰度值
    for (int i = offset; i < height - offset; ++i)
    {
        int index = i * width;
        int dst_index = (i - offset) * width ;
        for (int j = offset; j < width - offset; ++j)
        {
            int  tmp = 0;
            for (int m = -offset; m < offset + 1; ++m)
            {
                int index_1 = m * width;
                for (int n = -offset; n < offset + 1; ++n)
                {
                    tmp += src_img[index + index_1 + j + n];
                }
            }

            tmp /= (cellsize * cellsize);
            tmp_img[dst_index + j - offset] = tmp;
        }
    }

    LBP_3X3_Ori(tmp_img, width, height, lbp_img);
    
    delete[] tmp_img;
}

int LBP( )
{

    VideoCapture vc;
    string vName;

    cout << "input video : " << endl;
    cin >> vName;
    //vName = "E:\\video\\test.avi";

    vc.open(vName);
    if (!vc.isOpened())
    {
        cout << "Open file failed!" << endl;
        return 0;
    } 

    Mat frame,YFrame;

    vc >> frame;
    int width = frame.cols;
    int height = frame.rows;
    Mat dst_img(height, width, CV_8UC1);
    Mat dst_ratio_img(height, width, CV_8UC1);
    Mat dst_circle_img(height, width, CV_8UC1);
    Mat dst_unifom_img(height, width, CV_8UC1);

    Mat dst_MB_img(height, width, CV_8UC1);

    for (int cnt = 0; ; cnt++)
    {
        vc >> frame;
        //printf("cnt= %d\n", cnt);
        if (frame.empty())
        {
            cout << "video end.";
            break;
            vc.release();
            vc.open(vName);
            continue;
        }

        cvtColor(frame, YFrame, CV_BGR2GRAY);

        int width  = YFrame.cols;
        int height = YFrame.rows;
         
        LBP_3X3_Ori(YFrame.data, width, height, dst_img.data);
        LBP_3X3_Ratio(YFrame.data, width, height, dst_ratio_img.data);

        int radius = 1;  //采样半径1  
        int neighbors = 8;  //8个采样点  
        LBP_Circle(YFrame.data,  width,  height,  radius,  neighbors, dst_circle_img.data);
        
        Unifom_LBP(YFrame.data, width, height, dst_unifom_img.data);

        int Block_size = 9;
        MB_LBP(YFrame.data, width, height, Block_size, dst_MB_img.data);

        //使用uniform lbp
        int numPatterns = 59;
        int grid_x = 9;
        int grid_y = 9;
        bool normed = true;
        getLBPH(dst_unifom_img,  numPatterns,  grid_x,  grid_y,  normed);

        imshow("YFrame", YFrame);
        imshow("lbp", dst_img);
        imshow("lbp_ratio", dst_ratio_img);
        imshow("circle_img", dst_circle_img);
        imshow("unifom_img", dst_unifom_img);
        imshow("mb_lbp", dst_MB_img);

        cvWaitKey(1);
    }

    cvWaitKey(0);

    getchar();
    return 0;

}

//计算LBP特征图像的直方图LBPH   
Mat getLBPH(Mat src, int numPatterns, int grid_x, int grid_y, bool normed)
{
    //Mat src = _src.getMat();   
    int width = src.cols / grid_x;
    int height = src.rows / grid_y;
    //定义LBPH的行和列，grid_x*grid_y表示将图像分割成这么些块，numPatterns表示LBP值的模式种类    
    Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
    if (src.empty())
    {
        return result.reshape(1, 1);
    }

    int resultRowIndex = 0;
    //对图像进行分割，分割成大小为grid_x*grid_y的块，grid_x，grid_y默认为8    
    for (int i = 0; i < grid_x; i++)
    {
        for (int j = 0; j < grid_y; j++)
        {
            //图像分块    
            Mat src_cell = Mat(src, Range(i * height, (i + 1) * height), Range(j * width, (j + 1) * width));
            //计算直方图    
            Mat hist_cell = getLocalRegionLBPH(src_cell, 0, (numPatterns - 1), normed);
            //将直方图放到result中     
            Mat rowResult = result.row(resultRowIndex);
            hist_cell.reshape(1, 1).convertTo(rowResult, CV_32FC1);
            resultRowIndex++;
        }
    }

    return result.reshape(1, 1);
}

//计算一个LBP特征图像块的直方图   
Mat getLocalRegionLBPH(const Mat& src, int minValue, int maxValue, bool normed)
{
    //定义存储直方图的矩阵     
    Mat result;
    //计算得到直方图bin的数目，直方图数组的大小     
    int histSize = maxValue - minValue + 1;
    //定义直方图每一维的bin的变化范围     
    float range[] = { static_cast<float>(minValue),static_cast<float>(maxValue + 1) };
    //定义直方图所有bin的变化范围
    const float* ranges = { range };
    //计算直方图，src是要计算直方图的图像，1是要计算直方图的图像数目，0是计算直方图所用的图像的通道序号，从0索引
    //Mat()是要用的掩模，result为输出的直方图，1为输出的直方图的维度，histSize直方图在每一维的变化范围
    //ranges，所有直方图的变化范围（起点和终点）
    calcHist(&src, 1, 0, Mat(), result, 1, &histSize, &ranges, true, false);
    //归一化
    if (normed)
    {
        result /= (int)src.total();
    }
    //结果表示成只有1行的矩阵
    return result.reshape(1, 1);
}
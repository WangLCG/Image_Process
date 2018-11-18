#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//for opencv
#include "cvhead.h"

using namespace cv;
using namespace std;

void Average_Mean_Bg_Fg(unsigned char* src_img , int width, int height, unsigned char* bg_mod, unsigned char* dst)
{
    int Thre = 25;
    double alp = 0.65;  //alp 越大对背景变化的适应速度越快  

    for (int i = 0; i < height; ++i)
    {
        int index = i * width;
        for (int j = 0; j < width; ++j)
        {
            unsigned char tmp = abs(src_img[index + j] - bg_mod[index + j]);
            dst[index + j] = (tmp > Thre) ? 255 : 0;

            //背景更新  不做背景阈值、方差更新(因为太麻烦了)
            bg_mod[index + j] = alp * src_img[index + j] + (1 - alp) * bg_mod[index + j];
            bg_mod[index + j] = bg_mod[index + j] > 255 ? 255 : bg_mod[index + j];
        }
    }

    return;
}

void Average_Mean_SBG_Fg(unsigned char* src_img, int width, int height,
                    unsigned char* bg_mod, unsigned char* sbg_mod, unsigned char* dst)
{
    int Thre  = 25;
    int Thres = 65;
    double alp = 0.65;  //alp 越大对背景变化的适应速度越快  

    for (int i = 0; i < height; ++i)
    {
        int index = i * width;
        for (int j = 0; j < width; ++j)
        {
            unsigned char tmp = abs(src_img[index + j] - bg_mod[index + j]);
            unsigned char tmps = abs(src_img[index + j] - sbg_mod[index + j]);

            dst[index + j] = ( (tmps > Thres) && (tmps > Thre)) ? 255 : 0;  //outputs == 1 and output == 1

            //更新辅助背景
            if (tmp < Thre)  //output == 0 更新辅助背景
            {
                sbg_mod[index + j] = src_img[index + j];
            }

            //背景更新  不做背景阈值、方差更新(因为太麻烦了)
            bg_mod[index + j] = alp * src_img[index + j] + (1 - alp) * bg_mod[index + j];
            bg_mod[index + j] = bg_mod[index + j] > 255 ? 255 : bg_mod[index + j];

            
        }
    }

    return;
}

//更新单高斯背景建模的标准差及方差矩阵
void Update_var_and_std(float varInit, int index, float *var_img, float *std_img)
{
    var_img[index] = varInit;
    std_img[index] = sqrt(varInit);
}

void Gauss_BGM(unsigned char* src_img, int width, int height, 
            unsigned char* bg_img, float* var_img, 
            float *std_img,  unsigned char* dst)
{
    float varInit = 0.0;  //初始化方差
    float alp = 0.02;     //背景建模alp 越小对背景变化的适应速度越慢(背景更新越慢)  
    int lamda = 3;        //背景更新参数

    for (int i = 0; i < height; ++i)
    {
        int index = i * width;
        for (int j = 0; j < width; ++j)
        {
            int tmp = src_img[index + j] - bg_img[index + j];

            //|I-U|<lamda * stdInit时认为是背景，进行背景更新  
            if (abs(tmp) < lamda * std_img[index+j])
            {
                //更新背景模型，既是期望
                bg_img[index + j] = (1 - alp) * bg_img[index + j] + alp * src_img[index + j];

                //更新方差、标准差
               varInit = (1 - alp) * var_img[index + j] + alp * tmp *tmp;
               Update_var_and_std(varInit, index + j, var_img, std_img);

                dst[index + j] = 0;
            }
            else
            {
                dst[index + j] = 255;
            }
        }
    }
}
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//for opencv
#include "cvhead.h"

using namespace cv;
using namespace std;

void Average_Mean_Bg_Fg(unsigned char* src_img, int width, int height, unsigned char* bg_mod, unsigned char* dst);

//均值背景建模改进，添加辅助背景SBG
void Average_Mean_SBG_Fg(unsigned char* src_img, int width, int height,
    unsigned char* bg_mod, unsigned char* sbg_mod, unsigned char* dst);

//单高斯背景建模
void Gauss_BGM(unsigned char* src_img, int width, int height,
    unsigned char* bg_img, float* var_img,
    float *std_img, unsigned char* dst);
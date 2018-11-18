#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//for opencv
#include "cvhead.h"
#include "BG_Mod.h"

using namespace cv;
using namespace std;

int main()
{
    VideoCapture vc;
    string vName;

    //cout << "input video : " << endl;
    //cin >> vName;
    vName = "D:\\GitSource\\Image_Process\\768x576.avi";

    vc.open(vName);
    if (!vc.isOpened())
    {
        cout << "Open file failed!" << endl;
        return -1;
    }

    Mat frame, YFrame;

    vc >> frame;
    int width = frame.cols;
    int height = frame.rows;
    Mat dst_img(height, width, CV_8UC1);
    Mat dst_sbg_img(height, width, CV_8UC1);
    Mat dst_sgm_img(height, width, CV_8UC1);

    unsigned char *bg_mod  = new unsigned char[width * height];   //背景模型
    unsigned char *sbg_mod = new unsigned char[width * height];  //辅助背景模型

    unsigned char *sgm_img = new unsigned char[width * height];      //单高斯背景模型
    float *sgm_var_img = new float[width * height];  //单高斯方差
    float *sgm_std_img = new float[width * height];  //单高斯标准

    for (int cnt = 0; ; cnt++)
    {
        vc >> frame;
        //printf("cnt= %d\n", cnt);
        if (frame.empty())
        {
            cout << "video end.";
            break;
        }

        cvtColor(frame, YFrame, CV_BGR2GRAY);

        if (cnt == 0)
        {
            //第一张图作为背景模型，这样就不用求均值，为了计算方便才这么做的
            memcpy(bg_mod, YFrame.data, width * height);
            memcpy(sbg_mod, YFrame.data, width * height);

            memcpy(sgm_img, YFrame.data, width * height);

            float stdInit = 20;                  //初始化标准差
            float varInit = stdInit * stdInit;  //初始化方差
            for (int i = 0; i < height; ++i)
            {
                int index = i * width;
                for (int j = 0 ; j < width; ++j)
                {
                    sgm_var_img[index + j] = varInit;
                    sgm_std_img[index + j] = stdInit;
                }
            }
            continue;
        }
        else
        {
            Average_Mean_Bg_Fg(YFrame.data, width, height, bg_mod, dst_img.data);

            Average_Mean_SBG_Fg(YFrame.data, width, height, bg_mod, sbg_mod, dst_sbg_img.data);

            Gauss_BGM(YFrame.data,  width,  height, sgm_img, sgm_var_img, sgm_std_img, dst_sgm_img.data);
        }

        imshow("YFrame", YFrame);
        imshow("aver_bg", dst_img);
        imshow("sbg_img", dst_sbg_img);
        imshow("sgm_img", dst_sgm_img);

        cvWaitKey(1);
    }

    delete[] bg_mod;
    delete[] sbg_mod;
    
    delete[] sgm_img;
    delete[] sgm_std_img;
    delete[] sgm_var_img;

    vc.release();
    getchar();
    return 0;

}

// Filter.cpp : 定义控制台应用程序的入口点。
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//for opencv
#include "cvhead.h"

using namespace cv;
using namespace std;

#ifdef __cplusplus
extern "C"
{
#endif

#include"Filter.h"

#ifdef __cplusplus
}
#endif

int main()
{

#if 0
    double sigma = 1.0;
    int N = 3;
    int* Gaus_s = new int[N * N];

    Create_NxN_GaussIq23( sigma, N, Gaus_s);

    delete[] Gaus_s;

    getchar();
    return 0;

#endif
    //读一张图片进来，转化为灰度图  
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
    imshow("src", y_img);
    
    int width = y_img.cols;
    int height = y_img.rows;
    Mat dst_img(height, width, CV_8UC1);

    LOG_5X5_Filter(y_img.data, width, height, dst_img.data);

    imshow("LoG", dst_img);

    double sigma1 = 0.3;
    double sigma2 = 1.0;
    Mat Gaus_img(height, width, CV_8UC1);
    DoG_3x3_Filter(y_img.data, Gaus_img.data, width, height, sigma1, sigma2);
    imshow("Gaus", Gaus_img);

    waitKey(0);

    printf("Enter once to exit \n");
    getchar();
    //system("pause");
    return 0;
}


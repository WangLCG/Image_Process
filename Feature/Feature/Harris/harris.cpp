
#include <iostream>  
#include <cmath>
#include "harris.h"

using namespace cv;
using namespace std;


int harris_main()
{
    const Mat srcImage = imread("3.jpg");
    if (!srcImage.data)
    {
        printf("could not load image...\n");
        return -1;
    }

    imshow("srcImage", srcImage);
    Mat srcGray;
    ConvertRGB2GRAY(srcImage, srcGray);
    //cvtColor(srcImage, srcGray, CV_BGR2GRAY);

    Mat imageSobelX;
    Mat imageSobelY;
    Mat resultImage;
    Mat_<float> imageSobelXX;
    Mat_<float> imageSobelYY;
    Mat_<float> imageSobelXY;
    Mat_<float> GaussianXX;
    Mat_<float> GaussianYY;
    Mat_<float> GaussianXY;
    Mat_<float> HarrisRespond;

    //计算Soble的XY梯度
    SobelGradDirction(srcGray, imageSobelX, imageSobelY);
    //计算X方向的梯度的平方
    SobelXX(imageSobelX, imageSobelXX);
    SobelYY(imageSobelY, imageSobelYY);
    SobelXY(imageSobelX, imageSobelY, imageSobelXY);
    //计算高斯模糊XX YY XY
    MyGaussianBlur(imageSobelXX, GaussianXX, 3);
    MyGaussianBlur(imageSobelYY, GaussianYY, 3);
    MyGaussianBlur(imageSobelXY, GaussianXY, 3);
    harrisResponse(GaussianXX, GaussianYY, GaussianXY, HarrisRespond, 0.05);
    LocalMaxValue(HarrisRespond, srcGray, resultImage, 3);

#if 0
    //opencv harris
    Mat corner;
    cornerHarris(srcGray, corner, 3, 3, 0.04);
    threshold(corner, corner,0.004, 255, THRESH_BINARY);
    imshow("CV_corner", corner);
#endif
    //imshow("imageSobelX", imageSobelX);
    //imshow("imageSobelY", imageSobelY);
    imshow("HarrisRespond", HarrisRespond);
    imshow("resultImage", resultImage);

    return 0;
}

int harris_main_CStyle()
{
    printf("In function: %s\n", __FUNCTION__);

    const Mat srcImage = imread("3.jpg");
    if (!srcImage.data)
    {
        printf("could not load image...\n");
        return -1;
    }

    imshow("C_srcImage", srcImage);
    Mat srcGray;
    ConvertRGB2GRAY(srcImage, srcGray);

    int width = srcGray.cols;
    int height = srcGray.rows;

    Mat resultImage(height, width, CV_8UC1);
    resultImage = srcGray.clone();
    //Mat resultImage = Mat::zeros(srcGray.size(), CV_8UC1);

    Mat imageSobelX(height, width, CV_32FC1);
    Mat imageSobelY(height, width, CV_32FC1);
   
    Mat imageSobelXX(height, width, CV_32FC1);
    Mat imageSobelYY(height, width, CV_32FC1);
    Mat imageSobelXY(height, width, CV_32FC1);
    Mat GaussianXX(height, width, CV_32FC1);
    Mat GaussianYY(height, width, CV_32FC1);
    Mat GaussianXY(height, width, CV_32FC1);
    Mat HarrisRespond(height, width, CV_32FC1);

    //计算Soble的XY梯度
    SobelGradDirction_CStyle(srcGray.data, width, height, (float*)imageSobelX.data, (float*)imageSobelY.data);

    //convertScaleAbs(imageSobelX, imageSobelX);
    //convertScaleAbs(imageSobelX, imageSobelY);
    //计算X方向的梯度的平方
    SobelXX_CStyle((float*)imageSobelX.data, width, height,  (float*)imageSobelXX.data);
    SobelYY_CStyle((float*)imageSobelY.data, width, height, (float*)imageSobelYY.data);
    SobelXY_CStyle((float*)imageSobelX.data, (float*)imageSobelY.data, width, height, (float*)imageSobelXY.data);
    
    //计算高斯模糊XX YY XY
    MyGaussianBlur_CStyle((float*)imageSobelXX.data, width, height, (float*)GaussianXX.data, 3, 1);
    MyGaussianBlur_CStyle((float*)imageSobelYY.data, width, height, (float*)GaussianYY.data, 3, 1);
    MyGaussianBlur_CStyle((float*)imageSobelXY.data, width, height, (float*)GaussianXY.data, 3, 1);
    harrisResponse_CStyle((float*)GaussianXX.data, (float*)GaussianYY.data, (float*)GaussianXY.data, (float*)HarrisRespond.data, width, height, 0.05);
    Local_3x3_MaxValue_CStyle((float*)HarrisRespond.data, srcGray.data, resultImage.data, width, height);

    //convertScaleAbs(imageSobelX, imageSobelX);
    //convertScaleAbs(imageSobelX, imageSobelY);
    //convertScaleAbs(imageSobelX, HarrisRespond);


    //imshow("C_imageSobelX", imageSobelX);
    //imshow("C_imageSobelY", imageSobelY);
    imshow("C_HarrisRespond", HarrisRespond);
    imshow("C_resultImage", resultImage);

    imwrite("./C_HarrisRespond.jpg", HarrisRespond);
    imwrite("./C_resultImage.jpg", resultImage);

    waitKey(0);
    return 0;

}

void ConvertRGB2GRAY(const Mat &image, Mat &imageGray)
{
    if (!image.data || image.channels() != 3)
    {
        return;
    }
    //创建一张单通道的灰度图像
    imageGray = Mat::zeros(image.size(), CV_8UC1);
    //取出存储图像像素的数组的指针
    uchar *pointImage = image.data;
    uchar *pointImageGray = imageGray.data;
    //取出图像每行所占的字节数
    size_t stepImage = image.step;
    size_t stepImageGray = imageGray.step;
    for (int i = 0; i < imageGray.rows; i++)
    {
        for (int j = 0; j < imageGray.cols; j++)
        {
            pointImageGray[i*stepImageGray + j] = (uchar)(0.114*pointImage[i*stepImage + 3 * j] + 0.587*pointImage[i*stepImage + 3 * j + 1] + 0.299*pointImage[i*stepImage + 3 * j + 2]);
        }
    }
}


//存储梯度膜长
void SobelGradDirction(Mat &imageSource, Mat &imageSobelX, Mat &imageSobelY)
{
    imageSobelX = Mat::zeros(imageSource.size(), CV_32SC1);
    imageSobelY = Mat::zeros(imageSource.size(), CV_32SC1);
    //取出原图和X和Y梯度图的数组的首地址
    uchar *P = imageSource.data;
    uchar *PX = imageSobelX.data;
    uchar *PY = imageSobelY.data;

    //取出每行所占据的字节数
    int step = imageSource.step;
    int stepXY = imageSobelX.step;

    int index = 0;//梯度方向角的索引
    for (int i = 1; i < imageSource.rows - 1; ++i)
    {
        for (int j = 1; j < imageSource.cols - 1; ++j)
        {
            //通过指针遍历图像上每一个像素   
            double gradY = P[(i + 1)*step + j - 1] + P[(i + 1)*step + j] * 2 + P[(i + 1)*step + j + 1] - P[(i - 1)*step + j - 1] - P[(i - 1)*step + j] * 2 - P[(i - 1)*step + j + 1];
            PY[i*stepXY + j*(stepXY / step)] = abs(gradY);

            double gradX = P[(i - 1)*step + j + 1] + P[i*step + j + 1] * 2 + P[(i + 1)*step + j + 1] - P[(i - 1)*step + j - 1] - P[i*step + j - 1] * 2 - P[(i + 1)*step + j - 1];
            PX[i*stepXY + j*(stepXY / step)] = abs(gradX);
        }
    }
    //将梯度数组转换成8位无符号整型
    convertScaleAbs(imageSobelX, imageSobelX);
    convertScaleAbs(imageSobelY, imageSobelY);
}

void SobelGradDirction_CStyle(unsigned char *imageSource, int width, int height,  float* imageSobelX,  float* imageSobelY)
{
    //取出原图和X和Y梯度图的数组的首地址
    unsigned char *P = imageSource;
    float *PX = imageSobelX;
    float *PY = imageSobelY;

    //取出每行所占据的字节数
    int step = width;

    for (int i = 1; i < height - 1; ++i)
    {
        for (int j = 1; j < width - 1; ++j)
        {
            //通过指针遍历图像上每一个像素   
            float gradY = P[(i + 1) * step + j - 1] + P[(i + 1) * step + j] * 2 + P[(i + 1)*step + j + 1]
                        - P[(i - 1)*step + j - 1] - P[(i - 1)*step + j] * 2 - P[(i - 1)*step + j + 1];
            PY[i * step + j] = abs(gradY);

            float gradX = P[(i - 1) * step + j + 1] + P[i * step + j + 1] * 2 + P[(i + 1)*step + j + 1]
                        - P[(i - 1)*step + j - 1] - P[i*step + j - 1] * 2 - P[(i + 1)*step + j - 1];
            PX[i * step + j] = abs(gradX);
        }
    }
}

void SobelXX(const Mat imageGradX, Mat_<float> &SobelAmpXX)
{
    SobelAmpXX = Mat_<float>(imageGradX.size(), CV_32FC1);
    for (int i = 0; i < SobelAmpXX.rows; i++)
    {
        for (int j = 0; j < SobelAmpXX.cols; j++)
        {
            SobelAmpXX.at<float>(i, j) = imageGradX.at<uchar>(i, j)*imageGradX.at<uchar>(i, j);
        }
    }
    //convertScaleAbs(SobelAmpXX, SobelAmpXX);
}

void SobelXX_CStyle(float* imageGradX, int width, int height, float* SobelAmpXX)
{
    
    for (int i = 0; i < height; i++)
    {
        int index = i * width;
        for (int j = 0; j < width; j++)
        {
            SobelAmpXX[index + j] = imageGradX[index + j] * imageGradX[index + j];
        }
    }
}

void SobelYY(const Mat imageGradY, Mat_<float> &SobelAmpYY)
{
    SobelAmpYY = Mat_<float>(imageGradY.size(), CV_32FC1);
    for (int i = 0; i < SobelAmpYY.rows; i++)
    {
        for (int j = 0; j < SobelAmpYY.cols; j++)
        {
            SobelAmpYY.at<float>(i, j) = imageGradY.at<uchar>(i, j)*imageGradY.at<uchar>(i, j);
        }
    }
    //convertScaleAbs(SobelAmpYY, SobelAmpYY);
}

void SobelYY_CStyle(float* imageGradY, int width, int height, float* SobelAmpYY)
{

    for (int i = 0; i < height; i++)
    {
        int index = i * width;
        for (int j = 0; j < width; j++)
        {
            SobelAmpYY[index + j] = imageGradY[index + j] * imageGradY[index + j];
        }
    }
}

void SobelXY(const Mat imageGradX, const Mat imageGradY, Mat_<float> &SobelAmpXY)
{
    SobelAmpXY = Mat_<float>(imageGradX.size(), CV_32FC1);
    for (int i = 0; i < SobelAmpXY.rows; i++)
    {
        for (int j = 0; j < SobelAmpXY.cols; j++)
        {
            SobelAmpXY.at<float>(i, j) = imageGradX.at<uchar>(i, j)*imageGradY.at<uchar>(i, j);
        }
    }
    //convertScaleAbs(SobelAmpXY, SobelAmpXY);
}

void SobelXY_CStyle(float* imageGradX, float* imageGradY, int width, int height, float* SobelAmpXY)
{

    for (int i = 0; i < height; i++)
    {
        int index = i * width;
        for (int j = 0; j < width; j++)
        {
            SobelAmpXY[index + j] = imageGradX[index + j] * imageGradY[index + j];
        }
    }
}


//计算一维高斯的权值数组
double *getOneGuassionArray(int size, double sigma)
{
    double sum = 0.0;
    //定义高斯核半径
    int kerR = size / 2;

    //建立一个size大小的动态一维数组
    double *arr = new double[size];
    for (int i = 0; i < size; i++)
    {

        // 高斯函数前的常数可以不用计算，会在归一化的过程中给消去
        arr[i] = exp(-((i - kerR)*(i - kerR)) / (2 * sigma*sigma));
        sum += arr[i];//将所有的值进行相加

    }
    //进行归一化 
    for (int i = 0; i < size; i++)
    {
        arr[i] /= sum;
        cout << arr[i] << endl;
    }
    return arr;
}

void MyGaussianBlur(Mat_<float> &srcImage, Mat_<float> &dst, int size)
{
    CV_Assert(srcImage.channels() == 1 || srcImage.channels() == 3); // 只处理单通道或者三通道图像
    int kerR = size / 2;
    dst = srcImage.clone();
    int channels = dst.channels();
    double* arr;
    arr = getOneGuassionArray(size, 1);//先求出高斯数组

                                       //遍历图像 水平方向的卷积
    for (int i = kerR; i < dst.rows - kerR; i++)
    {
        for (int j = kerR; j < dst.cols - kerR; j++)
        {
            float GuassionSum[3] = { 0 };
            //滑窗搜索完成高斯核平滑
            for (int k = -kerR; k <= kerR; k++)
            {

                if (channels == 1)//如果只是单通道
                {
                    GuassionSum[0] += arr[kerR + k] * srcImage.at<float>(i, j + k);//行不变，列变换，先做水平方向的卷积
                }
                else if (channels == 3)//如果是三通道的情况
                {
                    Vec3f bgr = srcImage.at<Vec3f>(i, j + k);
                    auto a = arr[kerR + k];
                    GuassionSum[0] += a*bgr[0];
                    GuassionSum[1] += a*bgr[1];
                    GuassionSum[2] += a*bgr[2];
                }
            }
            for (int k = 0; k < channels; k++)
            {
                if (GuassionSum[k] < 0)
                    GuassionSum[k] = 0;
                else if (GuassionSum[k] > 255)
                    GuassionSum[k] = 255;
            }
            if (channels == 1)
                dst.at<float>(i, j) = static_cast<float>(GuassionSum[0]);
            else if (channels == 3)
            {
                Vec3f bgr = { static_cast<float>(GuassionSum[0]), static_cast<float>(GuassionSum[1]), static_cast<float>(GuassionSum[2]) };
                dst.at<Vec3f>(i, j) = bgr;
            }

        }
    }

    //竖直方向
    for (int i = kerR; i < dst.rows - kerR; i++)
    {
        for (int j = kerR; j < dst.cols - kerR; j++)
        {
            float GuassionSum[3] = { 0 };
            //滑窗搜索完成高斯核平滑
            for (int k = -kerR; k <= kerR; k++)
            {

                if (channels == 1)//如果只是单通道
                {
                    GuassionSum[0] += arr[kerR + k] * srcImage.at<float>(i + k, j);//行变，列不换，再做竖直方向的卷积
                }
                else if (channels == 3)//如果是三通道的情况
                {
                    Vec3f bgr = srcImage.at<Vec3f>(i + k, j);
                    auto a = arr[kerR + k];
                    GuassionSum[0] += a*bgr[0];
                    GuassionSum[1] += a*bgr[1];
                    GuassionSum[2] += a*bgr[2];
                }
            }
            for (int k = 0; k < channels; k++)
            {
                if (GuassionSum[k] < 0)
                    GuassionSum[k] = 0;
                else if (GuassionSum[k] > 255)
                    GuassionSum[k] = 255;
            }
            if (channels == 1)
                dst.at<float>(i, j) = static_cast<float>(GuassionSum[0]);
            else if (channels == 3)
            {
                Vec3f bgr = { static_cast<float>(GuassionSum[0]), static_cast<float>(GuassionSum[1]), static_cast<float>(GuassionSum[2]) };
                dst.at<Vec3f>(i, j) = bgr;
            }

        }
    }
    delete[] arr;
}

//单通道的
int MyGaussianBlur_CStyle(float *srcImage, int width, int height, float* dst , int k_size, int channels)
{
    if (channels != 1 ) // 只处理单通道或者三通道图像
    {
        printf("[%s]: Invalid Params\n", __FUNCTION__);
        return -1;
    }

    int kerR = k_size / 2;
    double* arr;
    arr = getOneGuassionArray(k_size, 1);//先求出高斯数组

                                       //遍历图像 水平方向的卷积
    for (int i = kerR; i < height - kerR; i++)
    {
        int index = i * width;
        for (int j = kerR; j < width - kerR; j++)
        {
            float GuassionSum =  0 ;
            //滑窗搜索完成高斯核平滑
            for (int k = -kerR; k <= kerR; k++)
            {
                GuassionSum += arr[kerR + k] * srcImage[index + j + k];//行不变，列变换，先做水平方向的卷积
            }
            
            if (GuassionSum < 0)
            {
                GuassionSum = 0;
            }
            else if (GuassionSum > 255)
            {
                GuassionSum = 255;
            }


            dst[index + j] = (float)GuassionSum;
        }
    }

    //竖直方向
    for (int i = kerR; i < height - kerR; i++)
    {
        int index = i * width;
        for (int j = kerR; j < width - kerR; j++)
        {
            float GuassionSum =  0 ;
            //滑窗搜索完成高斯核平滑
            for (int k = -kerR; k <= kerR; k++)
            {
              GuassionSum += arr[kerR + k] * srcImage[index + k * width + j];//行变，列不换，再做竖直方向的卷积
            }

            if (GuassionSum < 0)
            {
                GuassionSum = 0;
            }
            else if (GuassionSum > 255)
            {
                GuassionSum = 255;
            }

            dst[index + j] = (float)GuassionSum;
        }
    }
    delete[] arr;
}


void harrisResponse(Mat_<float> &GaussXX, Mat_<float> &GaussYY, Mat_<float> &GaussXY, Mat_<float> &resultData, float k)
{
    //创建一张响应函数输出的矩阵
    resultData = Mat_<float>(GaussXX.size(), CV_32FC1);
    for (int i = 0; i < resultData.rows; i++)
    {
        for (int j = 0; j < resultData.cols; j++)
        {
            float a = GaussXX.at<float>(i, j);
            float b = GaussYY.at<float>(i, j);
            float c = GaussXY.at<float>(i, j);
            resultData.at<float>(i, j) = a*b - c*c - k*(a + b)*(a + b);
        }
    }
}

void harrisResponse_CStyle(float* GaussXX, float* GaussYY, float* GaussXY, float* resultData, int width, int height, float k)
{
    for (int i = 0; i < height; i++)
    {
        int index = i * width;
        for (int j = 0; j < width; j++)
        {
            float a = GaussXX[index+j];
            float b = GaussYY[index + j];
            float c = GaussXY[index + j];
            
            //2x2矩阵特征值  
            float r1 = (float) ( (a + b - sqrt((a - b)*(a - b) + 4 * c*c) ) / 2.0);

#ifdef ENABLE_SHI_TOMASI_CONNER_POINT
            resultData[index + j] = r1;
#else
            float r2 = (float) ( (a + b + sqrt((a - b)*(a - b) + 4 * c*c) ) / 2.0);
            resultData[index + j] = r1 * r2 - k * (r1 + r2) * (r1 + r2);
#endif
            //resultData[index + j] = a * b - c * c - k * (a + b) * (a + b);
        }
    }
}


//3x3区域内填充255
void draw_point(unsigned char* src_img, int width, int height, int index_w, int index_h)
{
    if (index_w < 1 ||
        index_w > width - 1 ||
        index_h < 1 ||
        index_h > height - 1
        )
        return;

    char color = 0;
    int index0 = (index_h - 1) * width;
    int index1 = index_h  * width;
    int index2 = (index_h + 1) * width;

    src_img[index0 + index_w - 1] = color;
    src_img[index0 + index_w]     = color;
    src_img[index0 + index_w + 1] = color;

    src_img[index1 + index_w - 1] = color;
    src_img[index1 + index_w]     = color;
    src_img[index1 + index_w + 1] = color;

    src_img[index2 + index_w - 1] = color;
    src_img[index2 + index_w]     = color;
    src_img[index2 + index_w + 1] = color;
}

#ifdef ENABLE_SHI_TOMASI_CONNER_POINT
#define  MAX_THREHOLD  (80)
#else
#define  MAX_THREHOLD  (8000)
#endif

//非极大值抑制
void LocalMaxValue(Mat_<float> &resultData, Mat &srcGray, Mat &ResultImage, int kSize)
{
    int r = kSize / 2;
    ResultImage = srcGray.clone();

    Mat test_img = Mat::zeros(srcGray.size(), CV_8UC1);
    unsigned char* ptest_img = test_img.data;

    //printf("result.rows = %d ; result.cols = %d \n", resultData.rows,  resultData.cols);
    //printf("ResultImage.rows = %d ; ResultImage.cols = %d \n", ResultImage.rows, ResultImage.cols);

    for (int i = r; i < ResultImage.rows - r; i++)
    {
        for (int j = r; j < ResultImage.cols - r; j++)
        {
            if (resultData.at<float>(i, j) > resultData.at<float>(i - 1, j - 1) &&
                resultData.at<float>(i, j) > resultData.at<float>(i - 1, j) &&
                resultData.at<float>(i, j) > resultData.at<float>(i - 1, j + 1) &&
                resultData.at<float>(i, j) > resultData.at<float>(i, j - 1) &&
                resultData.at<float>(i, j) > resultData.at<float>(i, j + 1) &&
                resultData.at<float>(i, j) > resultData.at<float>(i + 1, j - 1) &&
                resultData.at<float>(i, j) > resultData.at<float>(i + 1, j) &&
                resultData.at<float>(i, j) > resultData.at<float>(i + 1, j + 1))
            {
                if ((int)resultData.at<float>(i, j) > MAX_THREHOLD)
                {
                    circle(ResultImage, Point(j, i), 1, Scalar(0, 0, 255), 2, 8, 0);
                    ptest_img[i * ResultImage.cols + j] = 255;
                    //draw_point(ptest_img, ResultImage.cols, ResultImage.rows, j, i);
                    //printf(" i = %d j = %d \n", i, j);
                }
            }

        }
    }
    imshow("test_img", test_img);
}


void Local_3x3_MaxValue_CStyle(float *resultData,unsigned char* srcGray, unsigned char* ResultImage, int width, int height)
{
    int r = 1;
    
    for (int i = r; i < height - r; i++)
    {
        int index0 = (i - r)*width;
        int index1 = i * width;
        int index2 = (i + r) * width;

        for (int j = r; j < width - r; j++)
        {
            if (resultData[index1 + j] > resultData[index0 + j - 1] &&
                resultData[index1 + j] > resultData[index0 + j ] &&
                resultData[index1 + j] > resultData[index0 + j + 1] &&
                resultData[index1 + j] > resultData[index1 + j - 1] &&
                resultData[index1 + j] > resultData[index1 + j + 1] &&
                resultData[index1 + j] > resultData[index2 + j - 1] &&
                resultData[index1 + j] > resultData[index2 + j ] &&
                resultData[index1 + j] > resultData[index2 + j + 1]
                )
            {
                if ((int)resultData[index1 + j] > MAX_THREHOLD)
                {
                    draw_point(ResultImage, width, height, j, i);
                }
            }

        }
    }
}
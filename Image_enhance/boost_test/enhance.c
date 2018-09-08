#include "enhance.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int HistEqulizedF32(
    unsigned char *   src,
    int width, int height,
    unsigned char *   dst,
    int *   scratchBuf)
{
    int i, j, temp;
    unsigned char *data = src;
    float size = height*width;

    //直方图  
    unsigned int *hist = (unsigned int *)scratchBuf;
    //归一化直方图  
    float *histPDF = (float *)(scratchBuf + 256);
    //累积直方图  
    float *histCDF = (float *)(scratchBuf + 256 * 2);
    //直方图均衡化,映射   
    int *histEQU = (int *)(scratchBuf + 256 * 3);

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            temp = data[i*width + j];
            hist[temp]++;
        }
    }

    for (i = 0; i < 255; i++)
    {
        histPDF[i] = (float)hist[i] / size;
    }

    for (i = 0; i < 256; i++)
    {
        if (0 == i) histCDF[i] = histPDF[i];
        else histCDF[i] = histCDF[i - 1] + histPDF[i];
    }

    for (i = 0; i < 256; i++)
    {
        histEQU[i] = (int)(255.0 * histCDF[i] + 0.5);
    }
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            temp = data[i*width + j];
            dst[i*width + j] = histEQU[temp];
        }
    }
    return 0;
}

int HistEqulizedInt(
    unsigned char *  src,
    int width, int height,
    unsigned char *  dst,
    int *   scratchBuf)
{
    int i, j, temp;
    unsigned char *   data = src;
    int size = height*width;

    //直方图  
    unsigned int *   hist = (unsigned int *)scratchBuf;
    //归一化直方图  
    int *  histPDF = scratchBuf + 256; //size=256  
    //累积直方图  
    int *  histCDF = scratchBuf + 256 * 2; //size=256  
    //直方图均衡化,映射  
    int *  histEQU = scratchBuf + 256 * 3; //size=256  
    ///float factor = 255.0 / size;
    int factorInt = (int)((255.0 / size) * (1 << 20));
    int factor0_5 = 1 << 19;//0.5 * (1 << 20);

    memset(scratchBuf, 0, 4 * 256 * sizeof(int));

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            temp = data[i*width + j];
            hist[temp]++;
        }
    }

    for (i = 0; i < 255; i++)
    {
        //histPDF[i]=(float)hist[i]/size;  
        histPDF[i] = hist[i];
    }

    for (i = 0; i < 256; i++)
    {
        if (0 == i)
            histCDF[i] = histPDF[i];
        else
            histCDF[i] = histCDF[i - 1] + histPDF[i];
    }

    for (i = 0; i < 256; i++)
    {
        //histEQU[i] = (int)(factor * histCDF[i] + 0.5);  
        histEQU[i] = (factorInt * histCDF[i] + factor0_5) >> 20;
    }
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            temp = data[i*width + j];
            dst[i * width + j] = histEQU[temp];
        }
    }
    return 0;
}

extern int gGaussianKernelIq23_3x3[9] =
{
    //0.085740, 0.121334, 0.085740,
    //0.121334, 0.171703, 0.121334,
    //0.085740, 0.121334, 0.085740

    719239,  1017823,  719239,
    1017823, 1440349, 1017823,
    719239,  1017823,  719239
};

int GaussianKernelFilterIq23(unsigned char *  src, unsigned char *  dst, int width, int height)
{
    if (!src || !dst
        || width < 0
        || height < 0)
    {
        return -1;
    }
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

            dstPtr[j] = (temp + halfIq23) >> 23;

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

                dstPtr[j] = (temp + halfIq23) >> 23;

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

            dstPtr[j] = (temp + halfIq23) >> 23;

            srcPtr0++;
            srcPtr1++;
        }
        dstPtr[0] = dstPtr[1];
        dstPtr[width - 1] = dstPtr[width - 2];
    }
    return 0;
}

int IFBoxFilter8UC5x5(
    const unsigned char*  srcImg,
    const int srcImgW,
    const int srcImgH,
    float*  dstImg)
{
    if (!srcImg || !dstImg || srcImgW <= 0 || srcImgH <= 0)
        return -1;

    {
        int sumArr[176];
        unsigned char*   srcPtr = NULL;
        float*   avgPtr = NULL;
        int w = 0, h = 0;
        int i = 0, idx = 0;
        int sum0 = 0, sum1 = 0;

        srcPtr = (unsigned char*)srcImg;
        memset(sumArr, 0, 176 * sizeof(int));

        // calculate the 5x5 neighbor summer value in the first row 
        for (h = 0; h < 5; h++)
        {
            srcPtr = (unsigned char*)srcImg + h * 176;
            for (w = 2; w < 174; w++)
            {
                sumArr[w - 2] += srcPtr[w];
                sumArr[w - 1] += srcPtr[w];
                sumArr[w] += srcPtr[w];
                sumArr[w + 1] += srcPtr[w];
                sumArr[w + 2] += srcPtr[w];
            }
        }

        // calculate the 5x5 average value in the first row 
        avgPtr = dstImg + 2 * 176;
        for (idx = 2; idx < 174; idx++)
        {
            avgPtr[idx] = 0.04 * sumArr[idx];// / 25; 
        }

        // calculate the 5x5 neighbor average value of the following row 
        for (h = 5; h < 144; h++)
        {
            unsigned char* ptr0 = (unsigned char*)srcImg + (h - 5) * 176;
            unsigned char* ptr1 = (unsigned char*)srcImg + h * 176;

            sum0 = ptr0[0] + ptr0[1] + ptr0[2] + ptr0[3] + ptr0[4];
            sum1 = ptr1[0] + ptr1[1] + ptr1[2] + ptr1[3] + ptr1[4];

            sumArr[2] += sum1 - sum0;
            for (w = 0; w < 171; w++)
            {
                // update the sum0 and sum1 value
                //sum0 += ptr0[w + 5] - ptr0[w];
                //sum1 += ptr1[w + 5] - ptr1[w];
                //sumArr[w + 3] += sum1 - sum0;

                sum0 -= ptr0[w];
                sum0 += ptr0[w + 5];

                sum1 -= ptr1[w];
                sum1 += ptr1[w + 5];

                sumArr[w + 3] -= sum0;
                sumArr[w + 3] += sum1;
            }

            avgPtr = dstImg + (h - 2) * 176;
            for (w = 3; w < 174; w++)
            {
                avgPtr[w] = sumArr[w] * 0.04;
            }
        }
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////
/// \brief Box filter used 5x5 kernel to process char image.
/// \details    
/// \param[in]  srcImg      The input source image data.
/// \param[in]  srcImgW     The width of input source image data.
/// \param[in]  srcImgH     The height of input source image data.
/// \param[out] dstImg      The output destination image data, the 5x5 neighbor summer value,
/// \                       it need to divide 25 to get the filter value  
/// \param[in]  scratchBuf  The iram buffer, size is srcImgW * srcImgH * sizeof(unsigned char) + 176 * sizeof(short) .
/// \return     The error code -1. 0 if succeeds.
//////////////////////////////////////////////////////////////////////////
#if 1
int IFBoxFilter8UCHAR5x5_QCIF(
    const unsigned char*   srcImg,
    const int srcImgW,
    const int srcImgH,
    short*   dstImg,
    unsigned char*   scratchBuf)
{
    if (!srcImg || !dstImg || srcImgW <= 0 || srcImgH <= 0)
        return -1;

    {
        unsigned char*   srcIramBuf = (unsigned char*  )scratchBuf;
        short*   sumArr = (short*  )(scratchBuf + srcImgW * srcImgH);
        const unsigned char*   srcPtr = NULL;
        short*   sumPtr = NULL;
        int w = 0, h = 0;
        int i = 0, idx = 0;
        int sum0 = 0, sum1 = 0;
        int tl = 0;

        // 把srcImg 复制到iram buffer中  
        memcpy(srcIramBuf, srcImg, srcImgW * srcImgH * sizeof(unsigned char));

        srcPtr = srcIramBuf;
        memset(sumArr, 0, 176 * sizeof(short));
        // calculate the 5x5 neighbor summer value in the first row 
        for (h = 0; h < 5; h++)
        {
            srcPtr = srcIramBuf + h * 176;
            sumArr[2] += srcPtr[0] + srcPtr[1];
            sumArr[3] += srcPtr[1];
            for (w = 2; w < 174; w++)
            {
                unsigned char srcTmp = srcPtr[w];
                sumArr[w - 2] += srcTmp;
                sumArr[w - 1] += srcTmp;
                sumArr[w] += srcTmp;
                sumArr[w + 1] += srcTmp;
                sumArr[w + 2] += srcTmp;
            }
            sumArr[172] += srcPtr[174];
            sumArr[173] += srcPtr[174] + srcPtr[175];
        }

        // calculate the 5x5 average value in the first row 
        sumPtr = dstImg + 2 * 176;

        //for (idx = 2; idx < 174; idx ++)
        //{
        //    sumPtr[idx] = sumArr[idx];// / 25;
        //}
        sumPtr[0] = sumPtr[1] = sumArr[2];
        memcpy(sumPtr + 2, sumArr + 2, 172 * sizeof(short));
        sumPtr[174] = sumPtr[175] = sumArr[173];

        //fill the edge
        memcpy(dstImg, sumPtr, 176 * sizeof(short));
        memcpy(dstImg + 176, sumPtr, 176 * sizeof(short));

        sumPtr = dstImg + 3 * 176;
        // calculate the 5x5 neighbor average value of the following row
        for (h = 5; h < 144; h++)
        {
            const unsigned char*   ptr0 = srcIramBuf + (h - 5) * 176;
            const unsigned char*   ptr1 = srcIramBuf + h * 176;

            sum0 = ptr0[0] + ptr0[1] + ptr0[2] + ptr0[3] + ptr0[4];
            sum1 = ptr1[0] + ptr1[1] + ptr1[2] + ptr1[3] + ptr1[4];

            sumArr[2] += sum1 - sum0;
            for (w = 0; w < 171; w++)
            {
                // update the sum0 and sum1 value
                int wTmp = w + 5;
                sum0 += ptr0[wTmp] - ptr0[w];
                sum1 += ptr1[wTmp] - ptr1[w];
                sumArr[w + 3] += sum1 - sum0;
            }

            //sumPtr = dstImg + (h - 2) * 176;

            for (w = 2; w < 174; w++)
            {
                sumPtr[w] = sumArr[w];
            }
            sumPtr[0] = sumPtr[1] = sumArr[2];
            sumPtr[174] = sumPtr[175] = sumArr[173];
            sumPtr += 176;
        }

        memcpy(dstImg + 176 * 142, dstImg + 176 * 141, 176 * sizeof(short));
        memcpy(dstImg + 176 * 143, dstImg + 176 * 141, 176 * sizeof(short));

    }

    return 0;
}

int IFBoxFilter8UCHAR3x3_QCIF(
    const unsigned char*   srcImg,
    const int srcImgW,
    const int srcImgH,
    short*   dstImg,
    unsigned char*   scratchBuf)
{
    if (!srcImg || !dstImg || srcImgW <= 0 || srcImgH <= 0)
        return -1;

    {
        unsigned char*   srcIramBuf = (unsigned char*  )scratchBuf;
        short*   sumArr = (short*  )(scratchBuf + srcImgW * srcImgH);
        const unsigned char*   srcPtr = NULL;
        short*   sumPtr = NULL;
        int w = 0, h = 0;
        int i = 0, idx = 0;
        int sum0 = 0, sum1 = 0;
        int tl = 0;

        // 把srcImg 复制到iram buffer中  
        memcpy(srcIramBuf, srcImg, srcImgW * srcImgH * sizeof(unsigned char));

        srcPtr = srcIramBuf;
        memset(sumArr, 0, 176 * sizeof(short));
        // calculate the 3x3 neighbor summer value in the first row 
        for (h = 0; h < 3; h++)
        {
            srcPtr = srcIramBuf + h * 176;
            sumArr[1] += srcPtr[0];
            for (w = 1; w < 175; w++)
            {
                unsigned char srcTmp = srcPtr[w];
                sumArr[w - 1] += srcTmp;
                sumArr[w] += srcTmp;
                sumArr[w + 1] += srcTmp;
            }
            sumArr[174] += srcPtr[175];
        }

        sumPtr = dstImg + 1 * 176;

        sumPtr[0] = sumArr[1];
        memcpy(sumPtr + 1, sumArr + 1, 174 * sizeof(short));
        sumPtr[175] = sumArr[174];

        //fill the edge
        memcpy(dstImg, sumPtr, 176 * sizeof(short));
        //memcpy(dstImg + 176, sumPtr, 176 * sizeof(short));

        sumPtr = dstImg + 2 * 176;
        // calculate the 3x3 neighbor average value of the following row
        for (h = 3; h < 144; h++)
        {
            const unsigned char*   ptr0 = srcIramBuf + (h - 3) * 176;
            const unsigned char*   ptr1 = srcIramBuf + h * 176;

            sum0 = ptr0[0] + ptr0[1] + ptr0[2];
            sum1 = ptr1[0] + ptr1[1] + ptr1[2];

            sumArr[1] += sum1 - sum0;
            for (w = 0; w < 173; w++)
            {
                // update the sum0 and sum1 value
                int wTmp = w + 3;
                sum0 += ptr0[wTmp] - ptr0[w];
                sum1 += ptr1[wTmp] - ptr1[w];
                sumArr[w + 2] += sum1 - sum0;

            }

            for (w = 1; w < 175; w++)
            {
                sumPtr[w] = sumArr[w];
            }
            sumPtr[0] = sumArr[1];
            sumPtr[175] = sumArr[174];
            sumPtr += 176;
        }

        //memcpy(dstImg + 176 * 142, dstImg + 176 * 141, 176 * sizeof(short));
        memcpy(dstImg + 176 * 143, dstImg + 176 * 142, 176 * sizeof(short));

    }

    return 0;
}

#endif

int IFBoxFilter8UCHAR5x5_CIF(
    const unsigned char*   srcImg,
    const int srcImgW,
    const int srcImgH,
    short*   dstImg,
    unsigned char*   scratchBuf)
{
    if (!srcImg || !dstImg || srcImgW <= 0 || srcImgH <= 0)
        return -1;

    {
        //unsigned char*   srcIramBuf = (unsigned char*  )scratchBuf;
        short*   sumArr = (short*  )(scratchBuf);
        const unsigned char*   srcPtr = NULL;
        short*   sumPtr = NULL;
        int w = 0, h = 0;
        int i = 0;
        int sum0 = 0, sum1 = 0;
        int tl = 0;

        // 把srcImg 复制到iram buffer中  
        //memcpy(srcIramBuf, srcImg, srcImgW * srcImgH * sizeof(unsigned char));

        srcPtr = srcImg;
        memset(sumArr, 0, 352 * sizeof(short));
        // calculate the 5x5 neighbor summer value in the fiveth row 
        for (h = 0; h < 5; h++)
        {
            srcPtr = srcImg + h * 352;
            sumArr[2] += srcPtr[0] + srcPtr[1];
            sumArr[3] += srcPtr[1];
            for (w = 2; w < 350; w++)
            {
                unsigned char srcTmp = srcPtr[w];
                sumArr[w - 2] += srcTmp;
                sumArr[w - 1] += srcTmp;
                sumArr[w] += srcTmp;
                sumArr[w + 1] += srcTmp;
                sumArr[w + 2] += srcTmp;
            }
            //sumArr[172] += srcPtr[174];
            //sumArr[173] += srcPtr[174] + srcPtr[175];
            sumArr[348] += srcPtr[350];
            sumArr[349] += srcPtr[350] + srcPtr[351];
        }

        // calculate the 5x5 average value in the first row 
        sumPtr = dstImg + 2 * 352;

        //for (idx = 2; idx < 174; idx ++)
        //{
        //    sumPtr[idx] = sumArr[idx];// / 25;
        //}
        sumPtr[0] = sumPtr[1] = sumArr[2];
        memcpy(sumPtr + 2, sumArr + 2, 348 * sizeof(short));
        //sumPtr[174] = sumPtr[175] = sumArr[173];
        sumPtr[350] = sumPtr[351] = sumArr[349];

        //fill the edge
        memcpy(dstImg, sumPtr, 352 * sizeof(short));
        memcpy(dstImg + 352, sumPtr, 352 * sizeof(short));

        sumPtr = dstImg + 3 * 352;
        // calculate the 5x5 neighbor average value of the following row
        for (h = 5; h < 288; h++)
        {
            const unsigned char*   ptr0 = srcImg + (h - 5) * 352;
            const unsigned char*   ptr1 = srcImg + h * 352;

            sum0 = ptr0[0] + ptr0[1] + ptr0[2] + ptr0[3] + ptr0[4];
            sum1 = ptr1[0] + ptr1[1] + ptr1[2] + ptr1[3] + ptr1[4];

            sumArr[2] += sum1 - sum0;
            for (w = 0; w < 347; w++)
            {
                // update the sum0 and sum1 value
                int wTmp = w + 5;
                sum0 += ptr0[wTmp] - ptr0[w];
                sum1 += ptr1[wTmp] - ptr1[w];
                sumArr[w + 3] += sum1 - sum0;
            }

            for (w = 2; w < 350; w++)
            {
                sumPtr[w] = sumArr[w];
            }
            sumPtr[0] = sumPtr[1] = sumArr[2];
            sumPtr[350] = sumPtr[351] = sumArr[349];
            sumPtr += 352;
        }

        memcpy(dstImg + 352 * 286, dstImg + 352 * 285, 352 * sizeof(short));
        memcpy(dstImg + 352 * 287, dstImg + 352 * 285, 352 * sizeof(short));

    }

    return 0;
}

int IFBoxFilter8UCHAR3x3_CIF(
    const unsigned char*   srcImg,
    const int srcImgW,
    const int srcImgH,
    short*   dstImg,
    unsigned char*   scratchBuf)
{
    if (!srcImg || !dstImg || srcImgW <= 0 || srcImgH <= 0)
        return -1;

    {
        //unsigned char*   srcIramBuf = (unsigned char*  )scratchBuf;
        short*   sumArr = (short*  )(scratchBuf);
        const unsigned char*   srcPtr = NULL;
        short*   sumPtr = NULL;
        int w = 0, h = 0;
        int i = 0;
        int sum0 = 0, sum1 = 0;
        int tl = 0;

        srcPtr = srcImg;
        memset(sumArr, 0, 352 * sizeof(short));
        // calculate the 3x3 neighbor summer value in the first row 
        for (h = 0; h < 3; h++)
        {
            srcPtr = srcImg + h * 352;
            sumArr[1] += srcPtr[0];
            for (w = 1; w < 351; w++)
            {
                unsigned char srcTmp = srcPtr[w];
                sumArr[w - 1] += srcTmp;
                sumArr[w] += srcTmp;
                sumArr[w + 1] += srcTmp;
            }
            sumArr[350] += srcPtr[351] ;
        }

        sumPtr = dstImg + 1 * 352;

        sumPtr[0] = sumArr[1];
        memcpy(sumPtr + 1, sumArr + 1, 350 * sizeof(short));
        sumPtr[351] = sumArr[350];

        //fill the edge
        memcpy(dstImg, sumPtr, 352 * sizeof(short));
        //memcpy(dstImg + 352, sumPtr, 352 * sizeof(short));

        sumPtr = dstImg + 2 * 352;
        // calculate the 3x3 neighbor average value of the following row
        for (h = 3; h < 288; h++)
        {
            const unsigned char*   ptr0 = srcImg + (h - 3) * 352;
            const unsigned char*   ptr1 = srcImg + h * 352;

            sum0 = ptr0[0] + ptr0[1] + ptr0[2];
            sum1 = ptr1[0] + ptr1[1] + ptr1[2];

            sumArr[1] += sum1 - sum0;
            for (w = 0; w < 349; w++)
            {
                // update the sum0 and sum1 value
                int wTmp = w + 3;
                sum0 += ptr0[wTmp] - ptr0[w];
                sum1 += ptr1[wTmp] - ptr1[w];
                sumArr[w + 2] += sum1 - sum0;
            }

            for (w = 1; w < 351; w++)
            {
                sumPtr[w] = sumArr[w];
            }
            sumPtr[0]   = sumArr[1];
            sumPtr[351] = sumArr[350];
            sumPtr += 352;
        }

        memcpy(dstImg + 352 * 287, dstImg + 352 * 286, 352 * sizeof(short));

    }

    return 0;
}
    
int IFBoxFilter8UCHAR3x3(
    const unsigned char*   srcImg,
    const int srcImgW,
    const int srcImgH,
    short*   dstImg,
    unsigned char*   scratchBuf)
{
    if (!srcImg || !dstImg || srcImgW <= 0 || srcImgH <= 0)
        return -1;

    {
        //unsigned char*   srcIramBuf = (unsigned char*  )scratchBuf;
        short*   sumArr = (short*  )(scratchBuf);
        const unsigned char*   srcPtr = NULL;
        short*   sumPtr = NULL;
        int w = 0, h = 0;
        int i = 0;
        int sum0 = 0, sum1 = 0;
        int tl = 0;

        // 把srcImg 复制到iram buffer中  
        //memcpy(srcIramBuf, srcImg, srcImgW * srcImgH * sizeof(unsigned char));

        srcPtr = srcImg;
        memset(sumArr, 0, srcImgW * sizeof(short));
        // calculate the 3x3 neighbor summer value in the first row 
        for (h = 0; h < 3; h++)
        {
            srcPtr = srcImg + h * srcImgW;
            sumArr[1] += srcPtr[0];
            for (w = 1; w < srcImgW - 1; w++)
            {
                unsigned char srcTmp = srcPtr[w];
                sumArr[w - 1] += srcTmp;
                sumArr[w] += srcTmp;
                sumArr[w + 1] += srcTmp;
            }
            sumArr[srcImgW - 2 ] += srcPtr[srcImgW - 1] ;
        }

        sumPtr = dstImg + 1 * srcImgW;

        sumPtr[0] = sumArr[1];
        memcpy(sumPtr + 1, sumArr + 1, (srcImgW - 2) * sizeof(short));
        sumPtr[srcImgW - 1] = sumArr[srcImgW - 2]; 

        //fill the edge
        memcpy(dstImg, sumPtr, srcImgW * sizeof(short));
        //memcpy(dstImg + 352, sumPtr, 352 * sizeof(short));

        sumPtr = dstImg + 2 * srcImgW;
        // calculate the 3x3 neighbor average value of the following row
        for (h = 3; h < srcImgH; h++)
        {
            const unsigned char*   ptr0 = srcImg + (h - 3) * srcImgW;
            const unsigned char*   ptr1 = srcImg + h * srcImgW;

            sum0 = ptr0[0] + ptr0[1] + ptr0[2];
            sum1 = ptr1[0] + ptr1[1] + ptr1[2];

            sumArr[1] += sum1 - sum0;
            for (w = 0; w < srcImgW - 3; w++)
            {
                // update the sum0 and sum1 value
                int wTmp = w + 3;
                sum0 += ptr0[wTmp] - ptr0[w];
                sum1 += ptr1[wTmp] - ptr1[w];
                sumArr[w + 2] += sum1 - sum0;
            }

            for (w = 1; w < srcImgW - 1; w++)
            {
                sumPtr[w] = sumArr[w];
            }
            sumPtr[0]   = sumArr[1];
            sumPtr[srcImgW - 1] = sumArr[srcImgW - 2];
            sumPtr += srcImgW;
        }

        memcpy(dstImg + srcImgW * (srcImgH - 1), dstImg + srcImgW * (srcImgH - 2), srcImgW * sizeof(short));

    }

    return 0;
}

#if 1 //OPTIMIZED
//使用3*3的laplacian算子进行边缘检测（该函数不支持原地运算）
int Laplacian8U_3x3(const unsigned char *   src, unsigned char *   dst, int width, int height)
{
    int x, y, i;//j;
    //常用的两种3*3laplacian算子
    //     int nMask1[9] ={   0, -1, 0,-1, 4, -1, 0, -1, 0};
    //     int nMask2[9] ={-1, -1, -1,  -1, 8, -1, -1, -1, -1};
    //int nMask2[9] = { 2, 0, 2, 0, -8, 0, 2, 0, 2 }; //{-1, -1, -1,  -1, 8, -1, -1, -1, -1};
    int sum = 0;
    //int energyTh = 255 * (1 - 0.9);
    //int imgMask[9];
    //地址检查:该函数不支持原地运算
    if (src == dst)
    {
        printf("The src image same with dst image which is not support!! \n");
        return -1;
    }
#if 0 
    imgMask[0] = -width - 1;
    imgMask[1] = -width;
    imgMask[2] = -width + 1;
    imgMask[3] = -1;
    imgMask[4] = 0;
    imgMask[5] = 1;
    imgMask[6] = width - 1;
    imgMask[7] = width;
    imgMask[8] = width + 1;
#endif

    //边缘值置0
    memset(dst, 0, width * height * sizeof(unsigned char));

    for (y = 1; y < height - 1; y++)
    {
        int yIdx = y * width;
        for (x = 1; x < width - 1; x++)
        {
            register int idx = yIdx + x;
            sum = 0;

            //for(i = 0; i < 9; i++)
            //{
            //    sum += src[imgMask[i]+idx]*nMask2[i];
            //}
            sum = (src[idx - width - 1] << 1) + (src[idx - width + 1] << 1) - (src[idx] << 3)
                + (src[idx + width - 1] << 1) + (src[width + 1 + idx] << 1);

            dst[idx] = sum;
            if (sum > 255)
            {
                dst[idx] = 255;
            }

            if (sum < 25/*energyTh = 255 * (1 - 0.9)*/)
            {
                dst[idx] = 0;
            }
        }
    }
    return 0;
}

#else
//使用3*3的laplacian算子进行边缘检测（该函数不支持原地运算）
int Laplacian8U_3x3(const unsigned char *   src, unsigned char *   dst, int nWidth, int nHeight)
{
    int x, y, i;//j;
                //常用的两种3*3laplacian算子  
                //     int nMask1[9] ={   0, -1, 0,-1, 4, -1, 0, -1, 0};
                //     int nMask2[9] ={-1, -1, -1,  -1, 8, -1, -1, -1, -1}; 
    int nMask2[9] = { 2, 0, 2, 0, -8, 0, 2, 0, 2 }; //{-1, -1, -1,  -1, 8, -1, -1, -1, -1};
    int sum = 0;
    int energyTh = 255 * (1 - 0.9);
    int imgMask[9];
    //地址检查:该函数不支持原地运算  
    DBG_NASSERT_COND(src == dst);

    imgMask[0] = -nWidth - 1;
    imgMask[1] = -nWidth;
    imgMask[2] = -nWidth + 1;
    imgMask[3] = -1;
    imgMask[4] = 0;
    imgMask[5] = 1;
    imgMask[6] = nWidth - 1;
    imgMask[7] = nWidth;
    imgMask[8] = nWidth + 1;

    //边缘值置0  
    memset(dst, 0, nWidth * nHeight * sizeof(unsigned char));

    for (y = 1; y < nHeight - 1; y++)
    {
        for (x = 1; x < nWidth - 1; x++)
        {
            int idx = y*nWidth + x;
            sum = 0;
            for (i = 0; i < 9; i++)
            {
                sum += src[imgMask[i] + idx] * nMask2[i];

            }
            if (sum > 255)
            {
                dst[idx] = 255;
            }
            else if (sum < energyTh)
            {
                dst[idx] = 0;
            }
            else
            {
                dst[idx] = sum;
            }
        }
    }
    return 0;
}
#endif

//使用5*5的laplacian算子进行边缘检测（该函数不支持原地运算）
int Laplacian8U_5x5(const unsigned char *   src, unsigned char *   dst, int nWidth, int nHeight)
{
    int x, y, i, j;
    //算法中使用的两组5*5 sobel算子 
    int nMask1[25] = { 1 ,0 ,-2 ,0, 1,
        4 ,0, -8 ,0, 4 ,
        6, 0,-12 ,0 ,6,
        4, 0, -8 ,0 ,4,
        1 ,0, -2 ,0 ,1 };

    int nMask2[25] = { 1, 4 ,6 ,4 ,1 ,
        0, 0,  0, 0, 0 ,
        -2,-8,-12,-8 - 2 ,
        0, 0,  0, 0, 0 ,
        1, 4,  6, 4, 1 };

    int sum = 0;
    int sum1 = 0;
    int sum2 = 0;
    int imgMask[25];

    //地址检查:该函数不支持原地运算 
    if (src == dst)
    {
        printf("The src image same with dst image which is not support!! \n");
        return -1;
    }

    imgMask[0] = -2 * nWidth - 2;
    //生成5*5掩码位置数组 
    for (i = 0; i < 5; i++)
    {
        int nWidthTmp = imgMask[0] + nWidth*i;
        for (j = 0; j < 5; j++)
        {
            imgMask[i * 5 + j] = nWidthTmp + j;
        }
    }

    //边缘值置0 
    memset(dst, 0, nWidth * nHeight * sizeof(unsigned char));

    for (y = 2; y < nHeight - 2; y++)
    {
        for (x = 2; x < nWidth - 2; x++)
        {
            int idx = y*nWidth + x;
            sum = 0;
            sum1 = 0;
            sum2 = 0;
            for (i = 0; i < 25; i++)
            {
                sum1 += src[imgMask[i] + idx] * nMask1[i];
            }
            for (i = 0; i < 25; i++)
            {
                sum2 += src[imgMask[i] + idx] * nMask2[i];
            }
            sum = sum1 + sum2;

            if (sum > 255)
            {
                dst[idx] = 255;
            }
            else if (sum < 0)
            {
                dst[idx] = 0;
            }
            else
            {
                dst[idx] = sum;
            }
        }
    }
    return 0;
}

int GnPC_Laplacian8U(const unsigned char *   src, unsigned char *   dst, int nWidth, int nHeight, int ksize)
{
    int ret = 0;

    if (!src || !dst || nWidth < 0 || nHeight < 0 || ksize < 0)
    {
        printf("Invalid param: %s %d\n", __FILE__  , __LINE__);
        return -1;
    }

    if (ksize == 3)
    {
        ret = Laplacian8U_3x3(src, dst, nWidth, nHeight);
    }
    else if (ksize == 5)
    {
        ret = Laplacian8U_5x5(src, dst, nWidth, nHeight);
    }
    else
    {
        //printf("error!!! kize is not equal 3 or 5 \n");
        ret = -1;
    }

    return ret;
}


int GnPC_CFLaplacian(unsigned char* src, unsigned char* dst, int width, int height, int ksize)
{
    char tpl_5x5[] = {
        -2,-4,-4,-4,-2,
        -4, 0, 8, 0,-4,
        -4, 8,24, 8,-4,
        -4, 0, 8, 0,-4,
        -2,-4,-4,-4,-2
    };
    //     char tpl_3x3[] = {
    //         0, -1, 0,
    //         -1, 4, -1,
    //         0, -1, 0
    //     };
    char tpl_3x3_default[] = {
        -1, -1, -1,
        -1, 8, -1,
        -1, -1, -1
    };

    //assert(ksize == 3 || ksize == 5);

    int tplWidth = ksize;
    char* tpl = ksize == 3 ? tpl_3x3_default : tpl_5x5;

#if 0
    float coef = 1.0f;
    int border = tplWidth / 2;	//
    int offset = 0;
    float result = 0;
    int num = 0;
    int x, y, i, j;
    const int size = width * height;

    for (y = border; y < height - border; y++)
    {
        for (x = border; x < width - border; x++)
        {
            result = 0;
            offset = size - width - y * width + x;

            for (i = -border; i <= border; i++)
            {
                for (j = -border; j <= border; j++)
                {
                    num = tpl[(i + border) * tplWidth + (j + border)];
                    result += src[offset + (-i) * width + j] * num;
                }
            }

            result *= coef;

            if (result > 255.0f)
                dst[offset] = 255;
            else if (result < 0.0f)
                dst[offset] = 0;
            else
                dst[offset] = (unsigned char)result;
        }
    }
#else
    int border = tplWidth / 2;
    int offset = 0;
    int result = 0;
    int num = 0;
    int x, y, i, j;
    const int size = width * height;

    //边缘值置0 
    memset(dst, 0, width * height * sizeof(unsigned char));

    for (y = border; y < height - border; y++)
    {
        for (x = border; x < width - border; x++)
        {
            result = 0;
            offset = size - width - y * width + x;

            for (i = -border; i <= border; i++)
            {
                for (j = -border; j <= border; j++)
                {
                    num = tpl[(i + border) * tplWidth + (j + border)];
                    result += src[offset + (-i) * width + j] * num;
                }
            }

            if (result > 255)
                dst[offset] = 255;
            else if (result < 0)
                dst[offset] = 0;
            else
                dst[offset] = result;
        }
    }
#endif

    return 0;
}

static unsigned char GammaLUT[256];   //查找表 
void updat_gamma_lut(float gamma)
{
    int i = 0;
    float f = 0.0f;
    
    for (i = 0; i < 256; i++)
    {
        f = (i + 0.5F) /256 ;
        f = (float)pow(f, gamma);
        GammaLUT[i] = (unsigned char)(f * 256  - 0.5F);
    }
}

void GammaCorrectiom(unsigned char*   src, int src_w, int src_h, 
            float gamma, unsigned char*   dst, int updat_LUT)
{
    int i = 0;
    float f = 0.0f;
    int w, h;
    //unsigned char GammaLUT[256];   //查找表 
    
    if(1 == updat_LUT)
    {
        for (i = 0; i < 256; i++)
        {
            f = (i + 0.5F) /256 ;
            f = (float)pow(f, gamma);
            GammaLUT[i] = (unsigned char)(f * 256  - 0.5F);
        }
    }
    
    for (h = 0; h < src_h; h++)
    {
        int row = h * src_w;
        for (w = 0; w < src_w; w++)
        {
            dst[row + w] = GammaLUT[src[row + w]];
        }
    }
}

#define ALIGN_128(x)   (x)
void uyvy2rgb24( unsigned char *   uyvydata, int width, int height, 
            unsigned char *  rgbdata, unsigned char *  buf, unsigned int*   rgb_sum)
{
    
    unsigned char*   rgb     = (unsigned char *)ALIGN_128(rgbdata);
    unsigned char*   uyvy422 = (unsigned char *)ALIGN_128(buf);
    
    memcpy(uyvy422, uyvydata, width * height * 2);
    
    {
        unsigned char *   rPtr = NULL;
        unsigned char *   gPtr = NULL;
        unsigned char *   bPtr = NULL;
        int elementsize = width * height;
        
        const short coeff[5] = { 0x2543, 0x3313, -0x0C8A, -0x1A04, 0x408D };
        unsigned char *  rgbImg = buf;
        int j, i;
        
        rPtr = rgb;
        gPtr = rgb + elementsize;
        bPtr = rgb + elementsize * 2;

#if 0
        VLIB_convertUYVYint_to_RGBpl(uyvy422,
            width, width, height, coeff, bPtr, gPtr, rPtr);
        
        //实际 RGB24是按照 bgr存储的。 rPtr中保存B分量   
        for (i = 0, j = 0; i < elementsize; i++, j += 3)
        {
            rgbImg[j]     = rPtr[i];
            rgbImg[j + 1] = gPtr[i];
            rgbImg[j + 2] = bPtr[i];

            rgb_sum[0] += rgbImg[j];
            rgb_sum[1] += rgbImg[j + 1];
            rgb_sum[2] += rgbImg[j + 2];
        }
#endif

   }
    
}

void  rgb2yuv422Packed(unsigned char *   rgbData, int width, int height,  unsigned char *  yuvData)
{
    int i;
    int loop = height * width / 2;
    unsigned char r, g, b, r1, g1, b1;
    int y, u, v, y1, u1, v1;
    
    for (i = 0; i < loop; i++)
    {
        b = *rgbData; rgbData++;
        g = *rgbData; rgbData++;
        r = *rgbData; rgbData++;
        b1 = *rgbData; rgbData++;
        g1 = *rgbData; rgbData++;
        r1 = *rgbData; rgbData++;

        y = ((GetY_256(r, g, b) + 128) >> 8) + 16;
        u = ((GetU_256(r, g, b) + 128) >> 8) + 128;
        v = ((GetV_256(r, g, b) + 128) >> 8) + 128;

        y1 = ((GetY_256(r1, g1, b1) + 128) >> 8) + 16;
        u1 = ((GetU_256(r1, g1, b1) + 128) >> 8) + 128;
        v1 = ((GetV_256(r1, g1, b1) + 128) >> 8) + 128;

        
        *yuvData++ = (TUNE(u) + TUNE(u1)) >> 1;
        *yuvData++ = TUNE(y);
        *yuvData++ = TUNE(v);
        *yuvData++ = TUNE(y1);
    }

}

void white_balance_gray_world( unsigned char *  rgb_img, int w, int h)
{
    int image_size   = w * h * 3;
    int channel_size = w * h;
    int j = 0;

    unsigned int r_sum = 0,  g_sum = 0,  b_sum = 0;
    float r_mean = 0.0, g_mean = 0.0, b_mean = 0.0, k_mean = 0.0;

    //增益系数   
    float rCoef = 0.0f, gCoef = 0.0f, bCoef = 0.0f;

    //实际 RGB24是按照 bgr存储的  
    for ( j = 0; j < image_size;  j += 3)
    {   
        //B
        b_sum += rgb_img[j];
        //G
        g_sum += rgb_img[j + 1];
        //R
        r_sum += rgb_img[j + 2];
    }

    r_mean = r_sum / channel_size;
    g_mean = g_sum / channel_size;
    b_mean = b_sum / channel_size;
    //printf("r_mean = %f g_mean = %f, b_mean = %f\n", r_mean, g_mean, b_mean);

    //三通道的平均灰度  
    k_mean = (r_mean + g_mean + b_mean) / 3;
    //printf("k_mean = %f\n", k_mean);

    //三通道的增益系数
    rCoef = k_mean / r_mean;
    gCoef = k_mean / g_mean;
    bCoef = k_mean / b_mean;

    //printf("rCoef = %f gCoef = %f, bCoef = %f\n", rCoef , gCoef , bCoef);

    for (j = 0; j < image_size; j += 3)
    {
         float value = 0.0f;
         //B
         value = rgb_img[j] * bCoef;
         if(value > 254.8)
         {
             rgb_img[j] = 255;
         }
         else
         {
             rgb_img[j] = (unsigned char)value;
         }
         
        //G
         value = rgb_img[j + 1] * gCoef;
         rgb_img[j+1] = (unsigned char)(rgb_img[j+1] * gCoef);
         if (value > 254.8)
         {
             rgb_img[j+1] = 255;
         }
         else
         {
             rgb_img[j+1] = (unsigned char)value;
         }

        //R
         value = rgb_img[j + 2] * rCoef;
         rgb_img[j+2] = (unsigned char)(rgb_img[j+2] * rCoef);
         if (value > 254.8)
         {
             rgb_img[j+2] = 255;
         }
         else
         {
             rgb_img[j+2] = (unsigned char)value;
         }
    }
}

int  uyvy_white_balance(unsigned char *  uyvybuf, int w, int h, 
        unsigned char *  rgbbuf, unsigned char *  buf)
{
    if(!uyvybuf || w < 0 || h < 0)
    { 
       printf("uyvy_white_balance FUNCTION error\n");
       return -1;
    }
    
    {
        unsigned char* rgb     = (unsigned char *)ALIGN_128(rgbbuf);
        unsigned char* uyvy422 = (unsigned char *)ALIGN_128(buf);
        unsigned char * rPtr = NULL;
        unsigned char * gPtr = NULL;
        unsigned char * bPtr = NULL;
        int elementsize      = w * h;
        
        const short coeff[5]    = { 0x2543, 0x3313, -0x0C8A, -0x1A04, 0x408D };
        unsigned char *rgbImg   = buf;
        int j, i;
        
        rPtr = rgb;
        gPtr = rgb + elementsize;
        bPtr = rgb + elementsize * 2;
        
        memcpy(uyvy422, uyvybuf, w * h * 2);
        
        //VLIB_convertUYVYint_to_RGBpl(uyvy422, w, w, h, coeff, bPtr, gPtr, rPtr);
        UYVY422ToRGB888(uyvy422, w, h, rgb);

        memcpy(buf, rgb, elementsize * 3);

        white_balance_gray_world(buf, w, h);

        rgb2yuv422Packed(buf, h, w, uyvybuf);

        return 0;
    }
}

void transform_I420_to_uyvy(
    unsigned char *y_plane,
    unsigned char *u_plane,
    unsigned char *v_plane,
    int y_stride, int uv_stride,
    unsigned char *image,
    int width, int height)
{
    int row;
    int col;
    unsigned char *pImg = image;
    for (row = 0; row < height; row = row + 1)
    {
        for (col = 0; col < width; col = col + 2)
        {
            pImg[0] = u_plane[row / 2 * uv_stride + col / 2];
            pImg[1] = y_plane[row * y_stride + col];
            pImg[2] = v_plane[row / 2 * uv_stride + col / 2];
            pImg[3] = y_plane[row * y_stride + col + 1];
            pImg += 4;
        }
    }
}

int UYVY422ToRGB888(const unsigned char *src_buffer, int w, int h, const unsigned char *des_buffer)
{
    unsigned char *yuv, *rgb;
    unsigned char u, v, y1, y2;

    yuv = src_buffer;
    rgb = des_buffer;

    if (yuv == NULL || rgb == NULL)
    {
        printf("error: input data null!\n");
        return -1;
    }

    int size = w * h;

    for (int i = 0; i < size; i += 2)
    {
        y1 = yuv[2 * i + 1];
        y2 = yuv[2 * i + 3];
        u = yuv[2 * i];
        v = yuv[2 * i + 2];

#if 0
        rgb[3 * i] = (unsigned char)(y1 + 1.402*(u - 128));  //b
        rgb[3 * i + 1] = (unsigned char)(y1 - 0.344*(u - 128) - 0.714*(v - 128));  //g
        rgb[3 * i + 2] = (unsigned char)(y1 + 1.772*(v - 128));   //r

        rgb[3 * i + 3] = (unsigned char)(y2 + 1.375*(u - 128));
        rgb[3 * i + 4] = (unsigned char)(y2 - 0.344*(u - 128) - 0.714*(v - 128));
        rgb[3 * i + 5] = (unsigned char)(y2 + 1.772*(v - 128));

#else
        //为提高性能此处用移位运算；
        rgb[3 * i] = (unsigned char)TUNE((y1 + (u - 128) + ((104 * (u - 128)) >> 8)));  //b
        rgb[3 * i + 1] = (unsigned char)TUNE((y1 - (89 * (v - 128) >> 8) - ((183 * (u - 128)) >> 8)));  //g
        rgb[3 * i + 2] = (unsigned char)TUNE((y1 + (v - 128) + ((199 * (v - 128)) >> 8)));   //r

        rgb[3 * i + 3] = (unsigned char)TUNE((y2 + (u - 128) + ((104 * (u - 128)) >> 8)));
        rgb[3 * i + 4] = (unsigned char)TUNE((y2 - (89 * (v - 128) >> 8) - ((183 * (u - 128)) >> 8)));
        rgb[3 * i + 5] = (unsigned char)TUNE((y2 + (v - 128) + ((199 * (v - 128)) >> 8)));
#endif 
    }

    return 0;
}


/**********************************************************************
Finds the 100 best features in an image, and tracks these
features to the next image.  Saves the feature
locations (before and after tracking) to text files and to PPM files, 
and prints the features to the screen.
**********************************************************************/

#include "pnmio.h"
#include "klt.h"

#include "cvhead.h"
#include "draw_img.h"

using namespace cv;
using namespace std;


#ifdef __cplusplus
extern "C"
{
#endif
    
    void RunExample1()
    {
        unsigned char *img1, *img2;
        KLT_TrackingContext tc;
        KLT_FeatureList fl;
        int nFeatures = 200;
        //int ncols, nrows;
        int i;

        //画线用  
        int x0 = 0, x1 = 0, y0 = 0, y1 = 0;

        tc = KLTCreateTrackingContext();
        KLTPrintTrackingContext(tc);
        fl = KLTCreateFeatureList(nFeatures);

        //img1 = pgmReadFile("img0.pgm", NULL, &ncols, &nrows);
        //img2 = pgmReadFile("img1.pgm", NULL, &ncols, &nrows);
        VideoCapture vc;
        string vName;

        cout << "input video : " << endl;
        cin >> vName;
        //vName = "E:\\video\\test.avi";

        vc.open(vName);
        if (!vc.isOpened())
        {
            cout << "Open file failed!" << endl;
            return ;
        }

        Mat frame, YFrame;

        vc >> frame;
        int width = frame.cols;
        int height = frame.rows;
        cvtColor(frame, YFrame, CV_BGR2GRAY);

        img1 = new unsigned char[width * height];
        memset(img1, 0 , width * height * sizeof(unsigned char));

        //Mat feature1(height, width, CV_8UC1);
        //Mat feature2(height, width, CV_8UC1);
        Mat track_result(height, width, CV_8UC1);

        //用于保存上一帧特征点信息
        XY_Point *pre_feature = new XY_Point[nFeatures * sizeof(XY_Point)];

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

            img2 = YFrame.data;

            //KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "feat1.ppm");
            //KLTWriteFeatureList(fl, "feat1.txt", "%3d");
            if (cnt == 0)
            {
                memcpy(img1, img2, width*height*sizeof(unsigned char));
                continue;
            }

            ///shi-tomasi角点检测
            KLTSelectGoodFeatures(tc, img1, width, height, fl);
            if (cnt == 1)
                KLTWriteFeatureList(fl, "feat1.txt", "%3d");

            //printf("\nIn first image:\n");
            //feature1 = YFrame.clone();
            for (i = 0; i < fl->nFeatures; i++)
            {
                /*printf("Feature #%d:  (%f,%f) with value of %d\n",
                i, fl->feature[i]->x, fl->feature[i]->y,
                fl->feature[i]->val);*/
                float x = fl->feature[i]->x;
                float y = fl->feature[i]->y;
                if (x >= 0.0) x += 0.5;
                if (y >= 0.0) y += 0.5;
                x0 = (int)x;
                y0 = (int)y;
                if (x0 < 0 || y0 < 0)
                    continue;
                pre_feature[i].x = x0;
                pre_feature[i].y = y0;
                //Customer_drawPoint(feature1.data, width, height, x0, y0, 5);
            }
            //KLTTrackFeatures(tc, img1, img2, width, height, fl);
            KLTTrackFeatures(tc, img1, img2, width, height, fl);

            if (cnt == 1)
                KLTWriteFeatureList(fl, "feat2.txt", "%3d");

            //printf("\nIn second image:\n");
            //memcpy(feature2.data, img1, width * height);
            //for (i = 0; i < fl->nFeatures; i++)
            //{
            //   /* printf("Feature #%d:  (%f,%f) with value of %d\n",
            //        i, fl->feature[i]->x, fl->feature[i]->y,
            //        fl->feature[i]->val);*/
            //    float x = fl->feature[i]->x;
            //    float y = fl->feature[i]->y;
            //    if (x >= 0.0) x += 0.5;
            //    if (y >= 0.0) y += 0.5;
            //    x1 = (int)x;
            //    y1 = (int)y;
            //    Customer_drawPoint(feature2.data, width, height, x1, y1, 5);
            //}

            memcpy(track_result.data, img2, width*height * sizeof(unsigned char));
            for (i = 0; i < fl->nFeatures; i++)
            {
                x0 = pre_feature[i].x;
                y0 = pre_feature[i].y;

                float x = fl->feature[i]->x;
                float y = fl->feature[i]->y;
                if (x >= 0.0) x += 0.5;
                if (y >= 0.0) y += 0.5;
                x1 = (int)x;
                y1 = (int)y;

                int dstx = abs((x0 - x1) ) * 4;  //画长4倍  
                int dsty = abs((y0 - y1) ) * 4;
                if ((x1 == 0 && y1 == 0)
                    ||( x1 < 0 || y1 < 0 )
                    )
                    continue;

                Customer_drawLine(track_result.data, width ,height, x0, y0, x1 + dstx, y1 + dsty, 5);
            }
            memcpy(img1, img2, width*height * sizeof(unsigned char));

            //imshow("feature1", feature1);
            //imshow("feature2", feature2);
            imshow("track",    track_result);

            cvWaitKey(1);
        }
       
        if (img1)
        {
            delete[] img1;
            img1 = NULL;
        }
        if (pre_feature)
        {
            delete[] pre_feature;
            pre_feature = NULL;
        }

        vc.release();
        //KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, "feat2.ppm");
        //KLTWriteFeatureList(fl, "feat2.fl", NULL);      /* binary file */
        //KLTWriteFeatureList(fl, "feat2.txt", "%5.1f");  /* text file   */
    }

#ifdef __cplusplus
}
#endif

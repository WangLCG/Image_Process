/************************************************
  Harr 特征 + Adaboost 级联实现 demo
  参考论文： Rapid Object Detection using a Boosted cascade of simple features
***********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <ctype.h>

#include "cvhead.h"

#ifdef _EiC
#define WIN32
#endif

using namespace cv;

//#define NUM_SAMPLES_MAX 501

#define NUM_SAMPLES_MAX 7000
#define HAAR_FEATURE_MAX 3
#define WEAKCLASSIFIER_NUM_MAX 300
#define Sample_W 19
#define Sample_H 19


int Image_Mg[2000][2000];

int every_haarfeature_value[NUM_SAMPLES_MAX];//全局数组变量用来保存所有样本的每个特征值，用于Cal_Allsamples_Featurej函数

#define FACE_TXT       "test1\\face\\face.txt"
#define NON_FACE_TXT   "test1\\non_face\\non_face.txt"

#define FACE_DIR_PATH     "test1\\face\\"
#define NON_FACE_DIR_PAH  "test1\\non_face\\"

#define DATAS_TXT         "test1\\data\\datas.txt"
#define DATA_TXT          "test1\\data\\data.txt"

#define  FACE1_TXT     "test1\\face\\face1.txt"
#define  DATA_220_TXT  "test1\\data\\data220.txt"

#define  NON_FACE_TXT_1  "E:\\test\\non-face\\non-face.txt"
#define  NON_FACE_DIR_PAH_1  "E:\\test\\non-face\\"

typedef struct HaarFeature      //一个HaarFeature对应于一个弱分类器
{
   
    int N;
    int kind;           //kind=1、2、3、4、5分别对应论文中的五种haar特征
    CvRect r;

    int threshold;      //弱分类器阈值
    int p;              //不等号方向
    double we;          //权重错误率
    //float e;          //弱分类器对所有样本的分类错误率

    //int n;           //实际样本数量
    //int Samples_FeatureValue[NUM_SAMPLES_MAX];        //用来保存所有样本的特征值
    //int Sorted_Value[NUM_SAMPLES_MAX];                //所有样本排序后的特征值
    double btt ;            /// 最小错误率 / 最大正确率 , 用于更新弱分类器的α权重   
    //char hy[NUM_SAMPLES_MAX];
}
HaarFeature;

HaarFeature Haarfeatures[78500];

int num_features=0;

typedef struct SimpleHaarFeature        //一个HaarFeature对应于一个弱分类器
{
    int kind;           //kind=1、2、3、4、5分别对应论文中的五种haar特征
    CvRect r;
    int threshold;      //弱分类器阈值
    int p;                  //不等号方向
    double we;              //权重错误率
    //float e;              //弱分类器对所有样本的分类错误率

}
SimpleHaarFeature;


typedef struct Ex_IplImage
{
    IplImage * image;
    //CvMat * Mg;
    int Mg[Sample_W + 1][Sample_H + 1];     //积分图   used to keep integral  of grey pixel
    int N;          //the image's number in all samples
    int Y;          //sign for positive of face or not
    int F;      //the feature value,be used to sort
    double W;       //the image's weight
    int IsFalseDetected;
}
Ex_IplImage;

Ex_IplImage Samples[NUM_SAMPLES_MAX];

int num_samples;


typedef struct NYFW
{
    int N;          //the image's number in all samples
    int Y;          //sign for positive of face or not
    int F;      //the feature value
    double W;       //the image's weight
}
NYFW;



/* a boosted battery of classifiers(=stage classifier):
   the stage classifier returns 1
   if the sum of the classifiers' responces
   is greater than threshold and 0 otherwise */
typedef struct HaarStageClassifier
{
    int  count;                         /* 该层强分类器所含弱分类器数量 number of classifiers in the battery */
    double threshold;                   /* 这一层的阈值，这个阈值要跟该层所有弱分类的输出之和比较，然后觉得这层的输出是0还是1 threshold for the boosted classifier */
    HaarFeature classifier[WEAKCLASSIFIER_NUM_MAX];         /* 该层强分类器所含弱分类器 array of classifiers pointer*/
    double alphat[WEAKCLASSIFIER_NUM_MAX];                      // 每个弱分类器的权重  array of at for every chassifier

    /* these fields are used for organizing trees of stage classifiers,
       rather than just stright cascades */
    int next;
    int child;
    int parent;
}
HaarStageClassifier;

/*
typedef struct CvAvgComp
{
    CvRect rect; //bounding rectangle for the object (average rectangle of a group) 
    int neighbors; // number of neighbor rectangles in the group 
}
CvAvgComp;
*/

void Output_Haarfeature(int i)
{
    printf("第%6d个Haar特征如下:\n",i);
    printf("kind:%3d\n",Haarfeatures[i].kind);
    printf("rect:%d, %d, %d, %d\n",Haarfeatures[i].r.x,Haarfeatures[i].r.y,Haarfeatures[i].r.width,Haarfeatures[i].r.height);
    printf("threshold:%d\n",Haarfeatures[i].threshold);
    printf("p:%d\n",Haarfeatures[i].p);
    //printf("e:%f\n",Haarfeatures[i].e);
}


void Cacu_Samples_Mg()
{
    printf("Cacu_Samples_Mg() running begin!\n");
    int c;
    for(c = 0; c < num_samples; c++)
    {
        int i,j;
        int step = Samples[c].image->widthStep/sizeof(uchar);

        uchar * data = (uchar*)(Samples[c].image->imageData);

        //CvMat * p=cvCreateMat(Sample_H + 1,Sample_W + 1,CV_32SC1);

        //Samples[c].Mg=cvCreateMat(Sample_H + 1,Sample_W + 1,CV_32SC1);

        //p -- 积分图  
        int p[Sample_H + 1][Sample_W + 1];

        //cvSetZero(p);

        //cvSetZero(Samples[c].Mg);

        //Mg初始化为0   
        for(i=0;i<=Sample_H ;i++)
            for(j=0;j<=Sample_W ;j++)
            {
                p[i][j]=0;
                Samples[c].Mg[i][j]=0;
            }

        int t;

        //开始计算积分图  
        for(i = 0; i < (Samples[c].image->height); i++)             //p[i][-1]=0, p(x,y)为每一行的累加值
        {
            t = data[i * step];   //第一列   

            //cvmSet(p,i,0,t);
            p[i][0] = t;
        }

        for(i=0;i< (Samples[c].image->height);i++)              //p[x][y]=p[x][y-1]+f[x][y],每一行累加和
            for(j=1;j < (Samples[c].image->width);j++)
            {
                t=data[i*step+j];
                //cvmSet(p,i,j,cvmGet(p,i,j-1)+t);
                p[i][j]=p[i][j-1]+t;    //第 i 行   

            }
        

        for (j = 0; j < (Samples[c].image->width); j++)//g[-1][y]=0;
        {
            //cvmSet(Samples[c].Mg,0,j,cvmGet(p,0,j));
            Samples[c].Mg[0][j] = p[0][j];  //第一行   
        }

        for(i=1;i<(Samples[c].image->height);i++)               //g[x][y]=g[x-1][y]+p[x][y]
            for(j=0;j< (Samples[c].image->width);j++)
            {
                //cvmSet(Samples[c].Mg,i,j,cvmGet(Samples[c].Mg,i-1,j)+cvmGet(p,i,j));
                Samples[c].Mg[i][j] = Samples[c].Mg[i-1][j] + p[i][j];  //积分图
            }
        
        //计算积分图结束  

        //cvReleaseMat(&p);
        /*
        cvNamedWindow("display",CV_WINDOW_AUTOSIZE);

        cvShowImage("display",Samples[c].image);

        cvWaitKey(0);
        
        printf("source image:\n");
        for(i=0;i<Samples[c].image->height;i++)
        {
            for(j=0;j<Samples[c].image->width;j++)
                printf("%3d ",data[i * step + j]);
            printf("\n");
        }
        printf("jifen image:\n");
        for(i=0;i<Samples[c].image->height;i++)
        {
            for(j=0;j<Samples[c].image->width;j++)
                printf("%3d ",Samples[c].Mg[i][j]);
            printf("\n");
        }
        //printf("Mg[i-1][j-1]=%f\n",cvmGet(img->Mg,imgc->height-1,imgc->width-1));
        printf("=========================================\n");
        system("pause");
        //cvWaitKey(0);
        */
    }
    printf("Cacu_Samples_Mg() running end!\n");
}


////////////////////此函数用来产生一种 Haar Feature
///产生的harr特征应该是《Rapid object detection using a boosted cascade of simple features》论文提到的   
///而不是扩展后的haar特征   
void Produce_OneFeature(int kind )
{
    int i,j;
    int x,y;
    //int count = 1;
    int s,t;        //haar feature 初始宽和高

    if( kind == 1)      //  水平特征矩形，一个白色区域在上一个黑色区域在下    /////*****
    {                   //      /////*****
        s = 2;          //      /////*****
        t = 1;          //      /////*****
    }                   //      /////*****

    else if( kind == 2) //   垂直特征矩形，一个白色区域在左一个黑色区域在右    **********
    {                   //      **********
        s = 1;          //      //////////
        t = 2;          //      //////////
    }

    else if( kind == 3) //  线性特征矩形，一个白色区域在上另一个白色区域在下，两个白色区域中间为一个黑色区域    ***///***
    {                   //      ***///***
        s = 3;          //      ***///***
        t = 1;          //      ***///***
    }

    else if( kind == 4) //  线性特征矩形，一个白色区域在左另一个白色区域在右，两个白色区域中间为一个黑色区域     ******
    {                   //      //////
        s = 1;          //      ******
        t = 3;
    }

    else if( kind == 5 || kind == 13 || kind == 10 || kind == 11 || kind == 12) //      ////****
    {                   //  四矩形特征，左上角右下角为白色区域，左下角及右上角为黑色区域    ////****
        s = 2;          //      ****////
        t = 2;          //      ****////
    }

    else if( kind == 6 || kind == 7 || kind == 8 || kind == 9)
    {
        s = 3;
        t = 3;
    }


    else return;
    
    int lx = Sample_W/s ;
    int ly = Sample_H/t ;
    for(i = 1; i <= lx ; i++)       //X、Y方向放大倍数分别为i、j
        for(j = 1; j <= ly ; j++)
        {
            //在Sample_W * Sample_H窗口内滑动 is * jt矩形，遍历每个可能位置  
            for(x = 0; x <= Sample_W - s*i ; x++)       //矩形的左上角顶点为(x,y)
                for(y = 0; y <= Sample_H - t*j ; y++)
                {
                    Haarfeatures[num_features+1].kind = kind;

                    Haarfeatures[num_features+1].r = cvRect( x , y , s*i , t*j);//用像素点个数标志长度

                    Haarfeatures[num_features+1].N = num_features+1 ;

                    num_features++;

                    //for(int k=0;k<10000;k++) int p=0;
                }
        }
    //count--;

}

//////////承接上个函数，用来产生所有的HAAR特征
//其实是产生所有的矩形框，用 kind 标志其类型
void Produce_AllFeatures()
{
    printf("Produce_AllFeatures begin!\n");
    int i;
    for(i=1;i<=5;i++)
        Produce_OneFeature(i);
    printf("the sum is %d\n",num_features);
    printf("Produce_AllFeatures end!\n");
}

//计算某个矩形框 r 的灰度积分
int Cal_Rect_Value1( CvRect r , int Mg[Sample_H + 1][Sample_W + 1])
{
    int i1 = Mg[r.x][r.y];
    int i2 = Mg[r.x + r.width][r.y];
    int i3 = Mg[r.x][r.y + r.height];
    int i4 = Mg[r.x + r.width][r.y + r.height];
    return i1 + i4 - i3 - i2;
}

//计算某个矩形框 r 的灰度积分,对于Samples[i]
int Cal_Rect_Value( CvRect r , int i )
{
    int i4 = Samples[i].Mg[r.y + r.height-1][r.x + r.width-1];

    if(r.x==0 && r.y==0)
        return i4;

    if(r.x==0)
        return i4-Samples[i].Mg[r.y-1][r.width-1];

    if(r.y==0)
        return i4-Samples[i].Mg[r.height-1][r.x-1];

    return i4 + Samples[i].Mg[r.y-1][r.x-1] - Samples[i].Mg[r.y-1][r.x+r.width-1] - Samples[i].Mg[r.y+r.height-1][r.x-1];
}


/*
///////////此函数用来计算一个样本的一个HAAR特征值
int Cal_HaarValue1(HaarFeature * Haarfeature,Ex_IplImage * e_img)
{
    //CvMat * Mg=e_img->Mg;
    int kind=Haarfeature->kind;
    CvRect r=Haarfeature->r;
    int x=r.x;
    int y=r.y;
    int width=r.width;
    int height=r.height;
    if(kind==1)
    {
        return Cal_Rect_Value(r,e_img->Mg) - 2 * Cal_Rect_Value( cvRect(x,y,width/2,height) ,e_img->Mg);
    }
    else if(kind==2)
    {
        return Cal_Rect_Value(r,e_img->Mg) - 2 * Cal_Rect_Value( cvRect(x,y+height/2,width,height/2) ,e_img->Mg);
    }
    else if(kind==3)
    {
        return Cal_Rect_Value(r,e_img->Mg) - 3 * Cal_Rect_Value( cvRect(x+width/3,y,width/3,height) ,e_img->Mg);
    }
    else if(kind==4)
    {
        return Cal_Rect_Value(r,e_img->Mg) - 3 * Cal_Rect_Value( cvRect(x,y+height/3,width,height/3) ,e_img->Mg);
    }
    else if(kind==5)
    {
        return Cal_Rect_Value(r,e_img->Mg) - 2 * Cal_Rect_Value( cvRect(x,y,width/2,height/2) ,e_img->Mg)
            
            - 2 * Cal_Rect_Value( cvRect(x+width/2,y+height/2,width/2,height/2) ,e_img->Mg);
    }
    else if(kind==6)
    {
        return Cal_Rect_Value(r,e_img->Mg) - 9 * Cal_Rect_Value( cvRect(x+width/3,y+height/3,width/3,height/3) ,e_img->Mg);
            
    }
    else if(kind==7)
    {
        return Cal_Rect_Value(r,e_img->Mg) - 9 * Cal_Rect_Value( cvRect(x,y,width,height/3) ,e_img->Mg) / 5
            
            - 9 * Cal_Rect_Value( cvRect(x+width/3,y+height/3,width/3,2*height/3) ,e_img->Mg) / 5;
    }
    else if(kind==8)
    {
        return Cal_Rect_Value(r,e_img->Mg) - 3 * Cal_Rect_Value( cvRect(x,y,width/3,height/3) ,e_img->Mg)
            
            - 3 * Cal_Rect_Value( cvRect(x+2*width/3,y,width/3,height/3) ,e_img->Mg)

            - 3 * Cal_Rect_Value( cvRect(x+width/3,y+2*height/3,width/3,height/3) ,e_img->Mg);
    }
    else if(kind==9)
    {
        return Cal_Rect_Value(r,e_img->Mg) - 9 * Cal_Rect_Value( cvRect(x+width/3,y,width/3,2*height/3) ,e_img->Mg) / 5
            
            - 9 * Cal_Rect_Value( cvRect(x,y+2*height/3,width,height/3) ,e_img->Mg) / 5;
    }
    return 0;
}
*/


/*
///////////此函数用来计算一个样本的一个HAAR特征值
int Cal_HaarValue1(HaarFeature Haarfeature,Ex_IplImage e_img)
{
    //CvMat * Mg=e_img->Mg;
    int kind=Haarfeature.kind;
    CvRect r=Haarfeature.r;
    int x=r.x;
    int y=r.y;
    int width=r.width;
    int height=r.height;
    if(kind==1)
    {
        return Cal_Rect_Value(r,e_img.Mg) - 2 * Cal_Rect_Value( cvRect(x,y,width/2,height) ,e_img.Mg);
    }
    else if(kind==2)
    {
        return Cal_Rect_Value(r,e_img.Mg) - 2 * Cal_Rect_Value( cvRect(x,y+height/2,width,height/2) ,e_img.Mg);
    }
    else if(kind==3)
    {
        return Cal_Rect_Value(r,e_img.Mg) - 3 * Cal_Rect_Value( cvRect(x+width/3,y,width/3,height) ,e_img.Mg);
    }
    else if(kind==4)
    {
        return Cal_Rect_Value(r,e_img.Mg) - 3 * Cal_Rect_Value( cvRect(x,y+height/3,width,height/3) ,e_img.Mg);
    }
    else if(kind==5)
    {
        return Cal_Rect_Value(r,e_img.Mg) - 2 * Cal_Rect_Value( cvRect(x,y,width/2,height/2) ,e_img.Mg)
            
            - 2 * Cal_Rect_Value( cvRect(x+width/2,y+height/2,width/2,height/2) ,e_img.Mg);
    }
    else if(kind==6)
    {
        return Cal_Rect_Value(r,e_img.Mg) - 9 * Cal_Rect_Value( cvRect(x+width/3,y+height/3,width/3,height/3) ,e_img.Mg);
            
    }
    else if(kind==7)
    {
        return Cal_Rect_Value(r,e_img.Mg) - 9 * Cal_Rect_Value( cvRect(x,y,width/3,height/3) ,e_img.Mg) / 4
            
            - 9 * Cal_Rect_Value( cvRect(x+width/3,y+height/3,width/3,2*height/3) ,e_img.Mg) / 4
            
            - 9 * Cal_Rect_Value( cvRect(x+2*width/3,y,width/3,height/3) ,e_img.Mg) / 4;
    }
    else if(kind==8)
    {
        return Cal_Rect_Value(r,e_img.Mg) - 3 * Cal_Rect_Value( cvRect(x,y,width/3,height/3) ,e_img.Mg)
            
            - 3 * Cal_Rect_Value( cvRect(x+2*width/3,y,width/3,height/3) ,e_img.Mg)

            - 3 * Cal_Rect_Value( cvRect(x+width/3,y+2*height/3,width/3,height/3) ,e_img.Mg);
    }
    else if(kind==9)
    {
        return Cal_Rect_Value(r,e_img.Mg) - 9 * Cal_Rect_Value( cvRect(x+width/3,y,width/3,2*height/3) ,e_img.Mg) / 5
            
            - 9 * Cal_Rect_Value( cvRect(x,y+2*height/3,width,height/3) ,e_img.Mg) / 5;
    }
    return 0;
}
*/

///////////此函数用来计算第 j 个样本的第 i 个HAAR特征值
int Cal_HaarValue(int i, int j)
{
    //CvMat * Mg=e_img->Mg;
    int kind=Haarfeatures[i].kind;
    CvRect r=Haarfeatures[i].r;
    int x=r.x;
    int y=r.y;
    int width=r.width;
    int height=r.height;
    if(kind==1)
    {
        return (Cal_Rect_Value(r, j) - 2 * Cal_Rect_Value(cvRect(x, y, width / 2, height), j));
    }
    else if(kind==2)
    {
        return (Cal_Rect_Value(r, j) - 2 * Cal_Rect_Value(cvRect(x, y + height / 2, width, height / 2), j));
    }
    else if(kind==3)
    {
        return (Cal_Rect_Value(r,j) - 3 * Cal_Rect_Value( cvRect(x+width/3,y,width/3,height) ,j));
    }
    else if(kind==4)
    {
        return (Cal_Rect_Value(r,j) - 3 * Cal_Rect_Value( cvRect(x,y+height/3,width,height/3) ,j));
    }
    else if(kind==5)
    {
        return (Cal_Rect_Value(r,j) - 2 * Cal_Rect_Value( cvRect(x,y,width/2,height/2) ,j)
            
            - 2 * Cal_Rect_Value( cvRect(x+width/2,y+height/2,width/2,height/2) ,j));
    }
    else if(kind==6)
    {
        return (Cal_Rect_Value(r,j) - 9 * Cal_Rect_Value( cvRect(x+width/3,y+height/3,width/3,height/3) ,j));
            
    }
    else if(kind==7)
    {
        return (Cal_Rect_Value(r,j) - 9 * Cal_Rect_Value( cvRect(x,y,width/3,height/3) ,j) / 4
            
            - 9 * Cal_Rect_Value( cvRect(x+width/3,y+height/3,width/3,2*height/3) ,j) / 4
            
            - 9 * Cal_Rect_Value( cvRect(x+2*width/3,y,width/3,height/3) ,j) / 4);
    }
    else if(kind==8)
    {
        return (Cal_Rect_Value(r,j) - 3 * Cal_Rect_Value( cvRect(x,y,width/3,height/3) ,j)
            
            - 3 * Cal_Rect_Value( cvRect(x+2*width/3,y,width/3,height/3) ,j)

            - 3 * Cal_Rect_Value( cvRect(x+width/3,y+2*height/3,width/3,height/3) ,j));
    }
    else if(kind==9)
    {
        return (Cal_Rect_Value(r,j) - 9 * Cal_Rect_Value( cvRect(x+width/3,y,width/3,2*height/3) ,j) / 5
            
            - 9 * Cal_Rect_Value( cvRect(x,y+2*height/3,width,height/3) ,j) / 5);
    }
    return 0;
}

void Cal_Allsamples_FeatureValue( )     //计算所有样本对所有特征的特征值,实则计算所有特征的Samples_FeatureValue数组
{
    printf("Cal_Allsamples_FeatureValue( ) begin!\n");
    int i,j;

    for( i = 1 ; i <= num_features ; i++ )      //对于所有的特征
    {
        for( j = 0 ; j < num_samples ; j++ )    //对所有样本进行计算
        {

        //  Haarfeatures[i].Samples_FeatureValue[j] = Cal_HaarValue( i , j );

            //printf("%d\n",Haarfeatures[i].Samples_FeatureValue[j]);

            //system("pause");
        }
    }
    printf("Cal_Allsamples_FeatureValue( ) end!\n");
}

void Test()
{
    CvRect r=cvRect(3,8,8,2);
    int x=r.x;
    int y=r.y;
    int width=r.width;
    int height=r.height;
    int j;
    for(j=0;j<num_samples;j++)
    
        printf("%d ",Cal_Rect_Value(r,j) - 2 * Cal_Rect_Value( cvRect(x,y+height/2,width,height/2) ,j));
    printf("\n");
    system("pause");

}

int comp(const void *a,const void *b)
{
    return ((Ex_IplImage *)a)->F - ((Ex_IplImage *)b)->F;
}


int cmp(const void *b,const void *a)
{
    return (*((NYFW *)b)).F - (*((NYFW *)a)).F;
}

double Produce_WeakClassifier( int n )  //特征n
{
    int i,j;
    NYFW nyfw[NUM_SAMPLES_MAX];     //暂时存储 Samples 排序需要的元素
    double ef=0,en=0;

    for( i=0;i<num_samples;i++)
    {
        nyfw[i].F = Cal_HaarValue(n, i) ;
        nyfw[i].N = i;
        nyfw[i].W = Samples[i].W ;
        nyfw[i].Y = Samples[i].Y ;

        if(Samples[i].Y>0)
            ef += Samples[i].W ;
        else
            en += Samples[i].W ;
    }

    /////////////////////对特征值进行排序
    //for(i=0;i<num_samples;i++)
    //  printf("%d ",nyfw[i].F);

    qsort(nyfw , num_samples , sizeof(NYFW) , cmp);     //从小到大

    //printf("sorted:\n");
    //for(i=0;i<num_samples;i++)
    //  printf("%d ",nyfw[i].F);
    //system("pause");
    
    double et = 10 ;
    int record = 0;
    int pp;
    ////////////////////求出此弱分类器
    for( i=0;i<num_samples;i++)
    {
        if(nyfw[i].Y>0)
        {
            ef -= nyfw[i].W;
            en += nyfw[i].W;
        }
        else
        {
            ef += nyfw[i].W;
            en -= nyfw[i].W;
        }

        double e;
        int pd;
        if(ef<en)
        {
            e=ef;
            pd=1;
        }
        else
        {
            e=en;
            pd=-1;
        }

        if(e<et)
        {
            et = e;
            record = i;
            pp=pd;
        }
    }
    int thd = nyfw[record].F ;
    
    double err_rate = 0 ;
    for( i=0;i<num_samples;i++)
    {
        int h = 0 ;
        if( pp * nyfw[i].F < pp * thd)
            h=1;
        
        err_rate += nyfw[i].W * abs( h - nyfw[i].Y );
    }
    return err_rate ;
}

void Make_WeakClassifier()
{
    printf("Make_WeakClassifier() begin!\n");
    int i,j;
    double sum_w = 0;

    for(i=0;i<num_samples;i++)
        sum_w+=Samples[i].W;

    for(i = 0 ; i < num_samples ; i++)
        Samples[i].W = Samples[i].W / sum_w;

    double ert = 100;
    j=1;
    for(i=1;i<=num_features;i++)
    {
        double e = Produce_WeakClassifier(i) ;
        if(e < ert )
        {
            ert = e ;
            j = i ;
        }
    }
    printf("%d %d %d %d %d\n",Haarfeatures[j].kind,Haarfeatures[j].r.x,Haarfeatures[j].r.y,
        Haarfeatures[j].r.width,Haarfeatures[j].r.height);

    printf("Make_WeakClassifier() end!\n");
    system("pause");
}


void Display()
{
    int ii,jj;

    for(ii=5200;ii<num_features;ii++)
    {
        
        for(jj=0;jj<num_samples;jj++)
        {
            printf("%d %d %d %d %d\n",Haarfeatures[ii].kind,Haarfeatures[ii].r.x,Haarfeatures[ii].r.y,
                    Haarfeatures[ii].r.width,Haarfeatures[ii].r.height);
            int step = Samples[jj].image->widthStep/sizeof(uchar);
            uchar * data = (uchar*)(Samples[jj].image->imageData);
            int i,j;

            cvNamedWindow("display",CV_WINDOW_AUTOSIZE);

            cvShowImage("display",Samples[jj].image);

            cvWaitKey(0);
        
            printf("source image:\n");
            for(i=0;i<Samples[jj].image->height;i++)
            {
                for(j=0;j<Samples[jj].image->width;j++)
                    printf("%3d ",data[i * step + j]);
                printf("\n");
            }
            printf("jifen image:\n");
            for(i=0;i<Samples[jj].image->height;i++)
            {
                for(j=0;j<Samples[jj].image->width;j++)
                    printf("%3d ",Samples[jj].Mg[i][j]);
                printf("\n");
            }
            //printf("Mg[i-1][j-1]=%f\n",cvmGet(img->Mg,imgc->height-1,imgc->width-1));
            printf("=========================================\n");
            CvRect r = Haarfeatures[ii].r;
            printf("Mg[1][0]=%d,Mg[0][0]=%d\n",Samples[jj].Mg[1][0],Samples[jj].Mg[0][0]);
            int v = Cal_Rect_Value(r,jj) - 2 * Cal_Rect_Value( cvRect(r.x,r.y,r.width/2,r.height) ,jj);
            printf("v=%d\n",v);
            printf("Infact:%d\n",Cal_HaarValue( ii , jj ));
            //printf("FeatureValue:%d\n",Haarfeatures[ii].Samples_FeatureValue[jj]);
            system("pause");
            //cvWaitKey(0);
        }
    }
}

int Select_WeakClassifier( )
{
    //Make_WeakClassifier();
    //printf("*******************************\n");
    //Display();
    printf("Select_WeakClassifier( ) begin!\n");
    double time=(double)cvGetTickCount();

    int i;
    
    double ef=0;        //当前所有正样本权值之和,对应于论文中的e+
    double en=0;        //当前所有负样本权值之和,对应于论文中的e-

    NYFW nyfw[NUM_SAMPLES_MAX];     //暂时存储 Samples 排序需要的元素
    
    int j ;

    double min_et = 100 ;       //最小的错误率

    int record = 1 ;        //最小错误率的下标

    for( j = 1 ; j <= num_features ; j++ )  //从所有的特征中挑选错误率最低的一个作为弱分类器
    {   
        ef = 0 ;
        en = 0 ;

        if (j % 500 == 0)
        {
            printf("Processing the %d feature\n", j);
        }

        ////////////////////////////////////////////////////////
        //用来归一化所有样本的权值
        double sum_w=0;

        for(i = 0 ; i < num_samples ; i++)
        {
            sum_w += Samples[i].W;
            //printf("%f ",Samples[j].W);
        }
        //printf("\nsum_w=%f\n",sum_w);
        //system("pause");

        for(i = 0 ; i < num_samples ; i++)
        {
            Samples[i].W = Samples[i].W / sum_w;

            if(Samples[i].Y == 1)           //初使化ef en
            {
                ef += Samples[i].W;   //全部正例的权重
            }
            else 
            {
                en += Samples[i].W;   //全部负例的权重
            }

        }

        //Test();
        //printf("ef=%lf,en=%lf\n",ef,en);
        //system("pause");
        //printf("the W:\n");
        //for(i=0;i<num_samples;i++)
        //  printf("%lf ",Samples[i].W);
        //system("pause");
        /////////////////////////////////////////////////////////
        
        for(i=0;i<num_samples;i++)
        {
            //Samples[i].F = Cal_HaarValue( Haarfeatures[j],Samples[i] );   //计算所有样本的第 j 个特征值
            
            //Haarfeatures[i].Samples_FeatureValue[j] = Cal_HaarValue( i , j );

            every_haarfeature_value[i] = Cal_HaarValue( j , i );

            //Samples[i].F = Haarfeatures[j].Samples_FeatureValue[i] ;
            Samples[i].F = every_haarfeature_value[i] ;

            //Haarfeatrues[j].Samples_FeatureValue[i] = Samples[i].F ;

            nyfw[i].N = i;           //所有样本的 N、Y、F、W 值传到 nyfw 中
            nyfw[i].F = Samples[i].F;
            nyfw[i].Y = Samples[i].Y;
            nyfw[i].W = Samples[i].W;
        }

        /////////////////////对特征值进行排序
        //for(i=0;i<num_samples;i++)
        //  printf("%d ",nyfw[i].F);

        qsort(nyfw , num_samples , sizeof(NYFW) , cmp);     //从小到大 
        ///排序理论参考论文《基于Adaboost算法的人脸检测》 -- 北京大学，赵楠

        //printf("sorted:\n");
        //for(i=0;i<num_samples;i++)
        //  printf("%d ",nyfw[i].F);
        //system("pause");

        ////////////////////求出此弱分类器
        double et=100;      //用来记录当前最小分类错误
        int t=0;            //用来记录当前最小分类错误的下标
        int pto=1;
        int pp=1;
        double e;
        
        //计算元素i之前正例的权重和及负例的权重和  但是感觉出错了 wangc 2018/12/17
        ///此处屡次出现错误,p的方向如何确定
        for(i=0;i<num_samples;i++)
        {
            
            //以下if-else结构求得是 s1+T2 - s2 和 是s2+T1-s1 
            ///s1：i之前的正例权重和, s2: i之前的负例权重和;
            ///T1: 总正例权重和， T2: 总负例权重和  
            if(nyfw[i].Y==1)
            {
                ef -= nyfw[i].W;  //i之前的正例权重和  （总正权重 - 出现在i之后的正例权重）
                en += nyfw[i].W;  //负权重 + i之后的正例权重（包括i）
            }
            else
            {
                ef += nyfw[i].W; 
                en -= nyfw[i].W; 
            }

            pto = 1;  //不等式方向标记 1 -- 大于  -1 -- 小于  
            e = ef;

            if(ef > en)
            {
                e = en;
                pto = -1;
            }

            if(e < et)
            {
                et=e;
                t=i;
                pp=pto;
            }
        }
        //printf("et=%lf\n",et);
        //printf("%d %d\n",t,nyfw[t].N);
        //system("pause");
        Haarfeatures[j].threshold = nyfw[t].F ;
        //t=nyfw[t].N;                              //找到此样本
        //Haarfeatures[j].threshold = Samples[t].F;
        //Haarfeatures[j].p = (Samples[t].Y==1?1:-1);           //p的方向用样本t的正负来标志
        Haarfeatures[j].p = pp;
        Haarfeatures[j].we = et;

        double error_rate = 0 ;
        for(i=0;i<num_samples;i++)      //计算错误率
        {
            int h=0;

            //if p*fj(i) < p*theata then hj(i)=1
            if(Haarfeatures[j].p * Samples[i].F < Haarfeatures[j].p * Haarfeatures[j].threshold)
                h=1;
            
            error_rate += Samples[i].W * (double)abs( h - Samples[i].Y);   //分类错误的算入   

            //Haarfeatures[j].hy[i] = abs( h - Samples[i].Y);       // | hj(xi) - yi |

        }

        if( error_rate < min_et)
        {
            min_et = error_rate ;
            record = j ;
        }
    }
    printf("error_rate=%lf\n",min_et);
    printf("%d %d %d %d %d %d %d\n",Haarfeatures[record].kind,Haarfeatures[record].r.x,Haarfeatures[record].r.y,
        Haarfeatures[record].r.width,Haarfeatures[record].r.height,Haarfeatures[record].threshold,Haarfeatures[record].p);
    //system("pause");
    
    //////////////////////////////////////////////////
    /////////已经找到此弱分类器，更新样本权值（不是权重）  
    Haarfeatures[record].btt = min_et/(1-min_et) ;  //（样本概率，初始时为1/N）
    //基于AdaBoost算法的人脸检测(北京大学,赵楠).pdf   

    printf("bt=%lf\n", Haarfeatures[record].btt);

    //double at = log((1-min_et)/min_et) ;
    for(i=0;i<num_samples;i++)
    {
        //Samples[i].F = Cal_HaarValue( Haarfeatures[j],Samples[i] );   //计算所有样本的第 j 个特征值
            
        //Haarfeatures[i].Samples_FeatureValue[j] = Cal_HaarValue( i , j );

        //every_haarfeature_value[i] = Cal_HaarValue( j , i );

        Samples[i].F = Cal_HaarValue( record , i );
    }

    //按照最佳弱分类器进行权值更新，公式：  新权值 = 旧权值 * e / ( 1 - e )^( 分类正确 ? 1：0 ) 
    for(j=0;j<num_samples;j++)          //对于所有样本图片
    {
        //表示样本被正确的分类，更新样本权值
        if ((Haarfeatures[record].p * Samples[j].F < Haarfeatures[record].p * Haarfeatures[record].threshold
            && Samples[j].Y == 1) 
            || (Haarfeatures[record].p * Samples[j].F >= Haarfeatures[record].p * Haarfeatures[record].threshold
            && Samples[j].Y == 0))
        {
            Samples[j].W *= Haarfeatures[record].btt;   //最终正确分类的权值更新公式为 w = w * e / ( 1 - e )
        }
        //错误分类的的样本权值不变
    }

    printf("Select_WeakClassifier( ) end!\n");
    time = (double)cvGetTickCount() - time ;

    printf("train time is %gs\n",time/((double)cvGetTickFrequency()*1000*1000));
    return record ;

}

int Result( HaarStageClassifier HSC , int n )   //强分类器HSC对样本 n 的检测结果
{
    int i,j;

    double w = 0 ;
    
    //printf("Result:HSC.count=%d,HSC.threshold=%lf,",HSC.count,HSC.threshold);
    for(i=1;i<=HSC.count;i++)
    {
        int t = Cal_HaarValue(HSC.classifier[i].N,n);

        if( HSC.classifier[i].p * t < HSC.classifier[i].p * HSC.classifier[i].threshold)

            w += HSC.alphat[i] ;
    }
    //printf("w=%lf\n",w);
    if( w >= HSC.threshold ) return 1 ;

    return 0 ;
}

void Single_Classifier( )
{
    
    Produce_AllFeatures();
    //system("pause");
    ////////////////////////////////////////////////////////////
    //读取样本图片
    
    FILE * fp;
    char pic_name[15];

    int m=2000;
    int n=2000;

    num_samples = m + n;

    printf("breakpoint1\n");

    if((fp=fopen(FACE_TXT,"r"))==NULL)
    {
        printf("can not open the face.txt!\n");
        exit(0);
    }

    double w = 1/((double)2.0*m);
    printf("w=%f\n",w);system("pause");

    int count_smp = 0 ;

    printf("breakpoint2\n");
    
    while( count_smp < m )
    {
        char s[30]= FACE_DIR_PATH;
        
        //if(fgets(pic_name,14,fp)!=NULL)
        memset(pic_name, 0, sizeof(char) * 33);
        if (fscanf(fp, "%s", pic_name) != EOF)
        {
            printf("%s\n",pic_name);
            strcat(s,pic_name);
            //Samples[count_smp].image = cvLoadImage(s);
            IplImage * img = cvLoadImage(s);
            Samples[count_smp].image = cvCreateImage( cvGetSize(img), 8, 1 );
            cvCvtColor(img,Samples[count_smp].image,CV_BGR2GRAY);
            cvEqualizeHist(Samples[count_smp].image,Samples[count_smp].image);
            cvReleaseImage(&img);
            //printf("def\n");
            Samples[count_smp].Y = 1;
            Samples[count_smp].N = count_smp;
            Samples[count_smp].W = w;
            Samples[count_smp].IsFalseDetected = 0;
            if(!Samples[count_smp].image) printf("Could not load image file:%s\n",pic_name);

            //cvNamedWindow(pic_name, CV_WINDOW_AUTOSIZE );

            //cvShowImage(pic_name, img );

            printf("image size is %d * %d,%d\n",Samples[count_smp].image->width,Samples[count_smp].image->height
                ,Samples[count_smp].image->nChannels);
            
            //system("pause");

            count_smp++;

            //cvWaitKey(0);
            
            //cvReleaseImage( &img );
    
            //cvDestroyWindow(pic_name);

        }
        fgetc(fp);
    }
    fclose(fp);

    /////////////////////////////////////////////////////////
    if((fp=fopen(NON_FACE_TXT,"r"))==NULL)
    {
        printf("can not open the non_face.txt!\n");
        exit(0);
    }

    w=1/((double)2.0*n);
    printf("w=%f\n",w);system("pause");
    while( count_smp < m+n )
    {
        char s[35] = NON_FACE_DIR_PAH;
        //if(fgets(pic_name,14,fp)!=NULL)
        memset(pic_name, 0, sizeof(char) * 33);
        if (fscanf(fp, "%s", pic_name) != EOF)
        {
            printf("%s\n",pic_name);
            strcat(s,pic_name);
            IplImage * img = cvLoadImage(s);
            Samples[count_smp].image = cvCreateImage( cvGetSize(img), 8, 1 );
            cvCvtColor(img,Samples[count_smp].image,CV_BGR2GRAY);
            cvEqualizeHist(Samples[count_smp].image,Samples[count_smp].image);
            cvReleaseImage(&img);
            Samples[count_smp].Y = 0;
            Samples[count_smp].N = count_smp;
            Samples[count_smp].W = w;
            Samples[count_smp].IsFalseDetected = 0;
            if(!Samples[count_smp].image) printf("Could not load image file:%s\n",pic_name);

            //cvNamedWindow(pic_name, CV_WINDOW_AUTOSIZE );

            //cvShowImage(pic_name, img );

            printf("image size is %d * %d,%d\n",Samples[count_smp].image->width,Samples[count_smp].image->height
                ,Samples[count_smp].image->nChannels);

            count_smp++;

            //cvWaitKey(0);
            
            //cvReleaseImage( &img );
    
            //cvDestroyWindow(pic_name);

        }
        fgetc(fp);
    }
    
    printf("the sum of samples is %d\n",count_smp);
    /////////////////////////////////////////////////////////////////////
    ///////////读取结束
    ////////////////////////////////////////////////////////
    //计算所有样本对所有特征的特征值，实则计算的是每个特征的hy数组的值
    Cacu_Samples_Mg();

    Cal_Allsamples_FeatureValue( );

    int i;
    FILE * keep;

    if((keep=fopen(DATA_TXT,"at+"))==NULL)
    {
        printf("can not open the data.txt for write!\n");
        exit(0);
    }

    int record_weakclassifier[WEAKCLASSIFIER_NUM_MAX];
    HaarStageClassifier HSC;
    double threshold_strong = 0;

    for( i=1;i<=5;i++)
    {
        /////////////////////////////////////////////////////////
        //找出错误率最小的弱分类器

        int record = Select_WeakClassifier( );              //找到一个弱分类器
        /////////////////////////////////////////////////////
        ////////保存此级强分类器结果于文件中
        HSC.classifier[i].kind = Haarfeatures[record].kind;
        HSC.classifier[i].r = Haarfeatures[record].r;
        HSC.classifier[i].threshold = Haarfeatures[record].threshold;
        HSC.classifier[i].p = Haarfeatures[record].p;
        HSC.classifier[i].we = Haarfeatures[record].we;
        HSC.classifier[i].btt = Haarfeatures[record].btt;
        HSC.classifier[i].N = Haarfeatures[record].N;

        double bt = Haarfeatures[record].btt ;

        double at = -log(bt);

        HSC.alphat[i] = at;

        threshold_strong += at;     //强分类器阈值

        printf("----------------------%d------------------------\n",i);
    }
        
    HSC.count = i - 1 ;
    ///////////////////////////////////////////////////////////
    ///已经训练出 ni[i] 个特征，将其级联，计算检测率和误检率
    HSC.threshold = threshold_strong/2; 

    fprintf(keep,"%d %f\n",HSC.count,HSC.threshold);
    int ii;
    for(ii=1 ; ii <= HSC.count; ii++)
    {
        fprintf(keep, "%d %d %d %d %d %d %d %f\n", HSC.classifier[ii].kind, HSC.classifier[ii].r.x, HSC.classifier[ii].r.y,
            HSC.classifier[ii].r.width, HSC.classifier[ii].r.height, HSC.classifier[ii].threshold, HSC.classifier[ii].p,
            HSC.alphat[ii]);
    }

    fclose(keep);
}

//////////////////////////////////////////////////////////////////////////
/// \brief  级联多个强分类器（max: 30），每个强分类器至多包含300个弱分类器
/// \remark
/// \param[in]  Ft   级联分类器的虚警率(假正例). 被检测为人脸的非人脸窗口数目，同非人脸样本总数比值  
/// \param[in]  f    每层强分类器的虚警率
/// \param[in]  d    每层强分类器的检测率
/// \param[in]  m    训练用正样本个数
/// \param[in]  n    训练用负样本个数
//////////////////////////////////////////////////////////////////////////
void Stage_Classifier_Cascade( double Ft , double f , double d , int m , int n )
{
    Produce_AllFeatures();
    //system("pause");
    ////////////////////////////////////////////////////////////
    //读取样本图片
    
    FILE * fp;
    FILE * keep;
    char pic_name[33];

    num_samples = m + n;

    printf("breakpoint1\n");

    if((fp=fopen( FACE_TXT, "r"))==NULL)
    {
        printf("can not open the face.txt!\n");
        exit(0);
    }

    double w = 1 / ((double)2.0 * m);  //正样本权重初始化   
    printf("w=%f\n", w); 
    system("pause");

    int count_smp = 0 ;

    printf("breakpoint2\n");
    
    while( count_smp < m )
    {
        char s[30]=  FACE_DIR_PATH;
        memset(pic_name, 0, sizeof(char) * 33);
        //if(fgets(pic_name,14,fp)!=NULL)
        if (fscanf(fp, "%s", pic_name) != EOF)  //wangc add 2018/12/18
        {
            printf("%s\n",pic_name);
            strcat(s,pic_name);
            //Samples[count_smp].image = cvLoadImage(s);
            IplImage * img = cvLoadImage(s);
            Samples[count_smp].image = cvCreateImage( cvGetSize(img), 8, 1 );
            cvCvtColor(img,Samples[count_smp].image,CV_BGR2GRAY);
            cvEqualizeHist(Samples[count_smp].image,Samples[count_smp].image);
            cvReleaseImage(&img);

            printf("Width = %d \n", Samples[count_smp].image->width);

            //printf("def\n");
            Samples[count_smp].Y = 1;
            Samples[count_smp].N = count_smp;
            Samples[count_smp].W = w;  //权重  
            Samples[count_smp].IsFalseDetected = 0;
            if(!Samples[count_smp].image) 
                printf("Could not load image file:%s\n",pic_name);

            //cvNamedWindow(pic_name, CV_WINDOW_AUTOSIZE );

            //cvShowImage(pic_name, img );

            printf("image size is %d * %d,%d\n",Samples[count_smp].image->width,Samples[count_smp].image->height
                ,Samples[count_smp].image->nChannels);
            
            //system("pause");

            count_smp++;

            //cvWaitKey(0);
            
            //cvReleaseImage( &img );
    
            //cvDestroyWindow(pic_name);

        }
        fgetc(fp);
    }
    fclose(fp);
    fp = NULL;

    printf("count_smp = %d\n", count_smp);

    /////////////////////////////////////////////////////////
    if((fp=fopen( NON_FACE_TXT , "r"))==NULL)
    {
        printf("can not open the non_face.txt!\n");
        exit(0);
    }

    w = 1 / ((double)2.0*n);  //负样本权重初始化  
    printf("w=%f\n",w);
    system("pause");

    while( count_smp < m + n )
    {
        
        char s[35]= NON_FACE_DIR_PAH;
        //if(fgets(pic_name,15,fp)!=NULL)
        memset(pic_name, 0, sizeof(char) * 33);
        if (fscanf(fp, "%s", pic_name) != EOF)
        {
            //printf("%s 111",pic_name);

            strcat(s, pic_name);
            //printf("%s \n", s);
            IplImage * img = NULL;
            img = cvLoadImage(s);
            if (!img)
            {
                printf("load file fail \n");
                getchar();

            }
            //cvShowImage("nag", img);
            //cvWaitKey(1);

            Samples[count_smp].image = cvCreateImage( cvGetSize(img), 8, 1 );
            cvCvtColor(img,Samples[count_smp].image,CV_BGR2GRAY);
            cvEqualizeHist(Samples[count_smp].image,Samples[count_smp].image);
            cvReleaseImage(&img);
            Samples[count_smp].Y = 0;
            Samples[count_smp].N = count_smp;
            Samples[count_smp].W = w;
            Samples[count_smp].IsFalseDetected = 0;
            
            if(!Samples[count_smp].image)
            {
                printf("Could not load image file:%s\n",pic_name);
            }

            //printf("Width = %d \n", Samples[count_smp].image->width);

            //cvNamedWindow(pic_name, CV_WINDOW_AUTOSIZE );

            //cvShowImage(pic_name, img );

            printf("image size is %d * %d,%d\n",Samples[count_smp].image->width,Samples[count_smp].image->height
                ,Samples[count_smp].image->nChannels);

            count_smp++;

            //cvWaitKey(0);
            
            //cvReleaseImage( &img );
    
            //cvDestroyWindow(pic_name);

        }
        fgetc(fp);
    }
    
    printf("the sum of samples is %d\n",count_smp);
    /////////////////////////////////////////////////////////////////////
    ///////////读取结束
    
    printf("breakpoint3\n");
    //检测率Di:被正确检测到的人脸数与原图像内包含的人脸数的比值
    //误检率Fi:又称为虚警率或误报率。被检测为人脸的非人脸窗口数目，同非人脸样本总数比值
    double Fi[30], Di[30] ;
    int ni[30];     /// ni[i] 为第i层强分类器中弱分类器的个数，故最多有30层强分类器   
    Fi[0] = 1.0;
    Di[0] = 1.0;
    int i = 0 ;
    /*
    if((keep=fopen("E:\\test1\\data\\data.txt","w"))==NULL)
    {
        printf("can not open the data.txt!\n");
        exit(0);
    }
    */
    printf("breakpoint4\n");

    while( Fi[i] > Ft )         //Fi[i]为前 i 层级联的虚警率
    {       
        double time=(double)cvGetTickCount();

        i++;
        ni[i] = 0 ;         // ni[i] 为此层强分类器中弱分类器的个数
        Fi[i] = Fi[i-1] ;

        int record_weakclassifier[WEAKCLASSIFIER_NUM_MAX]; ///300个弱分类器  
        HaarStageClassifier HSC;  //一个强分类器
        double threshold_strong = 0;

        ////////////////////////////////////////////////////////
        //计算所有样本对所有特征的特征值
        Cacu_Samples_Mg();

        //Cal_Allsamples_FeatureValue( );

        //Cal_Allsamples_Allfeatures( );

        ////////////////////////////////////////////////////////
        //int min = 0 ;
        //第i层强分类器训练  
        while( Fi[i] > Fi[i-1] * f )//
        {
            //min++;
            Fi[i] = Fi[i-1] ;
            ni[i]++;
            HSC.count = ni[i] ;

            ////////////////////////////////////////////////////////
            ///////////////以下部分训练一个强分类器，包含 ni[i] 个 Haar 特征
            int j , k;

            //////////////////重新赋为0
            for( j = m ; j < num_samples ; j++ )
            { 
                Samples[j].IsFalseDetected = 0 ;
            }
            
            /////////////////////////////////////////////////////////
            //找出错误率最小的弱分类器
            int record = Select_WeakClassifier( );              //找到最佳弱分类器(分类误差最小)  
            /// Haarfeatures[record] 既是第record个Haar特征   

            //printf("error_rate = %f\n",error_rate);system("pause");
            //////////////////////////////////////////////////
            //对于所有特征已计算完毕，故已经找到了最小错误率及相应特征

            record_weakclassifier[ni[i]] = record;      //记录下来这个特征(弱分类器)

            //printf("第%3d个弱分类器是第%6d个HAAR特征\n",i,record);
        
            //Output_Haarfeature(record);

            //system("pause");

        
            //HSC->classifier[i] = (HaarFeature *)Haarf[record];
            {
                //HSC->classifier[i].Haar_Value = Haarf[record].Haar_Value;
                HSC.classifier[ni[i]].kind = Haarfeatures[record].kind;
                HSC.classifier[ni[i]].r = Haarfeatures[record].r;
                HSC.classifier[ni[i]].threshold = Haarfeatures[record].threshold;
                HSC.classifier[ni[i]].p = Haarfeatures[record].p;
                HSC.classifier[ni[i]].we = Haarfeatures[record].we;
                HSC.classifier[ni[i]].btt = Haarfeatures[record].btt;
                HSC.classifier[ni[i]].N = Haarfeatures[record].N;
                //HSC.classifier[i].e = Haarfeatures[record].e;
                //HSC->classifier[i].n = Haarfeatures[record].n;
                //for(int k=0; k < num_samples;k++)
                //HSC.classifier[ni[i]].Samples_FeatureValue[k] = Haarfeatures[record].Samples_FeatureValue[k];
                //Samples[k].F = Cal_HaarValue( record , k );

            }

        
            double bt = Haarfeatures[record].btt ;

            double at = -log(bt);  // log( ( 1- min_et) / min_et)

            HSC.alphat[ni[i]] = at;  //级联时该强分类器权重   

#if 0
            threshold_strong += at;  //强分类器阈值
            ///////////////////////////////////////////////////////////
            ///已经训练出 ni[i] 个特征，将其级联，计算检测率和误检率
            HSC.threshold = threshold_strong/2;
#else
            //wangc nodified 2018/12/19  
            threshold_strong += at/2;  //强分类器阈值
            ///////////////////////////////////////////////////////////
            ///已经训练出 ni[i] 个特征，将其级联，计算检测率和误检率
            HSC.threshold = threshold_strong ;
#endif
            int Dt_True = 0 ;       //检测到的正面样本数目

            int Smp_True = 0 ;      //其中真正正面样本的数目

            double ah[9000];        //全部样本图片对应的强分类器值

            //printf("break point p\n");

            for(j=0;j<num_samples;j++)          //对于所有样本图片
            {
                ah[j] = 0 ;
                
                for(k=1;k<=ni[i];k++)               //对于强分类器中每一个特征
                {
                    //因在数组 Samples 中，前 m 个为人脸样本，剩余的为非人脸样本
                    //if( (j < m && HSC.classifier[k].hy[j] == 0) || (j >= m && HSC.classifier[k].hy[j] == 1) )
                    
                    if (HSC.classifier[k].p * Cal_HaarValue(HSC.classifier[k].N, j)
                        < HSC.classifier[k].p * HSC.classifier[k].threshold)
                    {
                        ah[j] += HSC.alphat[k];            //ht(x)=1 正确分类
                    }
                }

                if( ah[j] >= HSC.threshold) //判断为人脸
                {
                    Dt_True++ ;

                    if( j < m)
                    {
                        Smp_True++;
                    }
                    else 
                    {
                        Samples[j].IsFalseDetected = 1;  ///误检    
                    }
                }
                
            }

            double ds = Smp_True * 1.0 ;

            double dt = (Dt_True - Smp_True) * 1.0 ;

            printf("fisrt:ds=%f,dt=%f,Smp_True=%d,Dt_True=%d\n",ds,dt,Smp_True,Dt_True);

            double DR = (ds/m) ;        //当前强分类器检测率     

            double FR = (dt/n) ;        //当前强分类器虚警率 

            if( DR >= d)
            {
                Di[i] = DR * Di[i-1] ;   // 级联起来的检测率

                Fi[i] = FR * Fi[i-1] ;   // 级联起来的虚警率
            }
            ////////////////////////////////////////////////////////////////
            //减小当前强分类器阈值
            else
            {

                while( DR < d )
                {
                    Dt_True = 0 ;               //检测到的正面样本数目

                    Smp_True = 0 ;              //其中真正正面样本的数目
                
                    HSC.threshold -= 0.0001 ;   //减小当前强分类器阈值

                    for(j=0;j<num_samples;j++)   //对于所有样本图片
                    {

                        if( ah[j] >= HSC.threshold) //判断为人脸
                        {
                            Dt_True++ ;

                            if( j < m)
                        
                                Smp_True++;

                            else Samples[j].IsFalseDetected = 1;
                        }
                
                    }

                    ds = Smp_True * 1.0 ;

                    dt = (Dt_True-Smp_True) * 1.0 ;

                    DR = (ds/m) ;         //当前强分类器检测率     

                    FR = (dt/n) ;         //当前强分类器虚警率

                    //printf("second:ds=%f,dt=%f,Smp_True=%d,Dt_True=%d\n",ds,dt,Smp_True,Dt_True);

                }
                printf("second:ds=%f,dt=%f,DR=%f,FR=%f\n",ds,dt,DR,FR);

                Di[i] = DR * Di[i-1] ;    // 级联起来的检测率

                Fi[i] = FR * Fi[i-1] ;    // 级联起来的虚警率
            }
            printf("DR=%f,FR=%f,Fi[%d]=%f\n",DR,FR,i,Fi[i]);
            
        }
        
        time = (double)cvGetTickCount() - time ;
        printf("--------stage%d有%d个弱分类器,训练时间为%gs--------\n",i,ni[i],time/((double)cvGetTickFrequency()*1000*1000));
        /////////////////////////////////////////////////////
        ////////保存此级强分类器结果于文件中
        if((keep=fopen(DATA_TXT,"at+"))==NULL)
        {
            printf("can not open the data.txt for write!\n");
            exit(0);
        }

        fprintf(keep,"%d %f\n",HSC.count,HSC.threshold);
        int ii;
        for(ii=1;ii<=HSC.count;ii++)
        {
            fprintf(keep,"%d %d %d %d %d %d %d %f\n",HSC.classifier[ii].kind , HSC.classifier[ii].r.x , HSC.classifier[ii].r.y ,
            HSC.classifier[ii].r.width , HSC.classifier[ii].r.height , HSC.classifier[ii].threshold , HSC.classifier[ii].p,
            HSC.alphat[ii]);
        }

        fclose(keep);
        keep = NULL;

        /////////////////////////////////////////////////////
        ///更新非人脸样本
        if((keep=fopen(DATA_TXT,"r"))==NULL)
        {
            printf("can not open the data.txt for read!\n");
            exit(0);
        }
        
        for(ii=1;ii<i;ii++) //对于每一个强分类器
        {
            fscanf(keep,"%d%lf",&HSC.count,&HSC.threshold);     //读其值到HSC
            fgetc(keep);
            printf("%d %f\n",HSC.count,HSC.threshold);
            int jj;
            for(jj=1;jj<=HSC.count;jj++)
            {
                fscanf(keep,"%d%d%d%d%d%d%d%lf",&HSC.classifier[jj].kind,&HSC.classifier[jj].r.x,&HSC.classifier[jj].r.y ,
                &HSC.classifier[jj].r.width , &HSC.classifier[jj].r.height , &HSC.classifier[jj].threshold , 
                &HSC.classifier[jj].p,&HSC.alphat[jj]);         

                printf("%d %d %d %d %d %d %d %f\n",HSC.classifier[jj].kind , HSC.classifier[jj].r.x , HSC.classifier[jj].r.y ,
                HSC.classifier[jj].r.width , HSC.classifier[jj].r.height , HSC.classifier[jj].threshold , HSC.classifier[jj].p,
                HSC.alphat[jj]);

                fgetc(keep);

            }
            int neg = m;

            while( neg < num_samples )                      //对所有的非人脸样本进行判断
            {
                if( Samples[neg].IsFalseDetected == 1 )     //被此级强分类器误判为人脸的非人脸样本
                {
                    if(Result( HSC , neg ) == 0)            //此样本判断为非人脸，表示判断正确

                        Samples[neg].IsFalseDetected = 0 ;
                }
    
                neg++;
            
            }
        }
        fclose(keep);

        int neg = m ;

        w=1/(2*(float)n);

        int temp = 0 ;
        while( neg < num_samples )
        {
            if( Samples[neg].IsFalseDetected == 0 )     //没有被此级强分类器误判为人脸的非人脸样本
            {
                char s[35]= NON_FACE_DIR_PAH ;
                //if(fgets(pic_name,14,fp)!=NULL)
                memset(pic_name, 0, sizeof(char) * 33);
                if (fscanf(fp, "%s", pic_name) != EOF)
                {
                    //printf("%s\n",pic_name);
                    strcat(s,pic_name);
                    IplImage * img = cvLoadImage(s);
                    Samples[neg].image = cvCreateImage( cvGetSize(img), 8, 1 );
                    cvCvtColor(img,Samples[neg].image,CV_BGR2GRAY);
                    cvEqualizeHist(Samples[neg].image,Samples[neg].image);
                    cvReleaseImage(&img);
                    Samples[neg].Y = 0;
                    Samples[neg].N = neg;
                    //Samples[neg].W = w;
                    //Samples[neg].IsFalseDetected = 0;
                    if(!Samples[neg].image) printf("Could not load image file:%s\n",pic_name);
                    //printf("image size is %d * %d\n",Samples[count_smp].image->width,Samples[count_smp].image->height);
                    temp++;
                }
                fgetc(fp);
            }
    
            neg++;
            
        }

        for(neg=m;neg<num_samples;neg++)
        {

            Samples[neg].W=w;
            Samples[neg].IsFalseDetected = 0 ;
        }

        printf("%s\n",pic_name);
        printf("被检测错误的非人脸样本数目为%d\n",n-temp);

        w=1/(2*(float)m);

        for( neg = 0 ; neg < m ; neg++ )

            Samples[neg].W = w;

    }

    fclose(fp);
    fclose(keep);
}


// Ft 为级联分类器的虚警率，f 为每层强分类器的虚警率，d 为每层强分类器的检测率
// m  、n 分别为训练用正负样本个数
void Stage_Classifier_Cascade2( double Ft , double f , double d , int m , int n )
{
    Produce_AllFeatures();
    //system("pause");
    ////////////////////////////////////////////////////////////
    //读取样本图片
    
    FILE * fp;
    FILE * keep;
    char pic_name[15];

    num_samples = m + n;

    printf("breakpoint1\n");

    if((fp=fopen( FACE_TXT , "r"))==NULL)
    {
        printf("can not open the face.txt!\n");
        exit(0);
    }

    double w = 1 / ((double)2.0*m);
    printf("w=%f\n",w);system("pause");

    int count_smp = 0 ;

    printf("breakpoint2\n");
    
    while( count_smp < m )
    {
        char s[30]= FACE_DIR_PATH;
        
        //if(fgets(pic_name,14,fp)!=NULL)
        memset(pic_name, 0, sizeof(char) * 33);
        if (fscanf(fp, "%s", pic_name) != EOF)
        {
            printf("%s\n",pic_name);
            strcat(s,pic_name);
            //Samples[count_smp].image = cvLoadImage(s);
            IplImage * img = cvLoadImage(s);
            Samples[count_smp].image = cvCreateImage( cvGetSize(img), 8, 1 );
            cvCvtColor(img,Samples[count_smp].image,CV_BGR2GRAY);
            cvReleaseImage(&img);
            //printf("def\n");
            Samples[count_smp].Y = 1;
            Samples[count_smp].N = count_smp;
            Samples[count_smp].W = w;
            Samples[count_smp].IsFalseDetected = 0;
            if(!Samples[count_smp].image) printf("Could not load image file:%s\n",pic_name);

            //cvNamedWindow(pic_name, CV_WINDOW_AUTOSIZE );

            //cvShowImage(pic_name, img );

            printf("image size is %d * %d,%d\n",Samples[count_smp].image->width,Samples[count_smp].image->height
                ,Samples[count_smp].image->nChannels);
            
            //system("pause");

            count_smp++;

            //cvWaitKey(0);
            
            //cvReleaseImage( &img );
    
            //cvDestroyWindow(pic_name);

        }
        fgetc(fp);
    }
    fclose(fp);

    /////////////////////////////////////////////////////////
    if((fp=fopen( NON_FACE_TXT, "r"))==NULL)
    {
        printf("can not open the non_face.txt!\n");
        exit(0);
    }

    w=1/((double)2.0*n);
    printf("w=%f\n",w);system("pause");
    while( count_smp < m+n )
    {
        char s[35]= NON_FACE_DIR_PAH;
        //if(fgets(pic_name,14,fp)!=NULL)
        memset(pic_name, 0, sizeof(char) * 33);
        if (fscanf(fp, "%s", pic_name) != EOF)
        {
            printf("%s\n",pic_name);
            strcat(s,pic_name);
            IplImage * img = cvLoadImage(s);
            Samples[count_smp].image = cvCreateImage( cvGetSize(img), 8, 1 );
            cvCvtColor(img,Samples[count_smp].image,CV_BGR2GRAY);
            cvReleaseImage(&img);
            Samples[count_smp].Y = 0;
            Samples[count_smp].N = count_smp;
            Samples[count_smp].W = w;
            Samples[count_smp].IsFalseDetected = 0;
            if(!Samples[count_smp].image) printf("Could not load image file:%s\n",pic_name);

            //cvNamedWindow(pic_name, CV_WINDOW_AUTOSIZE );

            //cvShowImage(pic_name, img );

            printf("image size is %d * %d,%d\n",Samples[count_smp].image->width,Samples[count_smp].image->height
                ,Samples[count_smp].image->nChannels);

            count_smp++;

            //cvWaitKey(0);
            
            //cvReleaseImage( &img );
    
            //cvDestroyWindow(pic_name);

        }
        fgetc(fp);
    }
    
    printf("the sum of samples is %d\n",count_smp);
    /////////////////////////////////////////////////////////////////////
    ///////////读取结束
    
    printf("breakpoint3\n");
    //检测率Di:被正确检测到的人脸数与原图像内包含的人脸数的比值
    //误检率Fi:又称为虚警率或误报率。被检测为人脸的非人脸窗口数目，同非人脸样本总数比值
    double Fi[30], Di[30] ;
    int ni[30];
    Fi[0] = 1.0;
    Di[0] = 1.0;
    int i = 0 ;
    /*
    if((keep=fopen("E:\\test1\\data\\data.txt","w"))==NULL)
    {
        printf("can not open the data.txt!\n");
        exit(0);
    }
    */
    printf("breakpoint4\n");

    while( Fi[i] > Ft )         //Fi[i]为前 i 层级联的虚警率
    {       
        double time=(double)cvGetTickCount();

        i++;
        ni[i] = 0 ;         // ni[i] 为此层强分类器中弱分类器的个数
        Fi[i] = Fi[i-1] ;

        int record_weakclassifier[WEAKCLASSIFIER_NUM_MAX];
        HaarStageClassifier HSC;
        double threshold_strong = 0;

        ////////////////////////////////////////////////////////
        //计算所有样本对所有特征的特征值，实则计算的是每个特征的hy数组的值
        Cacu_Samples_Mg();

        Cal_Allsamples_FeatureValue( );

        //Cal_Allsamples_Allfeatures( );

        ////////////////////////////////////////////////////////

        while( Fi[i] > Fi[i-1] * f )
        {
            Fi[i] = Fi[i-1] ;
            ni[i]++;
            HSC.count = ni[i] ;
            ////////////////////////////////////////////////////////
            ///////////////以下部分训练一个强分类器，包含 ni[i] 个 Haar 特征
            int j , k;
            //////////////////重新赋为0
            for( j = m ; j < num_samples ; j++ )
            
                Samples[j].IsFalseDetected = 0 ;
            
            /////////////////////////////////////////////////////////
            //找出错误率最小的弱分类器

            int record = Select_WeakClassifier( );              //找到一个弱分类器
        
            //printf("error_rate = %f\n",error_rate);system("pause");
            //////////////////////////////////////////////////
            //对于所有特征已计算完毕，故已经找到了最小错误率及相应特征

            record_weakclassifier[ni[i]] = record;      //记录下来这个特征(弱分类器)

            //printf("第%3d个弱分类器是第%6d个HAAR特征\n",i,record);
        
            //Output_Haarfeature(record);

            //system("pause");

        
            //HSC->classifier[i] = (HaarFeature *)Haarf[record];
            {
                //HSC->classifier[i].Haar_Value = Haarf[record].Haar_Value;
                HSC.classifier[ni[i]].kind = Haarfeatures[record].kind;
                HSC.classifier[ni[i]].r = Haarfeatures[record].r;
                HSC.classifier[ni[i]].threshold = Haarfeatures[record].threshold;
                HSC.classifier[ni[i]].p = Haarfeatures[record].p;
                HSC.classifier[ni[i]].we = Haarfeatures[record].we;
                HSC.classifier[ni[i]].btt = Haarfeatures[record].btt;
                HSC.classifier[ni[i]].N = Haarfeatures[record].N;
                //HSC.classifier[i].e = Haarfeatures[record].e;
                //HSC->classifier[i].n = Haarfeatures[record].n;
                //for(int k=0; k < num_samples;k++)
                //  HSC.classifier[ni[i]].Samples_FeatureValue[k] = Haarfeatures[record].Samples_FeatureValue[k];

            }

        
            double bt = Haarfeatures[record].btt ;

            double at = -log(bt);

            HSC.alphat[ni[i]] = at;

            threshold_strong += at;     //强分类器阈值

            ///////////////////////////////////////////////////////////
            ///已经训练出 ni[i] 个特征，将其级联，计算检测率和误检率
            HSC.threshold = threshold_strong/2;

            int Dt_True = 0 ;       //检测到的正面样本数目

            int Smp_True = 0 ;      //其中真正正面样本的数目

            double ah[9000];        //全部样本图片对应的强分类器值

            //printf("break point p\n");

            for(j=0;j<num_samples;j++)       //对于所有样本图片
            {
                ah[j] = 0 ;
                
                for(k=1;k<=ni[i];k++)        //对于强分类器中每一个特征
                {
                    //因在数组 Samples 中，前 m 个为人脸样本，剩余的为非人脸样本
                    //if( (j < m && HSC.classifier[k].hy[j] == 0) || (j >= m && HSC.classifier[k].hy[j] == 1) )
                    if( HSC.classifier[k].p * Cal_HaarValue( HSC.classifier[k].N , j )
                        < HSC.classifier[k].p * HSC.classifier[k].threshold)
                        
                        ah[j] += HSC.alphat[k] ;            //ht(x)=1
                }

                if( ah[j] >= HSC.threshold) //判断为人脸
                {
                    Dt_True++ ;

                    if( j < m)
                        
                        Smp_True++;

                    else Samples[j].IsFalseDetected = 1;
                }
                
            }

            double ds = Smp_True * 1.0 ;

            double dt = (Dt_True-Smp_True) * 1.0 ;

            printf("fisrt:ds=%f,dt=%f,Smp_True=%d,Dt_True=%d\n",ds,dt,Smp_True,Dt_True);

            double DR = (ds/m) ;                //当前强分类器检测率     

            double FR = (dt/n) ;                //当前强分类器虚警率 

            if( DR >= d)
            {
                Di[i] = DR * Di[i-1] ;          // 级联起来的检测率

                Fi[i] = FR * Fi[i-1] ;          // 级联起来的虚警率
            }
            
            ////////////////////////////////////////////////////////////////
            //减小当前强分类器阈值
            else
            {

                while( DR < d )
                {
                    Dt_True = 0 ;               //检测到的正面样本数目

                    Smp_True = 0 ;              //其中真正正面样本的数目
                
                    HSC.threshold -= 0.0001 ;               //减小当前强分类器阈值

                    for(j=0;j<num_samples;j++)          //对于所有样本图片
                    {

                        if( ah[j] >= HSC.threshold) //判断为人脸
                        {
                            Dt_True++ ;

                            if( j < m)
                        
                                Smp_True++;

                            else Samples[j].IsFalseDetected = 1;
                        }
                
                    }

                    ds = Smp_True * 1.0 ;

                    dt = (Dt_True-Smp_True) * 1.0 ;

                    DR = (ds/m) ;               //当前强分类器检测率     

                    FR = (dt/n) ;               //当前强分类器虚警率

                    //printf("second:ds=%f,dt=%f,Smp_True=%d,Dt_True=%d\n",ds,dt,Smp_True,Dt_True);

                }
                printf("second:ds=%f,dt=%f,DR=%f,FR=%f\n",ds,dt,DR,FR);

                Di[i] = DR * Di[i-1] ;          // 级联起来的检测率

                Fi[i] = FR * Fi[i-1] ;          // 级联起来的虚警率
            }
            printf("DR=%f,FR=%f,Fi[%d]=%f\n",DR,FR,i,Fi[i]);
            
        }
        
        time = (double)cvGetTickCount() - time ;
        printf("--------stage%d有%d个弱分类器,训练时间为%gs--------\n",i,ni[i],time/((double)cvGetTickFrequency()*1000*1000));
        /////////////////////////////////////////////////////
        ////////保存此级强分类器结果于文件中
        if((keep=fopen( DATA_TXT,"at+"))==NULL)
        {
            printf("can not open the data.txt for write!\n");
            exit(0);
        }

        fprintf(keep,"%d %f\n",HSC.count,HSC.threshold);
        int ii;
        for(ii=1;ii<=HSC.count;ii++)
        {
            fprintf(keep,"%d %d %d %d %d %d %d %f\n",HSC.classifier[ii].kind , HSC.classifier[ii].r.x , HSC.classifier[ii].r.y ,
            HSC.classifier[ii].r.width , HSC.classifier[ii].r.height , HSC.classifier[ii].threshold , HSC.classifier[ii].p,
            HSC.alphat[ii]);
        }

        fclose(keep);
        /////////////////////////////////////////////////////
        ///更新非人脸样本
        if((keep=fopen(DATA_TXT, "r"))==NULL)
        {
            printf("can not open the data.txt for read!\n");
            exit(0);
        }
        
        for(ii=1;ii<i;ii++) //对于每一个强分类器
        {
            fscanf(keep,"%d%lf",&HSC.count,&HSC.threshold);     //读其值到HSC
            fgetc(keep);
            printf("%d %f\n",HSC.count,HSC.threshold);
            int jj;
            for(jj=1;jj<=HSC.count;jj++)
            {
                fscanf(keep,"%d%d%d%d%d%d%d%lf",&HSC.classifier[jj].kind,&HSC.classifier[jj].r.x,&HSC.classifier[jj].r.y ,
                &HSC.classifier[jj].r.width , &HSC.classifier[jj].r.height , &HSC.classifier[jj].threshold , 
                &HSC.classifier[jj].p,&HSC.alphat[jj]);         

                printf("%d %d %d %d %d %d %d %f\n",HSC.classifier[jj].kind , HSC.classifier[jj].r.x , HSC.classifier[jj].r.y ,
                HSC.classifier[jj].r.width , HSC.classifier[jj].r.height , HSC.classifier[jj].threshold , HSC.classifier[jj].p,
                HSC.alphat[jj]);

                fgetc(keep);

            }
            int neg = m;

            while( neg < num_samples )                      //对所有的非人脸样本进行判断
            {
                if( Samples[neg].IsFalseDetected == 1 )     //被此级强分类器误判为人脸的非人脸样本
                {
                    if(Result( HSC , neg ) == 0)            //此样本判断为非人脸，表示判断正确

                        Samples[neg].IsFalseDetected = 0 ;
                }
    
                neg++;
            
            }
        }
        fclose(keep);

        int neg = m ;

        w=1/(2*(float)n);

        int temp = 0 ;
        while( neg < num_samples )
        {
            if( Samples[neg].IsFalseDetected == 0 )     //没有被此级强分类器误判为人脸的非人脸样本
            {
                char s[35]= NON_FACE_DIR_PAH;
                //if(fgets(pic_name,14,fp)!=NULL)
                memset(pic_name, 0, sizeof(char) * 33);
                if (fscanf(fp, "%s", pic_name) != EOF)
                {
                    //printf("%s\n",pic_name);
                    strcat(s,pic_name);
                    IplImage * img = cvLoadImage(s);
                    Samples[neg].image = cvCreateImage( cvGetSize(img), 8, 1 );
                    cvCvtColor(img,Samples[neg].image,CV_BGR2GRAY);
                    cvReleaseImage(&img);
                    Samples[neg].Y = 0;
                    Samples[neg].N = neg;
                    //Samples[neg].W = w;
                    //Samples[neg].IsFalseDetected = 0;
                    if(!Samples[neg].image) printf("Could not load image file:%s\n",pic_name);
                    //printf("image size is %d * %d\n",Samples[count_smp].image->width,Samples[count_smp].image->height);
                    temp++;
                }
                fgetc(fp);
            }
    
            neg++;
            
        }

        for(neg=m;neg<num_samples;neg++)
        {

            Samples[neg].W=w;
            Samples[neg].IsFalseDetected = 0 ;
        }

        printf("%s\n",pic_name);
        printf("被检测错误的非人脸样本数目为%d\n",n-temp);

        w=1/(2*(float)m);

        for( neg = 0 ; neg < m ; neg++ )

            Samples[neg].W = w;

    }

    fclose(fp);
    fclose(keep);
}



//n is the number of sample,Haarf is a kind of haarfeature
//used to train a weak classifier
void Cal_Allsamples_Featurej( int j )
{
    //float value[1000];
    int i;
    int t=0;        //用来记录当前最小分类错误的下标
    float ef=0;     //当前所有正样本权值之和,对应于论文中的e+
    float en=0;     //当前所有负样本权值之和,对应于论文中的e-
    float et=100;       //用来记录当前最小分类错误
    
    for(i=0;i < num_samples ; i++)
    {
        if(Samples[i].Y == 1)

            ef += Samples[i].W;

        else en += Samples[i].W;
    }

    NYFW nyfw[NUM_SAMPLES_MAX];     //暂时存储 Samples 排序需要的元素

    //用一次 every_haarfeature_value 初始化一次
    //for(i=0;i<NUM_SAMPLES_MAX;i++)
            //every_haarfeature_value[i]=0;

    for(i=0;i<num_samples;i++)
    {
        //Samples[i].F = Cal_HaarValue( Haarfeatures[j],Samples[i] );   //计算所有样本的第 j 个特征值

        //every_haarfeature_value[i] = Haarfeatures[j].Samples_FeatureValue[i];

        Samples[i].F = Cal_HaarValue( j , i ) ;

        //Haarfeatrues[j].Samples_FeatureValue[i] = Samples[i].F ;

        nyfw[i].N = i;                                              //所有样本的 N、Y、F、W 值传到 nyfw 中
        nyfw[i].F = Samples[i].F;
        nyfw[i].Y = Samples[i].Y;
        nyfw[i].W = Samples[i].W;
    }

    //qsort(e_img,n,sizeof(*e_img),comp);       //sort

    //qsort(Samples , num_samples , sizeof(Ex_IplImage) , comp);
    
    //for(i=0;i<num_samples;i++) printf("%d ",nyfw[i].F);
    //printf("\n");

    qsort(nyfw , num_samples , sizeof(NYFW) , cmp);

    //for(i=0;i<num_samples;i++) printf("%d ",nyfw[i].F);
    //printf("\n");
    //system("pause");

    for(i=0;i<num_samples;i++)
    {
        if(nyfw[i].Y==1)
        {
            ef -= nyfw[i].W;
            en += nyfw[i].W;
        }
        else
        {
            ef += nyfw[i].W;
            en -= nyfw[i].W;
        }

        float e=ef<en?ef:en;

        if(e<et)
        {
            et=e;
            t=i;
        }
    }
    
    //printf("t=%d,n=%d\n",t,nyfw[t].N);
    t=nyfw[t].N;                                //找到此样本
    Haarfeatures[j].threshold = Samples[t].F;
    Haarfeatures[j].p = (Samples[t].Y==1?1:-1);         //p的方向用样本t的正负来标志
    Haarfeatures[j].we = et;

    for(i=0;i<num_samples;i++)
    {
        int h=0;

        //if p*fj(i) < p*theata then hj(i)=1
        if(Haarfeatures[j].p * Samples[i].F < Haarfeatures[j].p * Haarfeatures[j].threshold)
            h=1;

        //Haarfeatures[j].hy[i] = abs( h - Samples[i].Y);       // | hj(xi) - yi |

    }

}

void Cal_Allsamples_Allfeatures()
{
    printf("Cal_Allsamples_Allfeatures() begin!\n");
    int i;
    for(i=1;i<=num_features;i++)
        Cal_Allsamples_Featurej( i );
    printf("Cal_Allsamples_Allfeatures() end!\n");
}

//m is the number of positive, l is the number of negative
void Stage_Classifier( )
{
    int record_weakclassifier[WEAKCLASSIFIER_NUM_MAX];
    int i;
    float threshold_strong=0;
    HaarStageClassifier HSC;
    /////////////////////////////////////////////////////////
    //初始化所有样本权值
    /*
    for(i=1;i<=m;i++)
    {
        e_img[i]->Y = 1;
        e_img[i]->W = 1/(2*m);
    }
    for(i=m+1;i<=m+l;i++)
    {
        e_img[i]->Y = 0;
        e_img[i]->W = 1/(2*l);
    }*/
    ////////////////////////////////////////////////////////
    //计算所有样本对所有特征的特征值，实则计算的是每个特征的hy数组的值

    Cal_Allsamples_Allfeatures( );

    ////////////////////////////////////////////////////////
    //进入循环，每次循环选取一个错误率最小的弱分类器（即HAAR特征），将其序号放入record_weakclassifier数组以待级联

    int t=1;        //t will be caculate later
    for(i=1;i<=t;i++)
    {
        int j;
        ////////////////////////////////////////////////////////
        //用来归一化所有样本的权值
        float sum_w=0;

        for(j = 0 ; j < num_samples ; j++)
        {
            sum_w += Samples[j].W;
            //printf("%f ",Samples[j].W);
        }
        //printf("\nsum_w=%f\n",sum_w);
        //system("pause");

        for(j = 0 ; j < num_samples ; j++)
            Samples[j].W = Samples[j].W / sum_w;
        /////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////
        //找出错误率最小的弱分类器

        float error_rate=100;   //用来记录所有弱分类器的最小错误率

        float et=0;             //中间变量，用来暂时保存所有特征的分类错误率

        int record = 1;         //用来保存所有弱分类器错误率最小的那个

        for(j=1;j <= num_features ; j++)        //对于所有的特征逐个进行计算
        {
            int count_samples;

            for(count_samples=0; count_samples < num_samples ; count_samples++) //计算出对所有样本的分类错误率

                //et += e_img[i]->W * Haarf[j]->hy[i];
            //  et += Samples[count_samples].W * (float)(Haarfeatures[j].hy[count_samples]);

            //printf("%f ",et);
            
            if(et < error_rate)         //需找出最小值，若是当前最小，则记录下来
            {
                error_rate = et;

                record = j;
            }

            et = 0;                     //et重新赋0以用于下一轮的计算

        }
        
        //printf("error_rate = %f\n",error_rate);system("pause");
        //////////////////////////////////////////////////
        //对于所有特征已计算完毕，故已经找到了最小错误率及相应特征

        record_weakclassifier[i] = record;      //记录下来这个特征(弱分类器)

        //printf("第%3d个弱分类器是第%6d个HAAR特征\n",i,record);
        
        //Output_Haarfeature(record);

        //system("pause");

        
        //HSC->classifier[i] = (HaarFeature *)Haarf[record];
        {
            //HSC->classifier[i].Haar_Value = Haarf[record].Haar_Value;
            HSC.classifier[i].kind = Haarfeatures[record].kind;
            HSC.classifier[i].r = Haarfeatures[record].r;
            HSC.classifier[i].threshold = Haarfeatures[record].threshold;
            HSC.classifier[i].p = Haarfeatures[record].p;
            HSC.classifier[i].we = Haarfeatures[record].we;
            //HSC.classifier[i].e = Haarfeatures[record].e;
            //HSC->classifier[i].n = Haarfeatures[record].n;
        //  for(int k=0; k < num_samples;k++)
            //  HSC.classifier[i].hy[k] = Haarfeatures[record].hy[k];

        }

        
        float bt = error_rate / (1 - error_rate);

        float at = log((1-error_rate)/error_rate);

        HSC.alphat[i] = at;

        threshold_strong += at;     //强分类器阈值

        for(j=0;j<num_samples;j++)  //对于所有样本图片
        {
        //  if(Haarfeatures[record].hy[j] == 0)     //表示样本被正确的分类，更新样本权值

                Samples[j].W *= bt;
        }
    }
    //////////////////////////////大循环结束////////////////////////////////
    threshold_strong/=2;
    HSC.count = t;
    HSC.threshold = threshold_strong;

    ////////////////////////////////////////////////////////////////////////
    //此级强分类器生成，将其保存于文件中，以便用于检测
    FILE * fp;

    if((fp=fopen(DATA_TXT,"w"))==NULL)
    {
        printf("can not open the data.txt!\n");
        exit(0);
    }
    
    fprintf(fp,"%d %f\n",HSC.count,HSC.threshold);
    for(i=1;i<=HSC.count;i++)
    {
        fprintf(fp,"%d %d %d %d %d %d %d %f\n",HSC.classifier[i].kind , HSC.classifier[i].r.x , HSC.classifier[i].r.y ,
            HSC.classifier[i].r.width , HSC.classifier[i].r.height , HSC.classifier[i].threshold , HSC.classifier[i].p,
            HSC.alphat[i]);
    }
    
}



// Ft 为级联分类器的虚警率，f 为每层强分类器的虚警率，d 为每层强分类器的检测率
// m  、n 分别为训练用正负样本个数
void Stage_Classifier_Cascade1( double Ft , double f , double d , int m , int n )
{
    Produce_AllFeatures();
    system("pause");
    ////////////////////////////////////////////////////////////
    //读取样本图片
    
    FILE * fp;
    FILE * keep;
    char pic_name[15];

    num_samples = m + n;

    printf("breakpoint1\n");

    if((fp=fopen( FACE_TXT,"r"))==NULL)
    {
        printf("can not open the face.txt!\n");
        exit(0);
    }

    double w = 1/((double)2.0*m);
    printf("w=%f\n",w);system("pause");

    int count_smp = 0 ;

    printf("breakpoint2\n");
    
    while( count_smp < m )
    {
        char s[30]= FACE_DIR_PATH;
        
        //if(fgets(pic_name,14,fp)!=NULL)
        memset(pic_name, 0, sizeof(char) * 33);
        if (fscanf(fp, "%s", pic_name) != EOF)
        {
            printf("%s\n",pic_name);
            strcat(s,pic_name);
            Samples[count_smp].image = cvLoadImage(s);
            Samples[count_smp].Y = 1;
            Samples[count_smp].N = count_smp;
            Samples[count_smp].W = w;
            Samples[count_smp].IsFalseDetected = 0;
            if(!Samples[count_smp].image) printf("Could not load image file:%s\n",pic_name);

            //cvNamedWindow(pic_name, CV_WINDOW_AUTOSIZE );

            //cvShowImage(pic_name, img );

            printf("image size is %d * %d\n",Samples[count_smp].image->width,Samples[count_smp].image->height);

            count_smp++;

            //cvWaitKey(0);
            
            //cvReleaseImage( &img );
    
            //cvDestroyWindow(pic_name);

        }
        fgetc(fp);
    }
    fclose(fp);

    /////////////////////////////////////////////////////////
    if((fp=fopen( NON_FACE_TXT, "r"))==NULL)
    {
        printf("can not open the non_face.txt!\n");
        exit(0);
    }

    w=1/((double)2.0*n);
    printf("w=%f\n",w);system("pause");
    while( count_smp < m+n )
    {
        char s[35]= NON_FACE_DIR_PAH;
        //if(fgets(pic_name,14,fp)!=NULL)
        memset(pic_name, 0, sizeof(char) * 33);
        if (fscanf(fp, "%s", pic_name) != EOF)
        {
            printf("%s\n",pic_name);
            strcat(s,pic_name);
            Samples[count_smp].image = cvLoadImage(s);
            Samples[count_smp].Y = 0;
            Samples[count_smp].N = count_smp;
            Samples[count_smp].W = w;
            Samples[count_smp].IsFalseDetected = 0;
            if(!Samples[count_smp].image) printf("Could not load image file:%s\n",pic_name);

            //cvNamedWindow(pic_name, CV_WINDOW_AUTOSIZE );

            //cvShowImage(pic_name, img );

            printf("image size is %d * %d\n",Samples[count_smp].image->width,Samples[count_smp].image->height);

            count_smp++;

            //cvWaitKey(0);
            
            //cvReleaseImage( &img );
    
            //cvDestroyWindow(pic_name);

        }
        fgetc(fp);
    }
    
    printf("the sum of samples is %d\n",count_smp);
    /////////////////////////////////////////////////////////////////////
    ///////////读取结束
    
    printf("breakpoint3\n");
    //检测率Di:被正确检测到的人脸数与原图像内包含的人脸数的比值
    //误检率Fi:又称为虚警率或误报率。被检测为人脸的非人脸窗口数目，同非人脸样本总数比值
    double Fi[30], Di[30] ;
    int ni[30];
    Fi[0] = 1.0;
    Di[0] = 1.0;
    int i = 0 ;

    if((keep=fopen(DATA_TXT,"w"))==NULL)
    {
        printf("can not open the data.txt!\n");
        exit(0);
    }
    
    printf("breakpoint4\n");

    while( Fi[i] > Ft )         //Fi[i]为前 i 层级联的虚警率
    {       
        i++;
        ni[i] = 0 ;         // ni[i] 为此层强分类器中弱分类器的个数
        Fi[i] = Fi[i-1] ;

        int record_weakclassifier[WEAKCLASSIFIER_NUM_MAX];
        HaarStageClassifier HSC;
        float threshold_strong = 0;

        ////////////////////////////////////////////////////////
        //计算所有样本对所有特征的特征值，实则计算的是每个特征的hy数组的值
        Cacu_Samples_Mg();

        Cal_Allsamples_FeatureValue( );

        //Cal_Allsamples_Allfeatures( );

        ////////////////////////////////////////////////////////

        while( Fi[i] > Fi[i-1] * f )
        {
            Fi[i] = Fi[i-1] ;
            ni[i]++;
            HSC.count = ni[i] ;
            ////////////////////////////////////////////////////////
            ///////////////以下部分训练一个强分类器，包含 ni[i] 个 Haar 特征
            int j , k;
            //////////////////重新赋为0
            for( j = m ; j < num_samples ; j++ )
            
                Samples[j].IsFalseDetected = 0 ;
            ////////////////////////////////////////////////////////
            //用来归一化所有样本的权值
            float sum_w=0;

            for(j = 0 ; j < num_samples ; j++)
            {
                sum_w += Samples[j].W;
                //printf("%f ",Samples[j].W);
            }
            //printf("\nsum_w=%f\n",sum_w);
            //system("pause");

            for(j = 0 ; j < num_samples ; j++)
                Samples[j].W = Samples[j].W / sum_w;
            /////////////////////////////////////////////////////////

            /////////////////////////////////////////////////////////

            /////////////////////////////////////////////////////////
            //找出错误率最小的弱分类器

            float error_rate=100;       //用来记录所有弱分类器的最小错误率

            float et=0;             //中间变量，用来暂时保存所有特征的分类错误率

            int record = 1;             //用来保存所有弱分类器错误率最小的那个

            for(j=1;j <= num_features ; j++)        //对于所有的特征逐个进行计算
            {
                int count_samples;

                for(count_samples=0; count_samples < num_samples ; count_samples++) //计算出对所有样本的分类错误率
                ;
                    //et += e_img[i]->W * Haarf[j]->hy[i];
                //  et += Samples[count_samples].W * (float)(Haarfeatures[j].hy[count_samples]);

                //printf("%f ",et);
            
                if(et < error_rate)         //需找出最小值，若是当前最小，则记录下来
                {
                    error_rate = et;

                    record = j;
                }

                et = 0;                     //et重新赋0以用于下一轮的计算

            }
        
            //printf("error_rate = %f\n",error_rate);system("pause");
            //////////////////////////////////////////////////
            //对于所有特征已计算完毕，故已经找到了最小错误率及相应特征

            record_weakclassifier[ni[i]] = record;      //记录下来这个特征(弱分类器)

            //printf("第%3d个弱分类器是第%6d个HAAR特征\n",i,record);
        
            //Output_Haarfeature(record);

            //system("pause");

        
            //HSC->classifier[i] = (HaarFeature *)Haarf[record];
            {
                //HSC->classifier[i].Haar_Value = Haarf[record].Haar_Value;
                HSC.classifier[ni[i]].kind = Haarfeatures[record].kind;
                HSC.classifier[ni[i]].r = Haarfeatures[record].r;
                HSC.classifier[ni[i]].threshold = Haarfeatures[record].threshold;
                HSC.classifier[ni[i]].p = Haarfeatures[record].p;
                HSC.classifier[ni[i]].we = Haarfeatures[record].we;
                //HSC.classifier[i].e = Haarfeatures[record].e;
                //HSC->classifier[i].n = Haarfeatures[record].n;
                //for(int k=0; k < num_samples;k++)
                    //HSC.classifier[ni[i]].hy[k] = Haarfeatures[record].hy[k];

            }

        
            float bt = error_rate / (1 - error_rate);

            float at = log((1-error_rate)/error_rate);

            HSC.alphat[ni[i]] = at;

            threshold_strong += at;     //强分类器阈值

            for(j=0;j<num_samples;j++)          //对于所有样本图片
            {
            //  if(Haarfeatures[record].hy[j] == 0)     //表示样本被正确的分类，更新样本权值

                    Samples[j].W *= bt;
            }
            ///////////////////////////////////////////////////////////
            ///已经训练出 ni[i] 个特征，将其级联，计算检测率和误检率
            HSC.threshold = threshold_strong/2;

            int Dt_True = 0 ;               //检测到的正面样本数目

            int Smp_True = 0 ;              //其中真正正面样本的数目

            float ah[10000];        //全部样本图片对应的强分类器值

            printf("break point p\n");

            for(j=0;j<num_samples;j++)          //对于所有样本图片
            {
                ah[j] = 0 ;
                
                for(k=1;k<=ni[i];k++)               //对于强分类器中每一个特征
                {
                    //因在数组 Samples 中，前 m 个为人脸样本，剩余的为非人脸样本
            //      if( (j < m && HSC.classifier[k].hy[j] == 0) || (j >= m && HSC.classifier[k].hy[j] == 1) )
                        
                        ah[j] += HSC.alphat[k] ;            //ht(x)=1
                }

                if( ah[j] >= HSC.threshold) //判断为人脸
                {
                    Dt_True++ ;

                    if( j < m)
                        
                        Smp_True++;

                    else Samples[j].IsFalseDetected = 1;
                }
                
            }

            double ds = Smp_True * 1.0 ;

            double dt = (Dt_True-Smp_True) * 1.0 ;

            printf("fisrt:ds=%f,dt=%f,Smp_True=%d,Dt_True=%d\n",ds,dt,Smp_True,Dt_True);

            double DR = (ds/m) ;                //当前强分类器检测率     

            double FR = (dt/n) ;                //当前强分类器虚警率 

            if( DR >= d)
            {
                Di[i] = DR * Di[i-1] ;          // 级联起来的检测率

                Fi[i] = FR * Fi[i-1] ;          // 级联起来的虚警率
            }
            
            ////////////////////////////////////////////////////////////////
            //减小当前强分类器阈值
            else
            {

                while( DR < d )
                {
                    Dt_True = 0 ;               //检测到的正面样本数目

                    Smp_True = 0 ;              //其中真正正面样本的数目
                
                    HSC.threshold -= 0.1 ;              //减小当前强分类器阈值

                    for(j=0;j<num_samples;j++)          //对于所有s样本图片
                    {

                        if( ah[j] >= HSC.threshold) //判断为人脸
                        {
                            Dt_True++ ;

                            if( j < m)
                        
                                Smp_True++;

                            else Samples[j].IsFalseDetected = 1;
                        }
                
                    }

                    ds = Smp_True * 1.0 ;

                    dt = (Dt_True-Smp_True) * 1.0 ;

                    DR = (ds/m) ;               //当前强分类器检测率     

                    FR = (dt/n) ;               //当前强分类器虚警率

                    //printf("second:ds=%f,dt=%f,Smp_True=%d,Dt_True=%d\n",ds,dt,Smp_True,Dt_True);

                }
                printf("second:ds=%f,dt=%f,DR=%f,FR=%f\n",ds,dt,DR,FR);

                Di[i] = DR * Di[i-1] ;          // 级联起来的检测率

                Fi[i] = FR * Fi[i-1] ;          // 级联起来的虚警率
            }
            printf("DR=%f,FR=%f,Fi[%d]=%f\n",DR,FR,i,Fi[i]);
            
        }
        
        printf("stage%d--------------------------------\n",i);
        /////////////////////////////////////////////////////
        ////////保存此级强分类器结果于文件中
        fprintf(keep,"%d %f\n",HSC.count,HSC.threshold);
        int ii;
        for(ii=1;ii<=HSC.count;ii++)
        {
            fprintf(keep,"%d %d %d %d %d %d %d %f\n",HSC.classifier[ii].kind , HSC.classifier[ii].r.x , HSC.classifier[ii].r.y ,
            HSC.classifier[ii].r.width , HSC.classifier[ii].r.height , HSC.classifier[ii].threshold , HSC.classifier[ii].p,
            HSC.alphat[ii]);
        }


        /////////////////////////////////////////////////////
        ///更新非人脸样本

        int neg = m ;

        w=1/(2*(float)n);

        int temp = 0 ;
        while( neg < num_samples )
        {
            if( Samples[neg].IsFalseDetected == 0 )     //从文件夹中读取非人脸样本
            {
                char s[35]= NON_FACE_DIR_PAH;
                //if(fgets(pic_name,14,fp)!=NULL)
                memset(pic_name, 0, sizeof(char) * 33);
                if (fscanf(fp, "%s", pic_name) != EOF)
                {
                    //printf("%s\n",pic_name);
                    strcat(s,pic_name);
                    Samples[neg].image = cvLoadImage(s);
                    Samples[neg].Y = 0;
                    Samples[neg].N = neg;
                    Samples[neg].W = w;
                    Samples[neg].IsFalseDetected = 0;
                    if(!Samples[neg].image) printf("Could not load image file:%s\n",pic_name);
                    //printf("image size is %d * %d\n",Samples[count_smp].image->width,Samples[count_smp].image->height);
                    temp++;
                }
                fgetc(fp);
            }
    
            neg++;
            
        }
        printf("%s\n",pic_name);
        printf("被检测错误的非人脸样本数目为%d\n",n-temp);

        w=1/(2*(float)m);

        for( neg = 0 ; neg < m ; neg++ )

            Samples[neg].W = w;

    }

    fclose(fp);
    fclose(keep);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/***********************************以下为检测部分******************************************************/

//此函数用于计算被检图像的灰度积分，将结果放入 Image_Mg 数组中

void Cacu_Image_Mg( IplImage * img )
{
    //printf("Cacu_Image_Mg() running begin!\n");

    int i,j;
    int step = img->widthStep/sizeof(uchar);

    uchar * data = (uchar*)(img->imageData);

    int pt[2000][2000];

    for(i=0;i< 2000 ;i++)
        for(j=0;j< 2000 ;j++)
        {
            pt[i][j]=0;
            Image_Mg[i][j]=0;
        }

    int t;

    for(i = 0; i < img->height; i++)                //pt[i][-1]=0, p(x,y)为每一行的累加值
    {
        t = data[i * step];

        pt[i][0] = t;
    }//
        //printf("=========================================\n");

    for(i=0;i< img->height;i++)             //pt[x][y]=pt[x][y-1]+f[x][y],每一行累加和
        for(j=1;j < img->width;j++)
        {
            t=data[i*step+j];
            //cvmSet(p,i,j,cvmGet(p,i,j-1)+t);
            pt[i][j]=pt[i][j-1]+t;

        }
        //printf("=========================================\n");

    for(j=0;j < img->width;j++)//g[-1][y]=0;

            Image_Mg[0][j] = pt[0][j];

        //printf("=========================================\n");

        for(i=1;i<img->height;i++)              //g[x][y]=g[x-1][y]+p[x][y]
            for(j=0;j< img->width;j++)
            {
                Image_Mg[i][j] = Image_Mg[i-1][j] + pt[i][j];
            }
        /*
        printf("width=%d,height=%d\n",img->width,img->height);
        cvNamedWindow("display",CV_WINDOW_AUTOSIZE);

        cvShowImage("display",img);

        cvWaitKey(0);
        
        printf("source image:\n");
        for(i=0;i<img->height;i++)
        {
            for(j=0;j<img->width;j++)
                printf("%3d ",data[i * step + j]);
            printf("\n");
        }
        printf("jifen image:\n");
        for(i=0;i<img->height;i++)
        {
            for(j=0;j<img->width;j++)
                printf("%d ",Image_Mg[i][j]);
            printf("\n");
        }
        //printf("Mg[i-1][j-1]=%f\n",cvmGet(img->Mg,imgc->height-1,imgc->width-1));
        printf("=========================================\n");
        system("pause");
        //cvWaitKey(0);
        */
    
    //printf("Cacu_Image_Mg() running end!\n");
}



//计算被检图像中某个矩形框 r 的灰度积分
int Cal_Image_Rect_Value( CvRect r )
{

    int i4 = Image_Mg[r.y + r.height-1][r.x + r.width-1];

    if(r.x==0 && r.y==0)
        return i4;

    if(r.x==0)
        return i4-Image_Mg[r.y-1][r.width-1];

    if(r.y==0)
        return i4-Image_Mg[r.height-1][r.x-1];

    return i4 + Image_Mg[r.y-1][r.x-1] - Image_Mg[r.y-1][r.x+r.width-1] - Image_Mg[r.y+r.height-1][r.x-1];

}



///////////此函数用来计算被检图像的一个HAAR特征值
int Cal_Pic_HaarValue(int kind , CvRect r )
{
    //CvMat * Mg=e_img->Mg;
    //int kind=Haarfeature.kind;
    //CvRect r=Haarfeature.r;
    int x=r.x;
    int y=r.y;
    int width=r.width;
    int height=r.height;
    if(kind==1)
    {
        return Cal_Image_Rect_Value(r) - 2 * Cal_Image_Rect_Value( cvRect(x,y,width/2,height));
    }
    else if(kind==2)
    {
        return Cal_Image_Rect_Value(r) - 2 * Cal_Image_Rect_Value( cvRect(x,y+height/2,width,height/2));
    }
    else if(kind==3)
    {
        return Cal_Image_Rect_Value(r) - 3 * Cal_Image_Rect_Value( cvRect(x+width/3,y,width/3,height));
    }
    else if(kind==4)
    {
        return Cal_Image_Rect_Value(r) - 3 * Cal_Image_Rect_Value( cvRect(x,y+height/3,width,height/3));
    }
    else if(kind==5)
    {
        return Cal_Image_Rect_Value(r) - 2 * Cal_Image_Rect_Value( cvRect(x,y,width/2,height/2))
            
            - 2 * Cal_Image_Rect_Value( cvRect(x+width/2,y+height/2,width/2,height/2));
    }
    else if(kind==6)
    {
        return Cal_Image_Rect_Value(r) - 9 * Cal_Image_Rect_Value( cvRect(x+width/3,y+height/3,width/3,height/3) );
            
    }
    else if(kind==7)
    {
        return Cal_Image_Rect_Value(r) - 9 * Cal_Image_Rect_Value( cvRect(x,y,width/3,height/3) ) / 4
            
            - 9 * Cal_Image_Rect_Value( cvRect(x+width/3,y+height/3,width/3,2*height/3) ) / 4
            
            - 9 * Cal_Image_Rect_Value( cvRect(x+2*width/3,y,width/3,height/3) ) / 4;
    }
    else if(kind==8)
    {
        return Cal_Image_Rect_Value(r) - 3 * Cal_Image_Rect_Value( cvRect(x,y,width/3,height/3) )
            
            - 3 * Cal_Image_Rect_Value( cvRect(x+2*width/3,y,width/3,height/3))

            - 3 * Cal_Image_Rect_Value( cvRect(x+width/3,y+2*height/3,width/3,height/3) );
    }
    else if(kind==9)
    {
        return Cal_Image_Rect_Value(r) - 9 * Cal_Image_Rect_Value( cvRect(x+width/3,y,width/3,2*height/3) ) / 5
            
            - 9 * Cal_Image_Rect_Value( cvRect(x,y+2*height/3,width,height/3) ) / 5;
    }
    return 0;
}



//此函数用来计算两个矩形的重合部分，若无返回空
CvRect Common_Rect( const CvRect rect1, const CvRect rect2 )
{
    CvPoint p11 = cvPoint( rect1.x , rect1.y );                             //矩形 rect1 左上顶点
    CvPoint p12 = cvPoint( rect1.x + rect1.width, rect1.y + rect1.height);  //矩形 rect1 右下顶点
    CvPoint p21 = cvPoint( rect2.x , rect2.y );                             //矩形 rect2 左上顶点
    CvPoint p22 = cvPoint( rect2.x + rect2.width, rect2.y + rect2.height);  //矩形 rect2 右下顶点
    int x1,y1,x2,y2;
  
    if(p11.x > p22.x || p11.y > p22.y || p12.x < p21.x || p12.y < p21.y)    //两个矩形没有相交   
    {      
        return cvRect(0,0,0,0);   
    }
    else
    {
        x1 = p11.x>p21.x?p11.x:p21.x;
        y1 = p11.y>p21.y?p11.y:p21.y;
        x2 = p12.x>p22.x?p22.x:p12.x;
        y2 = p12.y>p22.y?p22.y:p12.y;
    }
    CvRect r = cvRect( x1 , y1 , x2-x1 , y2-y1 );
    return r;
}

bool Merge( const CvRect rect1, const CvRect rect2 )
{
    CvRect r = Common_Rect(rect1, rect2 );
    if(r.width*r.height*2>=rect1.width*rect1.height) return true;
    return false;
}


void Merge_SameSize_Rect( CvRect * rs , int n ,bool IsMerged[])
{
    //bool IsMerged[1000];
    int i,j;
    bool finished = false;

    for(i=0;i<n;i++)
        IsMerged[i] = false;

    int count = 1;
    while(!finished)
    {
        printf("finished=%d,count=%d\n",finished?1:0,count++);
        finished = true;
        for(i=0;i<n;i++)
        {
            if( IsMerged[i] == false)
            {
                for(j=0;j<n;j++)
                    if( IsMerged[j] == false && j != i && Merge(rs[i],rs[j]))
                    {
                        rs[i].x = (rs[i].x + rs[j].x )/ 2;
                        rs[i].y = (rs[i].y + rs[j].y )/ 2;
                        IsMerged[j] = true;
                        finished = false;
                        printf("j=%d\n",j);
                    }
            }
        }
    }
    printf("while loop end!\n");
}



static int is_equal( const void* _r1, const void* _r2, void* )
{
    const CvRect* r1 = (const CvRect*)_r1;
    const CvRect* r2 = (const CvRect*)_r2;
    int distance = cvRound(r1->width*0.2);

    return r2->x <= r1->x + distance &&
           r2->x >= r1->x - distance &&
           r2->y <= r1->y + distance &&
           r2->y >= r1->y - distance &&
           r2->width <= cvRound( r1->width * 1.2 ) &&
           cvRound( r2->width * 1.2 ) >= r1->width;
}


//给出一个矩形序列 rs ，将其合并 , 结果放入 result_seq 中返回
CvSeq * Merge( CvRect * rs , int count )
{

    CvSeq* seq = 0;
    CvSeq* seq2 = 0;
    CvSeq* idx_seq = 0;
    CvSeq* result_seq = 0;
    CvMemStorage* temp_storage = 0;
    CvMemStorage* storage = 0;
    CvAvgComp* comps = 0;
    int i;
    int min_neighbors = 1;
    
    //CV_CALL( temp_storage = cvCreateChildMemStorage( storage ));
    temp_storage = cvCreateMemStorage(0);
    storage = cvCreateMemStorage(0);
    seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvRect), temp_storage );
    seq2 = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvAvgComp), temp_storage );
    result_seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvAvgComp), storage );
    
    //if( min_neighbors == 0 )
    //    seq = result_seq;
        
    //CvRect rect = cvRect(ix,iy,win_size.width,win_size.height);
    //cvSeqPush( seq, &rect );

    for(i=0;i<count;i++)
    {
        cvSeqPush( seq, &rs[i] );
    }

    if( min_neighbors != 0 )
    {
        // group retrieved rectangles in order to filter out noise 
        int ncomp = cvSeqPartition( seq, 0, &idx_seq, is_equal, 0 );
        comps = (CvAvgComp*)cvAlloc( (ncomp+1)*sizeof(comps[0]));
        memset( comps, 0, (ncomp+1)*sizeof(comps[0]));

        // count number of neighbors
        for( i = 0; i < seq->total; i++ )
        {
            CvRect r1 = *(CvRect*)cvGetSeqElem( seq, i );
            int idx = *(int*)cvGetSeqElem( idx_seq, i );
            assert( (unsigned)idx < (unsigned)ncomp );

            comps[idx].neighbors++;
             
            comps[idx].rect.x += r1.x;
            comps[idx].rect.y += r1.y;
            comps[idx].rect.width += r1.width;
            comps[idx].rect.height += r1.height;
        }

        // calculate average bounding box
        for( i = 0; i < ncomp; i++ )
        {
            int n = comps[i].neighbors;
            if( n >= min_neighbors )
            {
                CvAvgComp comp;
                comp.rect.x = (comps[i].rect.x*2 + n)/(2*n);
                comp.rect.y = (comps[i].rect.y*2 + n)/(2*n);
                comp.rect.width = (comps[i].rect.width*2 + n)/(2*n);
                comp.rect.height = (comps[i].rect.height*2 + n)/(2*n);
                comp.neighbors = comps[i].neighbors;

                cvSeqPush( seq2, &comp );
            }
        }

        // filter out small face rectangles inside large face rectangles
        for( i = 0; i < seq2->total; i++ )
        {
            CvAvgComp r1 = *(CvAvgComp*)cvGetSeqElem( seq2, i );
            int j, flag = 1;

            for( j = 0; j < seq2->total; j++ )
            {
                CvAvgComp r2 = *(CvAvgComp*)cvGetSeqElem( seq2, j );
                int distance = cvRound( r2.rect.width * 0.2 );
            
                if( i != j &&
                    r1.rect.x >= r2.rect.x - distance &&
                    r1.rect.y >= r2.rect.y - distance &&
                    r1.rect.x + r1.rect.width <= r2.rect.x + r2.rect.width + distance &&
                    r1.rect.y + r1.rect.height <= r2.rect.y + r2.rect.height + distance
                    && (r2.neighbors > MAX( 3, r1.neighbors ) || r1.neighbors < 3) )
                {
                    flag = 0;
                    break;
                }
            }

            if( flag )
            {
                cvSeqPush( result_seq, &r1 );
                /* cvSeqPush( result_seq, &r1.rect ); */
            }
        }
    }
    return result_seq;

}

void Detect1( IplImage * img )
{
    int i,j;

    HaarFeature Haarf[WEAKCLASSIFIER_NUM_MAX];

    float at[WEAKCLASSIFIER_NUM_MAX];

    const int k = 2 ;

    float s;        // s 为检测窗口放大倍数,假设每次放大 1.25 倍

    int x,y;

    int t;

    FILE * fp;

    int n;

    float ts;

    if((fp=fopen( DATA_TXT,"r"))==NULL)
    {
        printf("can not open the data.txt!\n");
        exit(0);
    }
    
    fscanf(fp,"%d%f",&n,&ts);

    printf("%d %f\n",n,ts);

    for(i=1;i<=n;i++)
    {
        fscanf(fp,"%d%d%d%d%d%d%d%f",&Haarf[i].kind , &Haarf[i].r.x , &Haarf[i].r.y ,
            &Haarf[i].r.width , &Haarf[i].r.height , &Haarf[i].threshold , &Haarf[i].p,
            &at[i]);
        //printf("%d %d %d %d %d %d %d %f\n",Haarf[i].kind , Haarf[i].r.x , Haarf[i].r.y ,
            //Haarf[i].r.width , Haarf[i].r.height , Haarf[i].threshold , Haarf[i].p,
            //at[i]);
        fgetc(fp);
    }
    //system("pause");
    printf("//////////////////////////////\n");

    Cacu_Image_Mg( img );

    printf("..............................\n");

    int smaller = (img->width<img->height)?img->width:img->height;  //图像的长宽较小者

    CvRect  rs[100000];

    int count = 0;

    bool Ismerged[10000];

    for( s = 1 ; s * (float)Sample_W < (float)smaller ; s *= 1.2 )
    {
        //count = 0;

        for( x = 0 ; x <= img->width - s * (float)Sample_W ; x += int(k * s))       //把检测窗口的左上顶点放在待检图像的(x,y)坐标上

            for( y = 0 ; y <= img->height - s * (float)Sample_H ; y += int(k * s))
            {
                
                float w = 0 ;
                    
                for( t = 1 ; t <= n ; t++ )                         //对于强分类器HSC里的所有弱分类器
                {
                    CvRect r = cvRect( x + Haarf[t].r.x * s , y + Haarf[t].r.y * s , Haarf[t].r.width * s , Haarf[t].r.height * s );

                    int hvalue = Cal_Pic_HaarValue( Haarf[t].kind , r );
                    
                    /*
                    for(i=1;i<=n;i++)
                    {
                        printf("%d %d %d %d %d %d %d\n",Haarf[i].kind , Haarf[i].r.x , Haarf[i].r.y ,
                            Haarf[i].r.width , Haarf[i].r.height , Haarf[i].threshold , Haarf[i].p
                            );
                        system("pause");
                    }
                    */
                    //printf("r = (%d,%d,%d,%d)\n",r.x,r.y,r.width,r.height);

                    int ht = 0;

                    if( Haarf[t].p * hvalue < Haarf[t].p * Haarf[t].threshold * s * s)

                        ht = 1;
                    /*
                    if(hvalue!=0)
                    {
                        printf("hvalue=%d,later=%f\n",Haarf[t].p * hvalue,Haarf[t].threshold * s * s);
                        system("pause");
                    }
                    */

                    w += ht * at[t];
                }
                /*
                if(w>0)
                {
                    printf("w=%f\n",w);
                    system("pause");
                }
                */

                if( w >= ts )       //通过此级强分类器
                {
                    printf("==================================%d\n",count);

                    //cvRectangle( img, cvPoint(x,y), cvPoint(x+s * Sample_W,y+s * Sample_W), CV_RGB(255,0,0), 1);
                    
                    rs[count++] = cvRect(x,y,s * Sample_W,s * Sample_H);
                } 

            }
            //printf("s=%f\n",s);
            //printf("aaaaaaaaaaaaaaaaaaaaaaaa\n");
            //Merge_SameSize_Rect( rs , count , Ismerged );
            //printf("bbbbbbbbbbbbbbbbbbbbbbbb\n");
            /*
            for(i=0;i<count;i++)
            {
                if(Ismerged[i] == false)
                    cvRectangle( img, cvPoint(rs[i].x,rs[i].y), cvPoint(rs[i].x+s * Sample_W,rs[i].y+s * Sample_H), CV_RGB(255,0,0), 1);
            }
            
            cvNamedWindow("detecting",CV_WINDOW_AUTOSIZE);

            cvShowImage("detecting",img);

            cvWaitKey(0);

            //cvReleaseImage( &himg );
            cvDestroyWindow("detecting");
            //system("pause");*/


    }

    
    CvSeq * faces = Merge( rs , count );
    static CvScalar colors[] = 
    {
        {{0,0,255}},
        {{0,128,255}},
        {{0,255,255}},
        {{0,255,0}},
        {{255,128,0}},
        {{255,255,0}},
        {{255,0,0}},
        {{255,0,255}}
    };
    
    double scale = 1.3;
    for( i = 0; i < (faces ? faces->total : 0); i++ )
        {
            CvRect* r = (CvRect*)cvGetSeqElem( faces, i );//函数 cvGetSeqElem 查找序列中索引所指定的元素，并返回指向该元素的指针
            CvPoint center;
            int radius;
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);
            radius = cvRound((r->width + r->height)*0.25*scale);
            cvCircle( img, center, radius, colors[i%8], 3, 8, 0 );
        }
    cvShowImage( "result", img );
    cvWaitKey(0);
    //cvReleaseImage( &small_img );

}

void Read_Samples()
{
    int Detect_Samples( IplImage * img );
    FILE * fp;
    char pic_name[15];

    if((fp=fopen(FACE1_TXT,"r"))==NULL)
    {
        printf("can not open the face.txt!\n");
        exit(0);
    }

    int count_smp = 0 ;
    int count=0;
    char red[100][15];
    int i=1;
    while( count_smp < 2000 )
    {
        char s[30]=FACE_DIR_PATH ;
        
        //if(fgets(pic_name,14,fp)!=NULL)
        memset(pic_name, 0, sizeof(char) * 33);
        if (fscanf(fp, "%s", pic_name) != EOF)
        {
            //printf("%s\n",pic_name);
            strcat(s,pic_name);
            //Samples[count_smp].image = cvLoadImage(s);
            IplImage * img = cvLoadImage(s);
            IplImage * image = cvCreateImage( cvGetSize(img), 8, 1 );
            cvCvtColor(img,image,CV_BGR2GRAY);
            cvEqualizeHist(image,image);
            cvReleaseImage(&img);
            if(Detect_Samples(image)==1)
                count++;
            else //red[i++]=pic_name;
            strcpy(red[i++],pic_name);
            printf("%d\n",count_smp);
        }
        fgetc(fp);
        count_smp++;
    }
    fclose(fp);
    printf("样本中被检测的人脸数目为%d，检测率为%lf\n",count,(double)count/2000);
    for(int j=1;j<i;j++)
        printf("%s\n",red[j]);
}
int Detect_Samples( IplImage * img )
{
    int i,j;

    HaarStageClassifier HSC[20] ;

    //HaarFeature Haarf[WEAKCLASSIFIER_NUM_MAX];

    //float at[WEAKCLASSIFIER_NUM_MAX];

    const int k = 2 ;

    double s;       // s 为检测窗口放大倍数,假设每次放大 1.25 倍

    int x,y;

    int t;

    FILE * fp;

    int n;

    double ts;
    
    //printf("breakpoint0\n");
    ////////////////////////////////////////////////////////////
    /////此部分为从文件中读取级联分类器

    if((fp=fopen(DATA_TXT,"r"))==NULL)
    {
        printf("can not open the data.txt!\n");
        exit(0);
    }
    
    i = 1 ;

    while(!feof(fp))
    {
        fscanf(fp,"%d%lf",&HSC[i].count,&HSC[i].threshold);

        //printf("%d %f\n",HSC[i].count,HSC[i].threshold) ;

        fgetc(fp);

        for( j = 1 ; j <= HSC[i].count ; j++ )
        {
            fscanf(fp,"%d%d%d%d%d%d%d%lf",&HSC[i].classifier[j].kind , &HSC[i].classifier[j].r.x , &HSC[i].classifier[j].r.y ,
            &HSC[i].classifier[j].r.width , &HSC[i].classifier[j].r.height , &HSC[i].classifier[j].threshold , 
            &HSC[i].classifier[j].p , &HSC[i].alphat[j]);
            
            //printf("%d %d %d %d %d %d %d %f\n",HSC[i].classifier[j].kind , HSC[i].classifier[j].r.x , HSC[i].classifier[j].r.y ,
            //HSC[i].classifier[j].r.width , HSC[i].classifier[j].r.height , HSC[i].classifier[j].threshold , 
            //HSC[i].classifier[j].p , HSC[i].alphat[j]);
            
            fgetc(fp);
        }

        i++;

    }

    fclose(fp);
    int num_stage = i-1 ; 

    ////////////读取结束
    /////////////////////////////////////////////////////////

    //printf("//////////////////////////////\n");

    Cacu_Image_Mg( img );

//  printf("..............................\n");

    int smaller = (img->width<img->height)?img->width:img->height;  //图像的长宽较小者

    //printf("breakpoint1\n");

    CvRect  rs1[1000000];

    CvRect  rs2[1000000];

    //printf("breakpoint2\n");

    int count_window = 0;

    i=1;
    
    //system("pause");
    ///////////////////////////////////////////////////////////////////
    /////////以参数 s 和 k 得到所有的待测窗口

    for( s = 1 ; s * (double)Sample_W <= (double)smaller+1 ; s *= 1.2 )
    {
        //count = 0;

        for( x = 0 ; x <= img->width - cvRound(s * (double)Sample_W)+1 ; x += cvRound(k * s))//把检测窗口的左上顶点放在待检图像的(x,y)坐标上

            for( y = 0 ; y <= img->height - cvRound(s * (double)Sample_H)+1 ; y += cvRound(k * s))
            {
                
                rs2[i] = cvRect( x , y , cvRound( Sample_W * s ), cvRound(Sample_H * s ) );

                i++;

            }
    }
    //printf("count_window1=%d\n",i-1);
    //printf("breakpoint3\n");system("pause");
    count_window = i - 1 ;
    //////////////////////////////////////////////////////////////////
    ///用每级强分类器检测

    int ct;

    int cn;

    for( i = 1 ; i <= num_stage ; i++ )             //对于每个强分类器
    {
        for( j = 1 ; j <= count_window ; j++ )

            rs1[j] = rs2[j] ;

        cn = 1 ;

        for( j = 1 ; j <= count_window ; j++ )      //对于每个待检窗口
        {
            double s = (double)rs1[j].width/Sample_W ;

            double w = 0 ;

            for( ct = 1 ; ct <= HSC[i].count ; ct++ )
            {
                //printf("CVRect:%d %d %d %d %d\n",HSC[i].classifier[ct].kind,HSC[i].classifier[ct].r.x,
                    //HSC[i].classifier[ct].r.y,HSC[i].classifier[ct].r.width,HSC[i].classifier[ct].r.height);
                //printf("rs1[j]:%d %d %d %d\n",rs1[j].x,rs1[j].y,rs1[j].width,rs1[j].height);

                CvRect r = cvRect( rs1[j].x + cvRound(HSC[i].classifier[ct].r.x * s ), rs1[j].y + cvRound(HSC[i].classifier[ct].r.y * s)  , 
                    cvRound(HSC[i].classifier[ct].r.width * s ), cvRound(HSC[i].classifier[ct].r.height * s ));

                int hvalue = Cal_Pic_HaarValue( HSC[i].classifier[ct].kind , r );

                int ht = 0;

                if( HSC[i].classifier[ct].p * hvalue < HSC[i].classifier[ct].p * HSC[i].classifier[ct].threshold * s * s)

                    ht = 1;

                w += ht * HSC[i].alphat[ct];
            }

            if( w >= HSC[i].threshold )         // rs2 用来暂时保存通过此强分类器的窗口

                rs2[cn++] = rs1[j] ;
        }

        count_window = cn - 1 ;
    }

    printf("count_window=%d\n",count_window);
    return count_window;
}

void Detect( IplImage * img )
{
    int i,j;

    HaarStageClassifier HSC[20] ;

    //HaarFeature Haarf[WEAKCLASSIFIER_NUM_MAX];

    //float at[WEAKCLASSIFIER_NUM_MAX];

    const int k = 2 ;

    double s;       // s 为检测窗口放大倍数,假设每次放大 1.25 倍

    int x,y;

    int t;

    FILE * fp;

    int n;

    double ts;
    
    //printf("breakpoint0\n");
    ////////////////////////////////////////////////////////////
    /////此部分为从文件中读取级联分类器

    if((fp=fopen(DATA_220_TXT, "r"))==NULL)
    {
        printf("can not open the data.txt!\n");
        exit(0);
    }
    
    i = 1 ;

    while(!feof(fp))
    {
        fscanf(fp,"%d%lf",&HSC[i].count,&HSC[i].threshold);

        printf("%d %f\n",HSC[i].count,HSC[i].threshold) ;

        fgetc(fp);

        for( j = 1 ; j <= HSC[i].count ; j++ )
        {
            fscanf(fp,"%d%d%d%d%d%d%d%lf",&HSC[i].classifier[j].kind , &HSC[i].classifier[j].r.x , &HSC[i].classifier[j].r.y ,
            &HSC[i].classifier[j].r.width , &HSC[i].classifier[j].r.height , &HSC[i].classifier[j].threshold , 
            &HSC[i].classifier[j].p , &HSC[i].alphat[j]);
            
            printf("%d %d %d %d %d %d %d %f\n",HSC[i].classifier[j].kind , HSC[i].classifier[j].r.x , HSC[i].classifier[j].r.y ,
            HSC[i].classifier[j].r.width , HSC[i].classifier[j].r.height , HSC[i].classifier[j].threshold , 
            HSC[i].classifier[j].p , HSC[i].alphat[j]);
            
            fgetc(fp);
        }

        i++;

    }

    int num_stage = i-1 ; 

    ////////////读取结束
    /////////////////////////////////////////////////////////

    //printf("//////////////////////////////\n");

    Cacu_Image_Mg( img );

    //printf("..............................\n");

    int smaller = (img->width<img->height)?img->width:img->height;  //图像的长宽较小者

    //printf("breakpoint1\n");

    CvRect  rs1[1000000];

    CvRect  rs2[1000000];

    //printf("breakpoint2\n");

    int count_window = 0;

    i=1;
    
    //system("pause");
    ///////////////////////////////////////////////////////////////////
    /////////以参数 s 和 k 得到所有的待测窗口

    for( s = 1 ; s * (double)Sample_W <= (double)smaller+1 ; s *= 1.1 )
    {
        //count = 0;

        for( x = 0 ; x <= img->width - cvRound(s * (double)Sample_W)+1 ; x += cvRound(k * s))//把检测窗口的左上顶点放在待检图像的(x,y)坐标上

            for( y = 0 ; y <= img->height - cvRound(s * (double)Sample_H)+1 ; y += cvRound(k * s))//
            {
                
                rs2[i] = cvRect( x , y , cvRound( Sample_W * s ), cvRound(Sample_H * s ) );

                i++;

            }
    }
    printf("count_window1=%d\n",i-1);
    //printf("breakpoint3\n");system("pause");
    count_window = i - 1 ;
    //////////////////////////////////////////////////////////////////
    ///用每级强分类器检测

    int ct;

    int cn;

    for( i = 1 ; i <= num_stage ; i++ )             //对于每个强分类器
    {
        for( j = 1 ; j <= count_window ; j++ )

            rs1[j] = rs2[j] ;

        cn = 1 ;

        for( j = 1 ; j <= count_window ; j++ )      //对于每个待检窗口
        {
            double s = (double)rs1[j].width/Sample_W ;

            double w = 0 ;

            for( ct = 1 ; ct <= HSC[i].count ; ct++ )
            {
                //printf("CVRect:%d %d %d %d %d\n",HSC[i].classifier[ct].kind,HSC[i].classifier[ct].r.x,
                //  HSC[i].classifier[ct].r.y,HSC[i].classifier[ct].r.width,HSC[i].classifier[ct].r.height);
                //printf("rs1[j]:%d %d %d %d\n",rs1[j].x,rs1[j].y,rs1[j].width,rs1[j].height);

                CvRect r = cvRect( rs1[j].x + cvRound(HSC[i].classifier[ct].r.x * s ), rs1[j].y + cvRound(HSC[i].classifier[ct].r.y * s)  , 
                    cvRound(HSC[i].classifier[ct].r.width * s ), cvRound(HSC[i].classifier[ct].r.height * s ));

                int hvalue = Cal_Pic_HaarValue( HSC[i].classifier[ct].kind , r );

                int ht = 0;

                if( HSC[i].classifier[ct].p * hvalue < HSC[i].classifier[ct].p * HSC[i].classifier[ct].threshold * s * s)

                    ht = 1;

                w += ht * HSC[i].alphat[ct];
            }

            if( w >= HSC[i].threshold )         // rs2 用来暂时保存通过此强分类器的窗口

                rs2[cn++] = rs1[j] ;
        }

        count_window = cn - 1 ;
    }

    printf("count_window=%d\n",count_window);
    for(i=1;i<=count_window;i++)
    {
    cvRectangle( img, cvPoint(rs2[i].x,rs2[i].y), cvPoint(rs2[i].x+rs2[i].width,rs2[i].y+rs2[i].height), CV_RGB(255,0,0), 1);
    }
    cvNamedWindow("result",CV_WINDOW_AUTOSIZE);
    cvShowImage( "result", img );
    cvWaitKey(0);
    printf("*****************************************\n");
    CvSeq * faces = Merge( rs2 , count_window );
    printf("faces=%d\n",faces->total);
    
    system("pause");
    static CvScalar colors[] = 
    {
        {{0,0,255}},
        {{0,128,255}},
        {{0,255,255}},
        {{0,255,0}},
        {{255,128,0}},
        {{255,255,0}},
        {{255,0,0}},
        {{255,0,255}}
    };
    
    double scale = 1.1;
    for( i = 0; i < (faces ? faces->total : 0); i++ )
        {
            CvRect* r = (CvRect*)cvGetSeqElem( faces, i );//函数 cvGetSeqElem 查找序列中索引所指定的元素，并返回指向该元素的指针
            CvPoint center;
            int radius;
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);
            if((radius = cvRound((r->width + r->height)*0.25*scale))>0)
            cvCircle( img, center, radius, colors[i%8], 3, 8, 0 );
        }
    printf("detect end1!\n");system("pause");
    cvShowImage( "result", img );
    cvWaitKey(0);
    printf("detect end!\n");system("pause");
    //cvReleaseImage( &small_img );
    

}



void Value_count()
{
    int w,h;
    scanf("%d%d",&w,&h);
    int count=0;
    int x,y;
    double s;
    const int k=3;
    int smaller = (w<h)?w:h;    //图像的长宽较小者
    for( s = 1 ; s * (float)Sample_W < (float)smaller ; s *= 1.2 )
    {

        for( x = 0 ; x <= w - s * (float)Sample_W ; x += int(k * s))        //把检测窗口的左上顶点放在待检图像的(x,y)坐标上

            for( y = 0 ; y <= h - s * (float)Sample_H ; y += int(k * s))
            {
                
                count++;
            }
    }
    printf("count=%d\n",count);
}


///////////////////级联分类器///////////////////

// Ft 为级联分类器的误检率（虚警率），f 为每层强分类器的虚警率，d 为每层强分类器的检测率
void Cascade( double Ft , double f , double d )
{
    double Fi = 1.0 , Di = 1.0 ;
    int i = 0 ;
}




/*
void Cacu_Mg( Ex_IplImage * img)
{
    IplImage * imgc=cvCloneImage(img->image);
    int i,j;
    int step=imgc->widthStep/sizeof(uchar);
    uchar * data=(uchar*)imgc->imageData;
    CvMat * p=cvCreateMat(1500,1500,CV_32FC1);
    img->Mg=cvCreateMat(1500,1500,CV_32FC1);
    cvSetZero(p);
    cvSetZero(img->Mg);
    float f;
    printf("imagew=%d,imageh=%d,channel=%d,origin=%d\n",imgc->width,imgc->height,imgc->nChannels,imgc->origin);
    for(i=0;i<imgc->height;i++)             //p[i][-1]=0
    {
        f=data[i*step];
        cvmSet(p,i,0,f);
    }//
    printf("=========================================\n");
    for(i=0;i<imgc->height;i++)             //p[x][y]=p[x][y-1]+f[x][y],每一行累加和
        for(j=1;j<imgc->width;j++)
        {
            f=data[i*step+j];
            cvmSet(p,i,j,cvmGet(p,i,j-1)+f);
        }
    printf("=========================================\n");
    for(j=0;j<imgc->width;j++)//g[-1][y]=0;
        cvmSet(img->Mg,0,j,cvmGet(p,0,j));
    printf("=========================================\n");
    for(i=1;i<imgc->height;i++)             //g[x][y]=g[x-1][y]+p[x][y]
        for(j=0;j<imgc->width;j++)
        {
            cvmSet(img->Mg,i,j,cvmGet(img->Mg,i-1,j)+cvmGet(p,i,j));
        }
    
    cvReleaseMat(&p);
    printf("Mg[i-1][j-1]=%f\n",cvmGet(img->Mg,imgc->height-1,imgc->width-1));
        printf("=========================================\n");
        cvReleaseImage( &imgc );
}

*/

void ReName( )
{

    FILE * fp;

    if((fp=fopen(NON_FACE_TXT_1, "r"))==NULL)
    {
        printf("can not open the face.txt!\n");
        exit(0);
    }
    while(!feof(fp))
    {
        char s[30]= NON_FACE_DIR_PAH_1;
        char pic_name[15];
        
        //if(fgets(pic_name,13,fp)!=NULL)
        memset(pic_name, 0, sizeof(char) * 33);
        if (fscanf(fp, "%s", pic_name) != EOF)
        {
            //printf("%s\n",pic_name);
            strcat(s,pic_name);
            IplImage * img = cvLoadImage(s);
            //printf("***********\n");
            char frt[20] = "a";
            strcat(frt,pic_name);
            //printf("%s\n",frt);
            cvSaveImage(frt,img);
            cvReleaseImage(&img);
        }
        fgetc(fp);
    }

    
}
/*
int main()
{
    ReName();
}
*/
/*
int main( int argc, char** argv )
{
    int i = 1;
    FILE * fp;
    char pic_name[15];
    if((fp=fopen("E:\\test1\\face\\face.txt","r"))==NULL)
    {
        printf("can not open the face.txt!\n");
        exit(0);
    }
    while(!feof(fp))
    {
        char s[30]="E:\\test1\\face\\";
        if(fgets(pic_name,14,fp)!=NULL)
        {
            printf("%s\n",pic_name);
            strcat(s,pic_name);
            IplImage* img = cvLoadImage(s);
            if(!img) printf("Could not load image file:%s\n",pic_name);

            cvNamedWindow(pic_name, CV_WINDOW_AUTOSIZE );

            cvShowImage(pic_name, img );

            printf("image size is %d * %d\n",img->width,img->height);

            cvWaitKey(0);
            
            cvReleaseImage( &img );
    
            cvDestroyWindow(pic_name);

        }
        fgetc(fp);
    }
    fclose(fp);

    return 0;
}*/




int main()
{
    double t=(double)cvGetTickCount();
    //printf("kjfksjflksjfjs\n");
    //system("pause");

    Stage_Classifier_Cascade( 0.0000001, 0.4 , 0.996 , 2427 , 1500 ) ;
    //Single_Classifier( );

    t = (double)cvGetTickCount() - t ;

    printf("train time is %gs\n",t/((double)cvGetTickFrequency()*1000*1000));
}

/*
int main( int argc, char** argv )
{
    IplImage* img = cvLoadImage( argv[1] );
    if(!img) printf("Could not load image file:%s\n",argv[1]);

    cvNamedWindow("source",CV_WINDOW_AUTOSIZE);

    cvShowImage("source",img);

    cvWaitKey(0);

    IplImage* gimg=cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

    //IplImage* himg=cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

    cvCvtColor(img,gimg,CV_BGR2GRAY);

    cvEqualizeHist( gimg, gimg );

    printf("**********************************\n");

    printf("image size is %d * %d , image channel is %d\n",gimg->width,gimg->height,gimg->nChannels);

    Detect( gimg );

    //Cacu_Image_Mg( himg );

    printf("===================================\n");

    cvNamedWindow("detected",CV_WINDOW_AUTOSIZE);

    cvShowImage("detected",gimg);

    cvWaitKey(0);

    cvReleaseImage( &img );
    cvReleaseImage( &gimg );
    //cvReleaseImage( &himg );
    cvDestroyWindow("source");
    cvDestroyWindow("detected");
    return 0;
}
*/
/*
int main()
{
    Read_Samples();
    return 0;
}
*/

/*
int main( int argc, char** argv )
{
    double t=(double)cvGetTickCount();
    int i = 0;
    FILE * fp;
    char pic_name[15];
    int m,l;            //m,l分别为正负样本个数

    printf("input the number of positive images and negtive images:");

    scanf("%d%d",&m,&l);

    num_samples = m + l;

    if((fp=fopen("E:\\test1\\face\\face.txt","r"))==NULL)
    {
        printf("can not open the face.txt!\n");
        exit(0);
    }

    float w = 1/(2*(float)m);
    printf("w=%f\n",w);system("pause");
    
    while(!feof(fp))
    {
        char s[30]="E:\\test1\\face\\";
        
        if(fgets(pic_name,14,fp)!=NULL)
        {
            printf("%s\n",pic_name);
            strcat(s,pic_name);
            Samples[i].image = cvLoadImage(s);
            Samples[i].Y = 1;
            Samples[i].N = i;
            Samples[i].W = w;
            Samples[i].IsFalseDetected = 0;
            if(!Samples[i].image) printf("Could not load image file:%s\n",pic_name);

            cvNamedWindow(pic_name, CV_WINDOW_AUTOSIZE );

            cvShowImage(pic_name, Samples[i].image );

            printf("image size is %d * %d\n",Samples[i].image->width,Samples[i].image->height);

            i++;

            cvWaitKey(0);
            
            cvReleaseImage( &Samples[i].image );
    
            cvDestroyWindow(pic_name);

        }
        fgetc(fp);
    }
    fclose(fp);

    /////////////////////////////////////////////////////////
    if((fp=fopen("E:\\test1\\non_face\\non_face.txt","r"))==NULL)
    {
        printf("can not open the non_face.txt!\n");
        exit(0);
    }

    w=1/(2*(float)l);
    printf("w=%f\n",w);system("pause");
    while(!feof(fp))
    {
        char s[35]="E:\\test1\\non_face\\";
        if(fgets(pic_name,14,fp)!=NULL)
        {
            printf("%s\n",pic_name);
            strcat(s,pic_name);
            Samples[i].image = cvLoadImage(s);
            Samples[i].Y = 0;
            Samples[i].N = i;
            Samples[i].W = w;
            Samples[i].IsFalseDetected = 0;
            if(!Samples[i].image) printf("Could not load image file:%s\n",pic_name);

            //cvNamedWindow(pic_name, CV_WINDOW_AUTOSIZE );

            //cvShowImage(pic_name, img );

            printf("image size is %d * %d\n",Samples[i].image->width,Samples[i].image->height);

            i++;

            //cvWaitKey(0);
            
            //cvReleaseImage( &img );
    
            //cvDestroyWindow(pic_name);

        }
        fgetc(fp);
    }
    fclose(fp);
    
    printf("the sum of samples is %d\n",i);

    Cacu_Samples_Mg();

    Produce_AllFeatures();

    //Cal_Allsamples_Featurej(60000);

    //Cal_Allsamples_Allfeatures();

    Stage_Classifier( );

    t=(double)cvGetTickCount()-t;
    printf("the time is %gs\n",t/((double)cvGetTickFrequency()*1000*1000));

    return 0;
}
*/

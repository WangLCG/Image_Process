/*********************************************************************
 * klt.h
 *
 * Kanade-Lucas-Tomasi tracker
 *********************************************************************/

#ifndef _KLT_H_
#define _KLT_H_

typedef float KLT_locType;
typedef unsigned char KLT_PixelType;

#define KLT_BOOL int

#ifndef TRUE
#define TRUE  1
#define FALSE 0
#endif

#ifndef NULL
#define NULL  0
#endif

#define KLT_TRACKED           (0)   // 跟踪成功   
#define KLT_NOT_FOUND        (-1)
#define KLT_SMALL_DET        (-2)   //feature is lost
#define KLT_MAX_ITERATIONS   (-3)   //tracking stopped because iterations timed out
#define KLT_OOB              (-4)  //跟踪出错   
#define KLT_LARGE_RESIDUE    (-5)  //残差过大  

/*******************
 * Structures
 */

typedef struct  {
  /* Available to user */
  int mindist;            /* min distance between selected features */
  int window_width, window_height; ////*特征窗口尺寸*/
  KLT_BOOL sequentialMode;    /* whether to save most recent image to save time */
  /* can set to TRUE manually, but don't set to */
  /* FALSE manually */
  KLT_BOOL smoothBeforeSelecting;   ////* whether to smooth image before 特征点筛选前平滑去噪*/
  /* selecting features */
  KLT_BOOL writeInternalImages;    ////* whether to write internal images 写积分函数*/
  
  /* Available, but hopefully can ignore */
  int min_eigenvalue;        /* smallest eigenvalue allowed for selecting  */
  float min_determinant;    ////* min determinant for declaring tracking failure  跟踪失败的最小..[待定]*/
  float min_displacement;    /* th for stopping tracking when pixel changes little */
  int max_iterations;        /* th for stopping tracking when too many iterations */
  float max_residue;        /* th for stopping tracking when residue is large */
  float grad_sigma;         ///*计算梯度的高斯核因子*/
  float smooth_sigma_fact;   ////*平滑滤波高斯核因子*/
  float pyramid_sigma_fact;  ////*金字塔高斯核因子*/
  int nSkippedPixels;       /// /* # of pixels skipped when finding features  用户加速计算*/
  int borderx;           /// /* border in which features will not be found */
  int bordery;
  int nPyramidLevels;       ////* computed from search_ranges*/
  int subsampling;        ////*    金字塔采样尺寸     " */
  
  /* User must not touch these */
  void *pyramid_last;
  void *pyramid_last_gradx;
  void *pyramid_last_grady;
}  KLT_TrackingContextRec, *KLT_TrackingContext;


typedef struct  {
  KLT_locType x;
  KLT_locType y;
  int val;    
/* will be needed when we do affine */
/*     int window_width, window_height; */
/*    uchar *img; */
}  KLT_FeatureRec, *KLT_Feature;

typedef struct  {
  int nFeatures;
  KLT_Feature *feature;
}  KLT_FeatureListRec, *KLT_FeatureList;

typedef struct  {
  int nFrames;
  KLT_Feature *feature;
}  KLT_FeatureHistoryRec, *KLT_FeatureHistory;

typedef struct  {
  int nFrames;
  int nFeatures;
  KLT_Feature **feature;
}  KLT_FeatureTableRec, *KLT_FeatureTable;



/*******************
 * Functions
 */
#ifdef __cplusplus
extern "C"
{
#endif
    /* Create */
    KLT_TrackingContext KLTCreateTrackingContext(void);
    KLT_FeatureList KLTCreateFeatureList(
        int nFeatures);
    KLT_FeatureHistory KLTCreateFeatureHistory(
        int nFrames);
    KLT_FeatureTable KLTCreateFeatureTable(
        int nFrames,
        int nFeatures);

    /* Free */
    void KLTFreeTrackingContext(
        KLT_TrackingContext tc);
    void KLTFreeFeatureList(
        KLT_FeatureList fl);
    void KLTFreeFeatureHistory(
        KLT_FeatureHistory fh);
    void KLTFreeFeatureTable(
        KLT_FeatureTable ft);

    /* Processing */
    void KLTSelectGoodFeatures(
        KLT_TrackingContext tc,
        KLT_PixelType *img,
        int ncols,
        int nrows,
        KLT_FeatureList fl);
    void KLTTrackFeatures(
        KLT_TrackingContext tc,
        KLT_PixelType *img1,
        KLT_PixelType *img2,
        int ncols,
        int nrows,
        KLT_FeatureList fl);
    void KLTReplaceLostFeatures(
        KLT_TrackingContext tc,
        KLT_PixelType *img,
        int ncols,
        int nrows,
        KLT_FeatureList fl);

    /* Utilities */
    int KLTCountRemainingFeatures(
        KLT_FeatureList fl);
    void KLTPrintTrackingContext(
        KLT_TrackingContext tc);
    void KLTChangeTCPyramid(
        KLT_TrackingContext tc,
        int search_range);
    void KLTUpdateTCBorder(
        KLT_TrackingContext tc);
    void KLTStopSequentialMode(
        KLT_TrackingContext tc);
    void KLTSetVerbosity(
        int verbosity);
    float _KLTComputeSmoothSigma(
        KLT_TrackingContext tc);

    /* Storing/Extracting Features */
    void KLTStoreFeatureList(
        KLT_FeatureList fl,
        KLT_FeatureTable ft,
        int frame);
    void KLTExtractFeatureList(
        KLT_FeatureList fl,
        KLT_FeatureTable ft,
        int frame);
    void KLTStoreFeatureHistory(
        KLT_FeatureHistory fh,
        KLT_FeatureTable ft,
        int feat);
    void KLTExtractFeatureHistory(
        KLT_FeatureHistory fh,
        KLT_FeatureTable ft,
        int feat);

    /* Writing/Reading */
    void KLTWriteFeatureListToPPM(
        KLT_FeatureList fl,
        KLT_PixelType *greyimg,
        int ncols,
        int nrows,
        char *filename);
    void KLTWriteFeatureList(
        KLT_FeatureList fl,
        char *filename,
        char *fmt);
    void KLTWriteFeatureHistory(
        KLT_FeatureHistory fh,
        char *filename,
        char *fmt);
    void KLTWriteFeatureTable(
        KLT_FeatureTable ft,
        char *filename,
        char *fmt);
    KLT_FeatureList KLTReadFeatureList(
        KLT_FeatureList fl,
        char *filename);
    KLT_FeatureHistory KLTReadFeatureHistory(
        KLT_FeatureHistory fh,
        char *filename);
    KLT_FeatureTable KLTReadFeatureTable(
        KLT_FeatureTable ft,
        char *filename);

#ifdef __cplusplus
}
#endif

#endif







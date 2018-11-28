#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//for opencv
#include "cvhead.h"

#include "Feature.h"
#include "Harris/harris.h"

using namespace cv;
using namespace std;


int main()
{
    //LBP();
    //test_cal_gradian();
    //harris_main();
    harris_main_CStyle();
    waitKey(0);
    getchar();
    return 0;
}
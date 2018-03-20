#include<stdio.h>
#include <pthread.h>
#include <math.h>

void* pthread_test(void* t)
{
    int * k = (int*)t;
    printf(" k =%d\n",*k);
    return NULL;
}

//5X5非极大值抑制使用的数组 
int ofs[25] = 
{   -2*176-2,    -2*176-1,  -2*176,     -2*176+1,   -2*176+2,
    -176-2,      -176-1,    -176,       -176+1,     -176+2,
    -2,          -1,        0,          1,          2,
    176-2 ,      176-1 ,    176,        176+1,      176+2,
   2*176-2,      2*176-1,   2*176,      2*176+1,    2*176+2
};

//5X5非极大值抑制 
/*
    img:输入图片，格式为Y格式，大小为176*144
    w=176  h=144
    out:极大值抑制的输出，为二值图像 
*/
void NO_MAX_EXPRESS_5X5(void* img, int w, int h,void* out)
{
        
        int index = 0, i = 0, j = 0,k = 0;
        short max_val = 0;
        int TotalNum = 0;
        
        unsigned char* imgptr = (unsigned char*)img + ( w + 1) * 2;  //5X5模板在图片边界处的宽度 
        unsigned char* imgout = (unsigned char*)out;
        
        for (j = 0; j < w - 2; j ++)
        {
            for (i = 0; i < h - 2; i ++)
            {
                int max_ofs = 0;
                int offset = i * w + j;
                int num = 0;
                max_val = imgptr[offset];

                for (k = 0; k < 25; k++)
                {
                    if (max_val >= imgptr[offset + ofs[k]])
                    {
                        //out[offset] = 255;
                        num++;
                    }
                }
                
                if(num == 25)
                {
                    if(imgptr[offset] > 80)    //阈值 
                       imgout[offset] = 255; 
                }
            }

        }

}

#define WIDTH     176
#define HIGHT     144
#define FILENAME  "2picture_out"

int main()
{
    FILE *fd = fopen(FILENAME,"rb+");
    unsigned char img[WIDTH*HIGHT];
    unsigned char out[WIDTH*HIGHT];
    
    memset(img, 0, WIDTH*HIGHT);
    
   if(!fd)
    {
        printf("Open error\n");
        return 0;
    }
    
    FILE *wfd  = fopen("img8x5_out","wb+");
    while( fread(img, 1, WIDTH*HIGHT, fd))
    {   
        memset(out, 0, WIDTH*HIGHT);
        NO_MAX_EXPRESS_5X5(img, WIDTH,HIGHT, out);
        fwrite(out, 1, WIDTH*HIGHT, wfd);
    }
    
    fclose(fd);
    fclose(wfd);
    
    fd  = NULL;
    wfd = NULL;
    
    return 0;
}

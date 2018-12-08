#include "draw_img.h"

//画点
void Customer_drawPoint(unsigned char * dst, int width, int height, int x, int y, int color)
{
    if (x <= 0) x = 1;
    if (x >= width - 1) x = width - 1;
    if (y <= 0) y = 1;
    if (y >= height - 1) y = height - 1;
    dst[y * width + x] = 255;
}

//划线
void Customer_drawLine(unsigned char *dst, int width, int height, int x0, int y0, int x1, int y1, int color)
{
    int x_min = 0, x_max = 0;
    int y_min = 0, y_max = 0;
    int x_step = 0, y_step = 0;
    int x = 0, y = 0;
    int y_t = 0;


    x_min = x0 < x1 ? x0 : x1;
    x_max = x0 < x1 ? x1 : x0;
    y_min = y0 < y1 ? y0 : y1;
    y_max = y0 < y1 ? y1 : y0;
    x_step = x_max - x_min;
    y_step = y_max - y_min;

    //绘制直线   
    if (x_step == 0 || y_step == 0)
    {
        for (x = x_min; x <= x_max; x++)
        {
            for (y = y_min; y <= y_max; y++)
            {
                Customer_drawPoint(dst, width, height, x, y, color);
            }
        }
    }
    else//绘制非直线  
    {
        for (x = x_min; x <= x_max; x++)
        {
            //利用x来偏移y   
            y_t = (int)(((float)y_step / x_step) * (x - x_min)) + y_min;
            Customer_drawPoint(dst, width, height, x, y_t, color);
        }
    }
}


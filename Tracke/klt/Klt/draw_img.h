/*********************************************************************
* draw_img.h
*
* draw point or line on image
*********************************************************************/

#ifndef _DRAW_IMG_
#define _DRAW_IMG_

typedef struct XY_Point
{
    int x;
    int y;
}XY_Point;

#ifdef __cplusplus
extern "C"
{
#endif

    //»­µã
    void Customer_drawPoint(unsigned char * dst, int width, int height, int x, int y, int color);

    //»®Ïß
    void Customer_drawLine(unsigned char *dst, int width, int height, int x0, int y0, int x1, int y1, int color);

#ifdef __cplusplus
}
#endif
#endif
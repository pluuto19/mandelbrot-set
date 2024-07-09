#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMG_W 1920
#define IMG_H 1080
#define CHANNELS 3
#define MAX_ITR 100
#define REAL_MIN -2.0
#define REAL_MAX 1.0
#define IMAG_MIN -0.85
#define IMAG_MAX 0.8375
#define INC_REAL (REAL_MAX - REAL_MIN) / IMG_W
#define INC_IMAG (IMAG_MAX - IMAG_MIN) / IMG_H
#define RATIO IMG_W / IMG_H
typedef struct complexNumber
{
    float real;
    float imag;
} C;

float complexAbs(C *c)
{
    return sqrt((c->real * c->real) + (c->imag * c->imag));
}

void complexAdd(C *z, C *cnst, C *res)
{
    res->real = z->real + cnst->real;
    res->imag = z->imag + cnst->imag;
}

void complexMult(C *x, C *y, C *res)
{
    res->real = (x->real * y->real) - (x->imag * y->imag);
    res->imag = (x->real * y->imag) + (x->imag * y->real);
}

void getColor(int itrs, unsigned char *r, unsigned char *g, unsigned char *b)
{
    *r = (unsigned char)(itrs * 2.0f);
    *g = (unsigned char)(itrs * 1.9f);
    *b = (unsigned char)(itrs * 2.35f);
}

int mandelbrot(C *cnst)
{
    C z = {0.0, 0.0};
    C zSq;
    for (int i = 0; i < MAX_ITR; i++)
    {
        if (complexAbs(&z) > 2)
        {
            return i;
        }
        complexMult(&z, &z, &zSq);
        complexAdd(&zSq, cnst, &z);
    }
    return MAX_ITR;
}

int main(void)
{
    unsigned char *image = (unsigned char *)malloc(IMG_W * IMG_H * CHANNELS);

    if (image == NULL)
    {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    float real = REAL_MIN;
    for (int x = 0; x < IMG_W; x++)
    {
        x *= RATIO;
        float imag = IMAG_MIN;
        for (int y = 0; y < IMG_H; y++)
        {
            C planarComplexNum = {real, imag};
            int itrs = mandelbrot(&planarComplexNum);
            int pixel_index = (y * IMG_W + x) * CHANNELS;
            unsigned char r, g, b;
            getColor(itrs, &r, &g, &b);
            image[pixel_index + 0] = r;
            image[pixel_index + 1] = g;
            image[pixel_index + 2] = b;

            imag += INC_IMAG;
        }
        real += INC_REAL;
    }

    if (!stbi_write_png("mandel-c.png", IMG_W, IMG_H, CHANNELS, image, IMG_W * CHANNELS))
    {
        fprintf(stderr, "Failed to write image\n");
        free(image);
        return 1;
    }

    free(image);
    printf("Image written to mandel-c.png\n");
    return 0;
}

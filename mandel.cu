#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMG_W 1920
#define IMG_H 1080
#define CHANNELS 3
#define MAX_ITR 100
#define NUM_THREADS 32 * 32 // GTX 1650 supports threads launching in 2 dimnesions, each with 32 threads
#define NUM_BLOCKS (int)ceil(((long double)IMG_W * IMG_H) / NUM_THREADS)

typedef struct complexNumber
{
    long double real;
    long double imag;
} C;

__device__ long double complexAbs(C *c)
{
    return sqrtf((c->real * c->real) + (c->imag * c->imag));
}

__device__ void complexAdd(C *z, C *cnst, C *res)
{
    res->real = z->real + cnst->real;
    res->imag = z->imag + cnst->imag;
}

__device__ void complexMult(C *x, C *y, C *res)
{
    res->real = (x->real * y->real) - (x->imag * y->imag);
    res->imag = (x->real * y->imag) + (x->imag * y->real);
}

__device__ int mandelbrot(C *cnst)
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

__device__ void getColor(int itrs, unsigned char *r, unsigned char *g, unsigned char *b)
{
    *r = (unsigned char)(itrs * 2.0f);
    *g = (unsigned char)(itrs * 1.9f);
    *b = (unsigned char)(itrs * 2.35f);
}

__global__ void parallelMandelbrot(unsigned char *dev_image, long double REAL_MIN, long double IMAG_MIN, long double INC_REAL, long double INC_IMAG){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < IMG_W && y < IMG_H && x*y < IMG_W*IMG_H){
        long double real = REAL_MIN + ( (long double)x * INC_REAL);
        long double imag = IMAG_MIN + ( (long double)y * INC_IMAG);
        C planarComplexNum = {real, imag};
        int itrs = mandelbrot(&planarComplexNum);
        int pixel_index = (y * IMG_W + x) * CHANNELS;
        unsigned char r, g, b;
        getColor(itrs, &r, &g, &b);
        dev_image[pixel_index + 0] = r;
        dev_image[pixel_index + 1] = g;
        dev_image[pixel_index + 2] = b;
    }
}

int main(void){
    long double REAL_MIN = -2.0;
    long double REAL_MAX = 1.0;
    long double IMAG_MIN = -0.85;
    long double IMAG_MAX = 0.8375;

    long double INC_REAL = (REAL_MAX - REAL_MIN) / IMG_W;
    long double INC_IMAG = (IMAG_MAX - IMAG_MIN) / IMG_H;

    unsigned char *host_image = (unsigned char *)malloc(IMG_W * IMG_H * CHANNELS);
    if (host_image == NULL)
    {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }
    unsigned char *dev_image;
    cudaMalloc(&dev_image, IMG_W * IMG_H * CHANNELS);
    dim3 block_dim(32,32,1);
    dim3 grid_dim(64,32,1);
    parallelMandelbrot<<<grid_dim, block_dim>>>(dev_image, REAL_MIN, IMAG_MIN, INC_REAL, INC_IMAG);
    cudaDeviceSynchronize();
    cudaMemcpy(host_image, dev_image, IMG_W * IMG_H * CHANNELS, cudaMemcpyDeviceToHost);
    if (!stbi_write_png("mandel-cuda.png", IMG_W, IMG_H, CHANNELS, host_image, IMG_W * CHANNELS))
    {
        fprintf(stderr, "Failed to write image\n");
        return 1;
    }
    free(host_image);
    cudaFree(dev_image);
    printf("Image(s) written!\n");
}

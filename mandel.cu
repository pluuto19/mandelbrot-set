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
#define IMAG_MIN -1.5
#define IMAG_MAX 1.5
#define INC_REAL (REAL_MAX - REAL_MIN) / IMG_W
#define INC_IMAG (IMAG_MAX - IMAG_MIN) / IMG_H
#define NUM_THREADS 32 * 32 // GTX 1650 supports threads launching in 2 dimnesions, each with 32 threads
#define NUM_BLOCKS (int)ceil(((float)IMG_W * IMG_H) / NUM_THREADS)

typedef struct complexNumber
{
    float real;
    float imag;
} C;

__device__ float complexAbs(C *c)
{
    return sqrt((c->real * c->real) + (c->imag * c->imag));
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

__global__ void parallelMandelbrot(unsigned char *dev_image){
    int x = threadIdx.x + (blockIdx.x % (IMG_W / blockDim.x)) * blockDim.x;
    int y = threadIdx.y + (blockIdx.x / (IMG_W / blockDim.x)) * blockDim.y;
    if (x < IMG_W && y < IMG_H && x*y < IMG_W*IMG_H){
        float real = REAL_MIN + ( (float)x * INC_REAL);
        float imag = IMAG_MIN + ( (float)y * INC_IMAG);
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
    // each kernel will be launched with a zoom factor as function parameter
    // flow:
    // loop with scaling factor
    // launch kernel with scaling factor
    // inside kernel:
    // adjust the ranges with scaling factor
    // repeat loop
    unsigned char *host_image = (unsigned char *)malloc(IMG_W * IMG_H * CHANNELS); //allocate memory on host before launching kernel to save computations in case memory fails to allocate

    if (host_image == NULL)
    {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    unsigned char *dev_image;

    cudaMalloc(&dev_image, IMG_W * IMG_H * CHANNELS);

    // handle allocation error

    dim3 block_dim(32,32,1);

    parallelMandelbrot<<<NUM_BLOCKS, block_dim>>>(dev_image);

    cudaDeviceSynchronize();

    cudaMemcpy(host_image, dev_image, IMG_W * IMG_H * CHANNELS, cudaMemcpyDeviceToHost);

    cudaFree(dev_image);

    if (!stbi_write_png("mandel-cuda.png", IMG_W, IMG_H, CHANNELS, host_image, IMG_W * CHANNELS))
    {
        fprintf(stderr, "Failed to write image\n");
        return 1;
    }

    free(host_image);
    printf("Image written to mandel-c.png\n");
}

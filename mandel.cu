#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define img_w 1920
#define img_h 1080
#define channels 3

int main(void){
    // each frame contains the pixels mapped to the region's range
    // scaling factor is to be multiplied by the region's range
    // change the region variables by scaling amount (and point of zoom) after each iteration

    // generating one frame
    // malloc w x h matrix on host
    // malloc w x h matric on device
    // call kernel with scaling factor, device matrix
    // copy matrix from kernel onto host
    // pass host matrix to stb_image_write
    // cudaDeviceSynchronize();
}
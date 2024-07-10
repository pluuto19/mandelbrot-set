#include<stdio.h>
#include<math.h>

__global__ void myprint(void){
    printf("Block ID x: %d, Block ID y: %d, Block ID z: %d, Thread ID x: %d, Thread ID y: %d, Thread ID z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

void main(void){
    printf("here");
    dim3 block_dim(32, 32, 1);
    myprint<<<(int)ceil( ((float)1920 * 1080) / (32 * 32)), block_dim>>>();
}
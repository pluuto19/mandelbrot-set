#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define img_w 1920
#define img_h 1080
#define channels 3

int main(void){
    unsigned char *image = (unsigned char *)malloc(img_w * img_h * channels);

    if (image == NULL) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    for (int y = 0; y < img_h; y++) {
        for (int x = 0; x < img_w; x++) {
            int index = (y * img_w + x) * channels;
            unsigned char color = (unsigned char)(255 * ((float)x / (img_w - 1)));
            image[index + 0] = color;
            image[index + 1] = color;
            image[index + 2] = color;
        }
    }

    if (!stbi_write_png("gradient.png", img_w, img_h, channels, image, img_w * channels)) {
        fprintf(stderr, "Failed to write image\n");
        free(image);
        return 1;
    }

    free(image);
    printf("Image written to gradient.png\n");
    return 0;
}
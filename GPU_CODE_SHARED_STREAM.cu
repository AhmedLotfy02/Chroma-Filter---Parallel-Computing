%%writefile kernel11.cu
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

__global__ void replaceChromaBackground(uchar3* src, uchar3* bg, uchar3* dst, int width, int height, uchar3 chromaKey, int threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    __shared__ uchar3 sharedSrc[16][16];
    __shared__ uchar3 sharedBg[16][16];
    
    if (x < width && y < height) {
        int idx = y * width + x;
        sharedSrc[threadIdx.y][threadIdx.x] = src[idx];
        sharedBg[threadIdx.y][threadIdx.x] = bg[idx];
        __syncthreads();
        
        uchar3 pixel = sharedSrc[threadIdx.y][threadIdx.x];
        int diff = abs(pixel.x - chromaKey.x) + abs(pixel.y - chromaKey.y) + abs(pixel.z - chromaKey.z);
        if (diff < threshold) {
            dst[idx] = sharedBg[threadIdx.y][threadIdx.x];
        } else {
            dst[idx] = pixel;
        }
    }
}

uchar3* loadImage(const std::string& filename, int& width, int& height) {
    int channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 3);
    if (!data) {
        std::cerr << "Error: Unable to load image " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    uchar3* image = new uchar3[width * height];
    for (int i = 0; i < width * height; ++i) {
        image[i].x = data[3 * i];
        image[i].y = data[3 * i + 1];
        image[i].z = data[3 * i + 2];
    }
    stbi_image_free(data);
    return image;
}

void saveImage(const std::string& filename, uchar3* image, int width, int height) {
    unsigned char* data = new unsigned char[width * height * 3];
    for (int i = 0; i < width * height; ++i) {
        data[3 * i] = image[i].x;
        data[3 * i + 1] = image[i].y;
        data[3 * i + 2] = image[i].z;
    }
    stbi_write_png(filename.c_str(), width, height, 3, data, width * 3);
    delete[] data;
}

int main() {
    int srcWidth, srcHeight, bgWidth, bgHeight;
    uchar3* srcImage = loadImage("image-700x900chr1.png", srcWidth, srcHeight);
    uchar3* bgImage = loadImage("image-700x900bg.jpg", bg

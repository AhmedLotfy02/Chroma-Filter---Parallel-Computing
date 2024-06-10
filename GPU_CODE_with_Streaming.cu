
// %%writefile kernel9.cu
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
    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 pixel = src[idx];
        int diff = abs(pixel.x - chromaKey.x) + abs(pixel.y - chromaKey.y) + abs(pixel.z - chromaKey.z);
        if (diff < threshold) {
            dst[idx] = bg[idx];
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
    uchar3* bgImage = loadImage("image-700x900bg.jpg", bgWidth, bgHeight);

    if (srcWidth != bgWidth || srcHeight != bgHeight) {
        std::cerr << "Error: Source and background images must be the same size." << std::endl;
        delete[] srcImage;
        delete[] bgImage;
        return EXIT_FAILURE;
    }

    uchar3* dstImage = new uchar3[srcWidth * srcHeight];
    uchar3* dev_src, * dev_bg, * dev_dst;
    size_t imageSize = srcWidth * srcHeight * sizeof(uchar3);

    cudaMalloc(&dev_src, imageSize);
    cudaMalloc(&dev_bg, imageSize);
    cudaMalloc(&dev_dst, imageSize);

    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    cudaMemcpyAsync(dev_src, srcImage, imageSize, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(dev_bg, bgImage, imageSize, cudaMemcpyHostToDevice, stream2);

    uchar3 chromaKey = {0, 255, 0};  // Green color
    int threshold = 100;

    dim3 block(16, 16);
    dim3 grid((srcWidth + block.x - 1) / block.x, (srcHeight + block.y - 1) / block.y);

    auto start = std::chrono::high_resolution_clock::now();
    replaceChromaBackground<<<grid, block, 0, stream1>>>(dev_src, dev_bg, dev_dst, srcWidth, srcHeight, chromaKey, threshold);
    cudaMemcpyAsync(dstImage, dev_dst, imageSize, cudaMemcpyDeviceToHost, stream3);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "GPU Time with Streams: " << elapsed.count() << " seconds" << std::endl;

    saveImage("result_gpu_streams.png", dstImage, srcWidth, srcHeight);

    cudaFree(dev_src);
    cudaFree(dev_bg);
    cudaFree(dev_dst);
    delete[] srcImage;
    delete[] bgImage;
    delete[] dstImage;

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    return 0;
}

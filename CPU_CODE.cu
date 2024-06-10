// %%writefile kernel8.cu
#include <iostream>
#include <chrono>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"



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

void replaceChromaBackgroundCPU(uchar3* src, uchar3* bg, uchar3* dst, int width, int height, uchar3 chromaKey, int threshold) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
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
}

int main() {
    int srcWidth, srcHeight, bgWidth, bgHeight;
    uchar3* srcImage = loadImage("1920x1080chr.png", srcWidth, srcHeight);
    uchar3* bgImage = loadImage("1920x1080bg.jpg", bgWidth, bgHeight);

    if (srcWidth != bgWidth || srcHeight != bgHeight) {
        std::cerr << "Error: Source and background images must be the same size." << std::endl;
        delete[] srcImage;
        delete[] bgImage;
        return EXIT_FAILURE;
    }

    uchar3* dstImage = new uchar3[srcWidth * srcHeight];
    uchar3 chromaKey = {0, 255, 0};
    int threshold = 100;

    auto start = std::chrono::high_resolution_clock::now();
    replaceChromaBackgroundCPU(srcImage, bgImage, dstImage, srcWidth, srcHeight, chromaKey, threshold);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "CPU Time: " << elapsed.count() << " seconds" << std::endl;

    saveImage("result_cpu_480x240.png", dstImage, srcWidth, srcHeight);

    delete[] srcImage;
    delete[] bgImage;
    delete[] dstImage;

    return 0;
}

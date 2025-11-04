## NAME : NAVEEN JAISANKER
## REG. NO. : 212224110039

# PCA-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD
Mini Project - Face Detection or Convert an image into gray scale image using CUDA GPU programming

## PROCEDURE

Enable GPU in Google Colab settings.


Install CUDA and OpenCV dependencies.


Create a new CUDA source file (grayscale.cu).


Write your CUDA and OpenCV code inside it.


Ensure the OpenCV image is stored in continuous memory.


Add CUDA error checking after kernel execution.


Compile the program using nvcc with OpenCV libraries.


Upload your color image to Colab.


Run the compiled executable with input and output filenames.

## PROGRAM

```
%%writefile grayscale.cu
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define CHANNELS 3

using namespace std;
using namespace cv;

// CUDA Kernel: Convert color image to grayscale
__global__
void colorConvertToGrey(unsigned char *rgb, unsigned char *grey, int rows, int cols)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < cols && row < rows)
    {
        int grey_offset = row * cols + col;
        int rgb_offset = grey_offset * CHANNELS;

        unsigned char b = rgb[rgb_offset + 0];
        unsigned char g = rgb[rgb_offset + 1];
        unsigned char r = rgb[rgb_offset + 2];

        grey[grey_offset] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

size_t loadImageFile(unsigned char **h_rgb_image, const std::string &input_file, int *rows, int *cols)
{
    cv::Mat img_data = cv::imread(input_file.c_str(), cv::IMREAD_COLOR);
    if (img_data.empty())
    {
        std::cerr << "Unable to load image file: " << input_file << std::endl;
        exit(1);
    }

    // Ensure the image data is continuous in memory
    if (!img_data.isContinuous()) {
        img_data = img_data.clone();
    }

    *rows = img_data.rows;
    *cols = img_data.cols;

    size_t num_pixels = (*rows) * (*cols);

    *h_rgb_image = (unsigned char *)malloc(num_pixels * CHANNELS);
    memcpy(*h_rgb_image, img_data.ptr<unsigned char>(0), num_pixels * CHANNELS);

    return num_pixels;
}


// Save grayscale image
void outputImage(const std::string &output_file, unsigned char *grey_image, int rows, int cols)
{
    cv::Mat greyData(rows, cols, CV_8UC1, grey_image);
    cv::imwrite(output_file.c_str(), greyData);
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: <executable> input_file output_file" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];

    unsigned char *h_rgb_image, *h_grey_image;
    unsigned char *d_rgb_image, *d_grey_image;
    int rows, cols;

    const size_t total_pixels = loadImageFile(&h_rgb_image, input_file, &rows, &cols);
    h_grey_image = (unsigned char *)malloc(sizeof(unsigned char) * total_pixels);

    cudaMalloc(&d_rgb_image, sizeof(unsigned char) * total_pixels * CHANNELS);
    cudaMalloc(&d_grey_image, sizeof(unsigned char) * total_pixels);
    cudaMemcpy(d_rgb_image, h_rgb_image, sizeof(unsigned char) * total_pixels * CHANNELS, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    colorConvertToGrey<<<grid, block>>>(d_rgb_image, d_grey_image, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_grey_image, d_grey_image, sizeof(unsigned char) * total_pixels, cudaMemcpyDeviceToHost);
    outputImage(output_file, h_grey_image, rows, cols);

    cudaFree(d_rgb_image);
    cudaFree(d_grey_image);
    free(h_rgb_image);
    free(h_grey_image);

    std::cout << "✅ Grayscale image saved as: " << output_file << std::endl;
    return 0;
}
```

```
!nvcc grayscale.cu -o grayscale -I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
```

```
from google.colab import files
uploaded = files.upload()
input_file = list(uploaded.keys())[0]
print("Uploaded:", input_file)
```

```
import cv2
import matplotlib.pyplot as plt
import numpy as np

original = cv2.imread("cow.jpg")
gray = cv2.imread("output.jpg", cv2.IMREAD_GRAYSCALE)

if original is None or gray is None:
    raise FileNotFoundError("Could not find input or output image.")

original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(original_rgb)
plt.title("Original Colour Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image (CUDA)")
plt.axis('off')

plt.tight_layout()
plt.show()
```


## OUTPUT:

<img width="944" height="333" alt="Screenshot 2025-11-04 142233" src="https://github.com/user-attachments/assets/f1b2651c-97f8-414d-9aea-45b29dfe8a0b" />

## RESULT:

The result is a grayscale version of the input image generated using CUDA GPU acceleration.

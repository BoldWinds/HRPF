#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>

// Kernel for merging two sorted sub-arrays of doubles
__global__ void mergeKernel(double* input, double* output, int* startA, int* startB, int* endA, int* endB, int numMerges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numMerges) {
        int a = startA[idx];
        int b = startB[idx];
        int c = a; // Output position starts at the same position as a
        
        // Perform merge until we reach the end of either array
        while (a < endA[idx] && b < endB[idx]) {
            if (input[a] <= input[b]) {
                output[c++] = input[a++];
            } else {
                output[c++] = input[b++];
            }
        }
        
        // Copy any remaining elements from the first sub-array
        while (a < endA[idx]) {
            output[c++] = input[a++];
        }
        
        // Copy any remaining elements from the second sub-array
        while (b < endB[idx]) {
            output[c++] = input[b++];
        }
    }
}

// Function to perform merge sort using CUDA for double values
void cudaMergeSort(double* data, int n) {
    double* d_data;
    double* d_temp;
    int* d_startA;
    int* d_startB;
    int* d_endA;
    int* d_endB;
    int* startA;
    int* startB;
    int* endA;
    int* endB;
    double* temp;
    
    // Allocate host memory for temporary arrays
    temp = (double*)malloc(n * sizeof(double));
    
    // Memory allocation for merge parameters
    int maxMerges = n / 2 + (n % 2); // Maximum number of merges at the lowest level
    startA = (int*)malloc(maxMerges * sizeof(int));
    startB = (int*)malloc(maxMerges * sizeof(int));
    endA = (int*)malloc(maxMerges * sizeof(int));
    endB = (int*)malloc(maxMerges * sizeof(int));
    
    // Allocate device memory
    cudaMalloc((void**)&d_data, n * sizeof(double));
    cudaMalloc((void**)&d_temp, n * sizeof(double));
    cudaMalloc((void**)&d_startA, maxMerges * sizeof(int));
    cudaMalloc((void**)&d_startB, maxMerges * sizeof(int));
    cudaMalloc((void**)&d_endA, maxMerges * sizeof(int));
    cudaMalloc((void**)&d_endB, maxMerges * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_data, data, n * sizeof(double), cudaMemcpyHostToDevice);
    
    // Start with size 1 subarrays and double the size each iteration
    for (int width = 1; width < n; width *= 2) {
        int numMerges = 0;
        
        // Set up merge parameters
        for (int i = 0; i < n; i += 2 * width) {
            startA[numMerges] = i;
            endA[numMerges] = i + width < n ? i + width : n;
            startB[numMerges] = endA[numMerges];
            endB[numMerges] = startB[numMerges] + width < n ? startB[numMerges] + width : n;
            numMerges++;
        }
        
        // Copy merge parameters to device
        cudaMemcpy(d_startA, startA, numMerges * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_startB, startB, numMerges * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_endA, endA, numMerges * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_endB, endB, numMerges * sizeof(int), cudaMemcpyHostToDevice);
        
        // Determine grid and block sizes
        int blockSize = 256;
        int gridSize = (numMerges + blockSize - 1) / blockSize;
        
        // Launch kernel
        if (width % 2 == 1) {
            mergeKernel<<<gridSize, blockSize>>>(d_data, d_temp, d_startA, d_startB, d_endA, d_endB, numMerges);
            // Next iteration will read from d_temp and write to d_data
        } else {
            mergeKernel<<<gridSize, blockSize>>>(d_temp, d_data, d_startA, d_startB, d_endA, d_endB, numMerges);
            // Next iteration will read from d_data and write to d_temp
        }
        
        // Synchronize to make sure all kernels have completed
        cudaDeviceSynchronize();
    }
    
    // Copy result back to host
    if ((int)(log2(n)) % 2 == 1) {
        cudaMemcpy(data, d_temp, n * sizeof(double), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(data, d_data, n * sizeof(double), cudaMemcpyDeviceToHost);
    }
    
    // Free memory
    free(temp);
    free(startA);
    free(startB);
    free(endA);
    free(endB);
    cudaFree(d_data);
    cudaFree(d_temp);
    cudaFree(d_startA);
    cudaFree(d_startB);
    cudaFree(d_endA);
    cudaFree(d_endB);
}

void loadData(double* datar, int length) {
    std::ifstream fin;
    fin.open("./data/datamer.txt");

    if(!fin)
    {
        std::cout<<"can not open the file data.txt"<<std::endl;
        exit(1);
    }

    for(int i = 0; i < length; ++i){
        fin >> datar[i];
    }
}

int main(int argc, char** argv) {
    std::size_t n = std::atoi(argv[1]);
    int max_run = std::atoi(argv[2]);
    double* data = (double*)malloc(n * sizeof(double));
    loadData(data, n);

    double milliseconds = 0;
    for(int run = 0; run < max_run; run++){
        struct timeval start, end;
        gettimeofday(&start, NULL);
        cudaMergeSort(data, n);
        gettimeofday(&end, NULL);
        milliseconds += (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    }
    milliseconds /= max_run;
    std::cout << milliseconds << std::endl;
    
    
    //Verify the sort (for small arrays)
    bool sorted = true;
    for (int i = 1; i < n; i++) {
        if (data[i] < data[i-1]) {
            sorted = false;
            break;
        }
    }
    
    printf("Array %s sorted.\n", sorted ? "is" : "is not");
    
    free(data);
    return 0;
}
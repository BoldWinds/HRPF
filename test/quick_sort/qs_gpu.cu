#include <iostream>
#include <fstream>
#include <sys/time.h>

#define MAX_DEPTH 16
#define INSERTION_SORT 1024

////////////////////////////////////////////////////////////////////////////////
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
////////////////////////////////////////////////////////////////////////////////
__device__ void selection_sort(double *data, int left, int right)
{
    for (int i = left; i <= right; ++i)
    {
        unsigned min_val = data[i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i + 1; j <= right; ++j)
        {
            unsigned val_j = data[j];

            if (val_j < min_val)
            {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx)
        {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Very basic quicksort algorithm, recursively launching the next level.
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_simple_quicksort(double *data, int left, int right,
                                     int depth)
{
    // If we're too deep or there are few elements left, we use an insertion
    // sort...
    if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT)
    {
        selection_sort(data, left, right);
        return;
    }

    double *lptr = data + left;
    double *rptr = data + right;
    double pivot = data[(left + right) / 2];

    // Do the partitioning.
    while (lptr <= rptr)
    {
        // Find the next left- and right-hand values to swap
        double lval = *lptr;
        double rval = *rptr;

        // Move the left pointer as long as the pointed element is smaller than the
        // pivot.
        while (lval < pivot)
        {
            lptr++;
            lval = *lptr;
        }

        // Move the right pointer as long as the pointed element is larger than the
        // pivot.
        while (rval > pivot)
        {
            rptr--;
            rval = *rptr;
        }

        // If the swap points are valid, do the swap!
        if (lptr <= rptr)
        {
            *lptr++ = rval;
            *rptr-- = lval;
        }
    }

    // Now the recursive part
    int nright = rptr - data;
    int nleft = lptr - data;

    // Launch a new block to sort the left part.
    if (left < (rptr - data))
    {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cdp_simple_quicksort<<<1, 1, 0, s>>>(data, left, nright, depth + 1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if ((lptr - data) < right)
    {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cdp_simple_quicksort<<<1, 1, 0, s1>>>(data, nleft, right, depth + 1);
        cudaStreamDestroy(s1);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Call the quicksort kernel from the host.
////////////////////////////////////////////////////////////////////////////////
void run_qsort(double *data, int nitems)
{
    // Prepare CDP for the max depth 'MAX_DEPTH'.

    // Launch on device
    int left = 0;
    int right = nitems - 1;
    std::cout << "Launching kernel on the GPU" << std::endl;
    cdp_simple_quicksort<<<1, 1>>>(data, left, right, 0);
    cudaDeviceSynchronize();
}

////////////////////////////////////////////////////////////////////////////////
// Initialize data on the host.
////////////////////////////////////////////////////////////////////////////////
void loadData(double *datar, int length)
{
    std::ifstream fin;
    fin.open("./data/datamer.txt");

    if (!fin)
    {
        std::cout << "can not open the file data.txt" << std::endl;
        exit(1);
    }

    for (int i = 0; i < length; ++i)
    {
        fin >> datar[i];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int n = std::atoi(argv[1]);

    // Create input data
    double *h_data = 0;
    double *d_data = 0;

    // Allocate CPU memory and initialize data.
    h_data = (double *)malloc(n * sizeof(double));
    loadData(h_data, n);

    // Allocate GPU memory.
    cudaMalloc((void **)&d_data, n * sizeof(double));
    cudaMemcpy(d_data, h_data, n * sizeof(double), cudaMemcpyHostToDevice);

    // Execute
    struct timeval start, end;
    gettimeofday(&start, NULL);
    run_qsort(d_data, n);
    gettimeofday(&end, NULL);
    double milliseconds = (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    std::cout << milliseconds << std::endl;

    free(h_data);
    cudaFree(d_data);

    exit(EXIT_SUCCESS);
}
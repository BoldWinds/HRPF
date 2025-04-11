#include "strassen.h"
#define THRESHOLD 256  // Increased threshold for GPU implementation

class CudaMatrix : public BasicMatrix {
public:
    CudaMatrix(int dim) : BasicMatrix(dim) {
        copyToGPU();
    }

    ~CudaMatrix() {}

    void matrixAdd(const BasicMatrix& A, const BasicMatrix& B, BasicMatrix& C) override {
        int n = A.getDim();
        
        const double* a = A.getGPUData();
        const double* b = B.getGPUData();
        double* c = C.getGPUData();
        
        // Use the provided CUDA function for matrix addition
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        sumMatrix((double*)a, (double*)b, c, n, n, n, n, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    void matrixSub(const BasicMatrix& A, const BasicMatrix& B, BasicMatrix& C) override {
        int n = A.getDim();
        
        const double* a = A.getGPUData();
        const double* b = B.getGPUData();
        double* c = C.getGPUData();
        
        // Use the provided CUDA function for matrix subtraction
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        subMatrix((double*)a, (double*)b, c, n, n, n, n, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    void matrixMul(const BasicMatrix& A, const BasicMatrix& B, BasicMatrix& C) override {
        int n = A.getDim();
        
        const double* a = A.getGPUData();
        const double* b = B.getGPUData();
        double* c = C.getGPUData();
        
        // Use the provided CUDA function for matrix multiplication
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        gemm((double*)a, (double*)b, c, n, n, n, n, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    void splitMatrix(const BasicMatrix& A, BasicMatrix& A11, BasicMatrix& A12, 
                     BasicMatrix& A21, BasicMatrix& A22) override {
        int n = A.getDim();
        int newSize = n / 2;
        
        // First copy matrices from GPU to CPU for easier splitting
        // (This could be optimized with a custom CUDA kernel for better performance)
        const double* src = A.getData();
        
        double* a11 = A11.getData();
        double* a12 = A12.getData();
        double* a21 = A21.getData();
        double* a22 = A22.getData();
        
        #pragma omp parallel for
        for (int i = 0; i < newSize; i++) {
            for (int j = 0; j < newSize; j++) {
                a11[i * newSize + j] = src[i * n + j];
                a12[i * newSize + j] = src[i * n + j + newSize];
                a21[i * newSize + j] = src[(i + newSize) * n + j];
                a22[i * newSize + j] = src[(i + newSize) * n + j + newSize];
            }
        }
        
        // Copy back to GPU
        A11.copyToGPU();
        A12.copyToGPU();
        A21.copyToGPU();
        A22.copyToGPU();
    }

    void mergeMatrix(BasicMatrix& C, const BasicMatrix& C11, const BasicMatrix& C12,
                     const BasicMatrix& C21, const BasicMatrix& C22) override {
        int n = C.getDim();
        int oldSize = n / 2;
        
        // First, copy matrices from GPU to CPU for easier merging
        // (This could be optimized with a custom CUDA kernel for better performance)
        double* dst = C.getData();
        
        const double* c11 = C11.getData();
        const double* c12 = C12.getData();
        const double* c21 = C21.getData();
        const double* c22 = C22.getData();
        
        #pragma omp parallel for
        for (int i = 0; i < oldSize; i++) {
            for (int j = 0; j < oldSize; j++) {
                dst[i * n + j] = c11[i * oldSize + j];
                dst[i * n + j + oldSize] = c12[i * oldSize + j];
                dst[(i + oldSize) * n + j] = c21[i * oldSize + j];
                dst[(i + oldSize) * n + j + oldSize] = c22[i * oldSize + j];
            }
        }
        
        // Copy back to GPU
        C.copyToGPU();
    }

    void strassenMultiply(const BasicMatrix& A, const BasicMatrix& B, BasicMatrix& C) {
        int n = A.getDim();
        
        if (n <= THRESHOLD) {
            matrixMul(A, B, C);
            return;
        }
        
        int newSize = n / 2;
        
        // Create submatrices
        CudaMatrix A11(newSize), A12(newSize), A21(newSize), A22(newSize);
        CudaMatrix B11(newSize), B12(newSize), B21(newSize), B22(newSize);
        CudaMatrix C11(newSize), C12(newSize), C21(newSize), C22(newSize);
        
        // Split input matrices
        splitMatrix(A, A11, A12, A21, A22);
        splitMatrix(B, B11, B12, B21, B22);
        
        // Temporary matrices for intermediate results
        CudaMatrix M1(newSize), M2(newSize), M3(newSize), M4(newSize);
        CudaMatrix M5(newSize), M6(newSize), M7(newSize);
        
        // Temporary matrices for additions/subtractions
        CudaMatrix temp1(newSize), temp2(newSize);
        
        // Create CUDA streams for potential parallelism
        cudaStream_t streams[7];
        for (int i = 0; i < 7; i++) {
            cudaStreamCreate(&streams[i]);
        }
        
        // M1 = (A11 + A22) * (B11 + B22)
        matrixAdd(A11, A22, temp1);
        matrixAdd(B11, B22, temp2);
        strassenMultiply(temp1, temp2, M1);
        
        // M2 = (A21 + A22) * B11
        matrixAdd(A21, A22, temp1);
        strassenMultiply(temp1, B11, M2);
        
        // M3 = A11 * (B12 - B22)
        matrixSub(B12, B22, temp1);
        strassenMultiply(A11, temp1, M3);
        
        // M4 = A22 * (B21 - B11)
        matrixSub(B21, B11, temp1);
        strassenMultiply(A22, temp1, M4);
        
        // M5 = (A11 + A12) * B22
        matrixAdd(A11, A12, temp1);
        strassenMultiply(temp1, B22, M5);
        
        // M6 = (A21 - A11) * (B11 + B12)
        matrixSub(A21, A11, temp1);
        matrixAdd(B11, B12, temp2);
        strassenMultiply(temp1, temp2, M6);
        
        // M7 = (A12 - A22) * (B21 + B22)
        matrixSub(A12, A22, temp1);
        matrixAdd(B21, B22, temp2);
        strassenMultiply(temp1, temp2, M7);
        
        // Synchronize all streams
        for (int i = 0; i < 7; i++) {
            cudaStreamSynchronize(streams[i]);
        }
        
        // Calculate C11, C12, C21, C22
        // C11 = M1 + M4 - M5 + M7
        matrixAdd(M1, M4, temp1);
        matrixSub(temp1, M5, temp2);
        matrixAdd(temp2, M7, C11);
        
        // C12 = M3 + M5
        matrixAdd(M3, M5, C12);
        
        // C21 = M2 + M4
        matrixAdd(M2, M4, C21);
        
        // C22 = M1 + M3 - M2 + M6
        matrixAdd(M1, M3, temp1);
        matrixSub(temp1, M2, temp2);
        matrixAdd(temp2, M6, C22);
        
        // Merge results into C
        mergeMatrix(C, C11, C12, C21, C22);
        
        // Destroy streams
        for (int i = 0; i < 7; i++) {
            cudaStreamDestroy(streams[i]);
        }
    }
};

class CudaStrassen : public Strassen {
public:
    CudaStrassen(int dim) : Strassen(dim) {
        A = new CudaMatrix(dim);
        B = new CudaMatrix(dim);
        C = new CudaMatrix(dim);
    }

    ~CudaStrassen() {}

    void prepare() override {
        A->generateRandomMatrix();
        B->generateRandomMatrix();
        A->copyToGPU();
        B->copyToGPU();
    }

    // Strassen matrix multiplication implementation
    void run() override {
        static_cast<CudaMatrix*>(C)->strassenMultiply(*A, *B, *C);
        C->copyFromGPU();  // Get the result back to CPU memory
    }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <number_of_runs>\n";
        return 1;
    }
    
    int n = std::atoi(argv[1]);
    int max_run = std::atoi(argv[2]);
    
    cudaFree(0);
    
    CudaStrassen strassen(n);
    double milliseconds = strassen.test(max_run);
    std::cout << milliseconds << std::endl;
    
    return 0;
}
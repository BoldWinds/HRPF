#include "strassen.h"
#define THRESHOLD 256

class OmpMatrix : public BasicMatrix {
public:
    OmpMatrix(int dim) : BasicMatrix(dim) {}

    void matrixAdd(const BasicMatrix& A, const BasicMatrix& B, BasicMatrix& C) override {
        int n = A.getDim();
        const double* a = A.getData();
        const double* b = B.getData();
        double* c = C.getData();
        
        #pragma omp parallel for
        for (int i = 0; i < n * n; ++i) {
            c[i] = a[i] + b[i];
        }
    }

    void matrixSub(const BasicMatrix& A, const BasicMatrix& B, BasicMatrix& C) override {
        int n = A.getDim();
        const double* a = A.getData();
        const double* b = B.getData();
        double* c = C.getData();
        
        #pragma omp parallel for
        for (int i = 0; i < n * n; ++i) {
            c[i] = a[i] - b[i];
        }
    }

    void matrixMul(const BasicMatrix& A, const BasicMatrix& B, BasicMatrix& C) override {
        int n = A.getDim();
        const double* a = A.getData();
        const double* b = B.getData();
        double* c = C.getData();
        
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    sum += a[i*n + k] * b[k*n + j];
                }
                c[i*n + j] = sum;
            }
        }
    }

    void splitMatrix(const BasicMatrix& A, BasicMatrix& A11, BasicMatrix& A12, 
                     BasicMatrix& A21, BasicMatrix& A22) override {
        int newSize = A.getDim() / 2;
        const double* src = A.getData();
        int srcDim = A.getDim();

        double *a11 = A11.getData();
        double *a12 = A12.getData();
        double *a21 = A21.getData();
        double *a22 = A22.getData();
        
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < newSize; i++) {
            for (int j = 0; j < newSize; j++) {
                a11[i * newSize + j] = src[i * srcDim + j];
                a12[i * newSize + j] = src[i * srcDim + j + newSize];
                a21[i * newSize + j] = src[(i + newSize) * srcDim + j];
                a22[i * newSize + j] = src[(i + newSize) * srcDim + j + newSize];
            }
        }
    }

    void mergeMatrix(BasicMatrix& C, const BasicMatrix& C11, const BasicMatrix& C12,
                     const BasicMatrix& C21, const BasicMatrix& C22) override {
        int oldSize = C11.getDim();
        double* dst = C.getData();
        int dstDim = C.getDim();
        
        const double* c11 = C11.getData();
        const double* c12 = C12.getData();
        const double* c21 = C21.getData();
        const double* c22 = C22.getData();
        
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < oldSize; i++) {
            for (int j = 0; j < oldSize; j++) {
                dst[i * dstDim + j] = c11[i * oldSize + j];
                dst[i * dstDim + j + oldSize] = c12[i * oldSize + j];
                dst[(i + oldSize) * dstDim + j] = c21[i * oldSize + j];
                dst[(i + oldSize) * dstDim + j + oldSize] = c22[i * oldSize + j];
            }
        }
    }

    void strassenMultiply(const BasicMatrix& A, const BasicMatrix& B, BasicMatrix& C) {
        int n = A.getDim();
        
        if (n <= THRESHOLD) {
            matrixMul(A, B, C);
            return;
        }
        
        int newSize = n / 2;
        
        OmpMatrix A11(newSize), A12(newSize), A21(newSize), A22(newSize);
        OmpMatrix B11(newSize), B12(newSize), B21(newSize), B22(newSize);
        OmpMatrix C11(newSize), C12(newSize), C21(newSize), C22(newSize);
        
        splitMatrix(A, A11, A12, A21, A22);
        splitMatrix(B, B11, B12, B21, B22);
        
        // Temporary matrices for intermediate results
        OmpMatrix M1(newSize), M2(newSize), M3(newSize), M4(newSize);
        OmpMatrix M5(newSize), M6(newSize), M7(newSize);
        
        // Temporary matrices for additions/subtractions
        OmpMatrix temp1(newSize), temp2(newSize);
        
        // 使用OpenMP任务并行计算7个子矩阵乘法
        #pragma omp parallel
        {
            #pragma omp single
            {
                // M1 = (A11 + A22) * (B11 + B22)
                #pragma omp task
                {
                    OmpMatrix localTemp1(newSize), localTemp2(newSize);
                    matrixAdd(A11, A22, localTemp1);
                    matrixAdd(B11, B22, localTemp2);
                    strassenMultiply(localTemp1, localTemp2, M1);
                }
                
                // M2 = (A21 + A22) * B11
                #pragma omp task
                {
                    OmpMatrix localTemp1(newSize);
                    matrixAdd(A21, A22, localTemp1);
                    strassenMultiply(localTemp1, B11, M2);
                }
                
                // M3 = A11 * (B12 - B22)
                #pragma omp task
                {
                    OmpMatrix localTemp1(newSize);
                    matrixSub(B12, B22, localTemp1);
                    strassenMultiply(A11, localTemp1, M3);
                }
                
                // M4 = A22 * (B21 - B11)
                #pragma omp task
                {
                    OmpMatrix localTemp1(newSize);
                    matrixSub(B21, B11, localTemp1);
                    strassenMultiply(A22, localTemp1, M4);
                }
                
                // M5 = (A11 + A12) * B22
                #pragma omp task
                {
                    OmpMatrix localTemp1(newSize);
                    matrixAdd(A11, A12, localTemp1);
                    strassenMultiply(localTemp1, B22, M5);
                }
                
                // M6 = (A21 - A11) * (B11 + B12)
                #pragma omp task
                {
                    OmpMatrix localTemp1(newSize), localTemp2(newSize);
                    matrixSub(A21, A11, localTemp1);
                    matrixAdd(B11, B12, localTemp2);
                    strassenMultiply(localTemp1, localTemp2, M6);
                }
                
                // M7 = (A12 - A22) * (B21 + B22)
                #pragma omp task
                {
                    OmpMatrix localTemp1(newSize), localTemp2(newSize);
                    matrixSub(A12, A22, localTemp1);
                    matrixAdd(B21, B22, localTemp2);
                    strassenMultiply(localTemp1, localTemp2, M7);
                }
                
                // 等待所有任务完成
                #pragma omp taskwait
                
                // 计算结果矩阵
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
            }
        }
        
        // Merge results into C
        mergeMatrix(C, C11, C12, C21, C22);
    }
};

class CpuOmpStrassen : public Strassen {
public:
    CpuOmpStrassen(int dim) : Strassen(dim) {
        A = new OmpMatrix(dim);
        B = new OmpMatrix(dim);
        C = new OmpMatrix(dim);
    }

    ~CpuOmpStrassen() {}

    void prepare() override {
        A->generateRandomMatrix();
        B->generateRandomMatrix();
    }

    void run() override {
        static_cast<OmpMatrix*>(C)->strassenMultiply(*A, *B, *C);
    }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <number_of_runs>\n";
        return 1;
    }
    
    int n = std::atoi(argv[1]);
    int max_run = std::atoi(argv[2]);
    
    omp_set_num_threads(omp_get_num_procs());
    
    CpuOmpStrassen strassen(n);
    double milliseconds = strassen.test(max_run);
    std::cout << milliseconds << std::endl;
    return 0;
}
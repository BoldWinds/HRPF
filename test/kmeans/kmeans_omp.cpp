#include <iostream>
#include <fstream>
#include <cmath>
#include <cfloat>
#include <sys/time.h>
#include <omp.h>
using namespace std;

class KMEANS
{
private:
    int k;
    int samples;
    int features;
    double **dataSet;
    double **centroids;
    int *labels;
    
    // 计算两个点之间的欧氏距离
    inline double euclideanDistance(double *point1, double *point2)
    {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int i = 0; i < features; i++)
        {
            double diff = point1[i] - point2[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }
    
    // 随机初始化质心
    void initCentroids()
    {
        bool *used = new bool[samples]();
        int count = 0;

        while (count < k)
        {
            int idx = rand() % samples;
            if (!used[idx])
            {
                for (int j = 0; j < features; j++)
                {
                    centroids[count][j] = dataSet[idx][j];
                }
                used[idx] = true;
                count++;
            }
        }

        delete[] used;
    }

public:
    KMEANS(int k, int samples, int features)
    {
        this->k = k;
        this->samples = samples;
        this->features = features;

        dataSet = new double*[samples];
        for (int i = 0; i < samples; ++i)
        {
            dataSet[i] = new double[features];
        }
        
        centroids = new double*[k];
        for (int i = 0; i < k; ++i)
        {
            centroids[i] = new double[features];
        }
        
        labels = new int[samples];

        srand(time(NULL));

        std::ifstream fin;
        std::string filename = "./data/kmdata_" + std::to_string(samples) + "_8.txt";
        fin.open(filename);

        if (!fin)
        {
            std::cout << "can not open the file data.txt" << std::endl;
            exit(1);
        }

        /* input the dataSet */
        for (int i = 0; i < samples; i++)
        {
            for (int j = 0; j < features; j++)
            {
                fin >> dataSet[i][j];
            }
        }
        fin.close();
        // 初始化质心
        initCentroids();
    }

    ~KMEANS()
    {
        for (int i = 0; i < samples; i++)
        {
            delete[] dataSet[i];
        }
        delete[] dataSet;

        for (int i = 0; i < k; i++)
        {
            delete[] centroids[i];
        }
        delete[] centroids;

        delete[] labels;
    }

    void run(double tolerance = 0.000001)
    {
        bool converged = false;
        
        // 预先分配重用的内存
        double **oldCentroids = new double*[k];
        for (int i = 0; i < k; i++)
        {
            oldCentroids[i] = new double[features];
        }
        
        // 预先分配簇大小和累加器数组
        double **newCentroids = new double*[k];
        for (int i = 0; i < k; i++)
        {
            newCentroids[i] = new double[features];
        }
        
        int *clusterSizes = new int[k];
        
        while (!converged)
        {   
            // 1. 为每个样本分配最近的质心 - 这是计算的主要部分
            #pragma omp parallel for
            for (int i = 0; i < samples; i++)
            {
                double minDist = DBL_MAX;
                int minIdx = 0;

                for (int j = 0; j < k; j++)
                {
                    double dist = euclideanDistance(dataSet[i], centroids[j]);
                    if (dist < minDist)
                    {
                        minDist = dist;
                        minIdx = j;
                    }
                }

                labels[i] = minIdx;
            }

            // 2. 保存旧的质心用于检查收敛
            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    oldCentroids[i][j] = centroids[i][j];
                }
            }

            // 3. 重置新质心和簇大小计数器
            for (int i = 0; i < k; i++)
            {
                clusterSizes[i] = 0;
                for (int j = 0; j < features; j++)
                {
                    newCentroids[i][j] = 0.0;
                }
            }

            // 4. 累加每个簇中的所有点 - 简化的并行实现
            #pragma omp parallel
            {
                // 每个线程的私有计数器
                int *privateClusterSizes = new int[k]();
                double **privateNewCentroids = new double*[k];
                for (int i = 0; i < k; i++)
                {
                    privateNewCentroids[i] = new double[features]();
                }
                
                // 并行处理所有样本点
                #pragma omp for nowait
                for (int i = 0; i < samples; i++)
                {
                    int cluster = labels[i];
                    privateClusterSizes[cluster]++;
                    
                    for (int j = 0; j < features; j++)
                    {
                        privateNewCentroids[cluster][j] += dataSet[i][j];
                    }
                }
                
                // 合并结果到共享变量
                #pragma omp critical
                {
                    for (int i = 0; i < k; i++)
                    {
                        clusterSizes[i] += privateClusterSizes[i];
                        for (int j = 0; j < features; j++)
                        {
                            newCentroids[i][j] += privateNewCentroids[i][j];
                        }
                    }
                }
                
                // 清理私有数据
                for (int i = 0; i < k; i++)
                {
                    delete[] privateNewCentroids[i];
                }
                delete[] privateNewCentroids;
                delete[] privateClusterSizes;
            }

            // 5. 计算平均值，更新质心
            for (int i = 0; i < k; i++)
            {
                if (clusterSizes[i] > 0)
                {
                    for (int j = 0; j < features; j++)
                    {
                        centroids[i][j] = newCentroids[i][j] / clusterSizes[i];
                    }
                }
            }

            // 6. 检查是否收敛
            converged = true;
            for (int i = 0; i < k && converged; i++)
            {
                double dist = euclideanDistance(centroids[i], oldCentroids[i]);
                if (dist > tolerance)
                {
                    converged = false;
                }
            }
        }
        
        // 释放临时内存
        for (int i = 0; i < k; i++)
        {
            delete[] oldCentroids[i];
            delete[] newCentroids[i];
        }
        delete[] oldCentroids;
        delete[] newCentroids;
        delete[] clusterSizes;
    }

    // 计算总体误差（每个点到其对应质心的距离平方和）
    double getTotalError()
    {
        double totalError = 0.0;
        
        #pragma omp parallel reduction(+:totalError)
        {
            #pragma omp for 
            for (int i = 0; i < samples; i++)
            {
                int cluster = labels[i];
                double dist = euclideanDistance(dataSet[i], centroids[cluster]);
                totalError += dist * dist;
            }
        }
        
        return totalError;
    }
};

int main(int argc, char **argv)
{
    int samples = std::atoi(argv[1]);
    const int features = 8;
    KMEANS kms(3, samples, features);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    kms.run();
    gettimeofday(&end, NULL);
    double milliseconds = (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    std::cout << milliseconds << std::endl;

    double error = kms.getTotalError();
    std::cout << "Total error: " << error << std::endl;
    return 0;
}

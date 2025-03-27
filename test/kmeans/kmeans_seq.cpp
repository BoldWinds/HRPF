#include <iostream>
#include <fstream>
#include <cmath>
#include <cfloat>
#include <sys/time.h>
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
    double euclideanDistance(double *point1, double *point2)
    {
        double sum = 0.0;
        for (int i = 0; i < features; i++)
        {
            sum += pow(point1[i] - point2[i], 2);
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

    void run()
    {
        int converged = 10;
        while (converged--)
        {   
            // 为每个样本分配最近的质心
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

            // 保存旧的质心用于检查收敛
            double **oldCentroids = new double *[k];
            for (int i = 0; i < k; i++)
            {
                oldCentroids[i] = new double[features];
                for (int j = 0; j < features; j++)
                {
                    oldCentroids[i][j] = centroids[i][j];
                }
            }

            // 更新质心位置
            // 首先计算每个簇的样本数
            int *clusterSizes = new int[k]();

            // 重置质心
            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    centroids[i][j] = 0.0;
                }
            }

            // 累加每个簇中的所有点
            for (int i = 0; i < samples; i++)
            {
                int cluster = labels[i];
                clusterSizes[cluster]++;

                for (int j = 0; j < features; j++)
                {
                    centroids[cluster][j] += dataSet[i][j];
                }
            }

            // 计算平均值
            for (int i = 0; i < k; i++)
            {
                if (clusterSizes[i] > 0)
                { // 防止除以零
                    for (int j = 0; j < features; j++)
                    {
                        centroids[i][j] /= clusterSizes[i];
                    }
                }
            }

            // 释放旧质心内存
            for (int i = 0; i < k; i++)
            {
                delete[] oldCentroids[i];
            }
            delete[] oldCentroids;
            delete[] clusterSizes;
        }
    }

    // 计算总体误差（每个点到其对应质心的距离平方和）
    double getTotalError()
    {
        double totalError = 0.0;
        for (int i = 0; i < samples; i++)
        {
            int cluster = labels[i];
            totalError += pow(euclideanDistance(dataSet[i], centroids[cluster]), 2);
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

    //double error = kms.getTotalError();
    //std::cout << "Total error: " << error << std::endl;
    return 0;
}

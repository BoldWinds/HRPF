#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
#include <sys/time.h>
#include <omp.h>

class KNN
{
private:
    int k;
    std::vector<std::vector<double>> X_train;
    std::vector<int> Y_train;
    std::vector<std::vector<double>> X_test;
    std::vector<int> Y_test;
    int samples;
    int features;
    int tests;
    double computeDistance(const std::vector<double> &a, const std::vector<double> &b) const
    {
        double dist = 0.0;
        for (size_t i = 0; i < features; i++)
        {
            dist += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return std::sqrt(dist);
    }
    int predictOne(const std::vector<double> &x) const
    {
        std::vector<std::pair<double, int>> distances;
        for (size_t i = 0; i < samples; i++)
        {
            double d = computeDistance(x, X_train[i]);
            distances.push_back({d, Y_train[i]});
        }
        std::sort(distances.begin(), distances.end(),
                  [](const std::pair<double, int> &a, const std::pair<double, int> &b)
                  {
                      return a.first < b.first;
                  });
        std::map<int, int> votes;
        for (int i = 0; i < k && i < samples; i++)
        {
            votes[distances[i].second]++;
        }
        int best_label = -1;
        int max_votes = 0;
        for (const auto &vote : votes)
        {
            if (vote.second > max_votes)
            {
                max_votes = vote.second;
                best_label = vote.first;
            }
        }
        return best_label;
    }

public:
    KNN(int k,int samples, int features, int tests)
    {
        this->k = k;
        this->X_train = std::vector<std::vector<double>>(samples, std::vector<double>(features));
        this->Y_train = std::vector<int>(samples);
        this->X_test = std::vector<std::vector<double>>(tests, std::vector<double>(features));
        this->Y_test = std::vector<int>(tests);
        this->samples = samples;
        this->features = features;
        this->tests = tests;

        std::ifstream finX, tfinX, finY, tfinY;
        finX.open("./data/train.txt");
        finY.open("./data/labeltrain.txt");

        if (!finX && !finY)
        {
            std::cout << "can not open the data file" << std::endl;
            exit(1);
        }

        for (int i = 0; i < samples; i++)
        {
            for (int j = 0; j < features; j++)
            {
                finX >> X_train[i][j];
            }
            finY >> Y_train[i];
        }

        tfinX.open("./data/test.txt");
        tfinY.open("./data/labeltest.txt");
        for (int i = 0; i < tests; ++i)
        {
            for (int j = 0; j < features; j++)
            {
                tfinX >> X_test[i][j];
            }
            tfinY >> Y_test[i];
        }
    }
    // 批量预测函数：对一组样本进行预测
    std::vector<int> predict() const
    {
        std::vector<int> predictions;
        for (const auto &sample : X_test)
        {
            predictions.push_back(predictOne(sample));
        }
        return predictions;
    }
    double score() const {
        int correct = 0;
        std::vector<int> preds = predict();
        for (size_t i = 0; i < tests; i++) {
            if (preds[i] == Y_test[i]) {
                correct++;
            }
        }
        return static_cast<double>(correct) / tests;
    }
};

int main(int argc, char **argv)
{
    KNN knn(3, 4000, 8, 1000);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    double acc = knn.score();
    std::cout << "Accuracy: " << acc << std::endl;
    gettimeofday(&end, NULL);
    double milliseconds = (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    std::cout << milliseconds << std::endl;
    return 0;
}
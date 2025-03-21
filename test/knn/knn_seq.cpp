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
    double **X_train;
    int *Y_train;
    double **X_test;
    int *Y_test;
    int samples;
    int features;
    int tests;

    double computeDistance(const double *a, const double *b) const
    {
        double dist = 0.0;
        for (int i = 0; i < features; i++)
        {
            dist += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return std::sqrt(dist);
    }

    int predictOne(const double *x) const
    {
        // Use arrays for distances instead of vector<pair>
        double *distances = new double[samples];
        int *labels = new int[samples];

        for (int i = 0; i < samples; i++)
        {
            distances[i] = computeDistance(x, X_train[i]);
            labels[i] = Y_train[i];
        }

        // Sort based on distances (simplified sort for the example)
        for (int i = 0; i < k; i++)
        {
            for (int j = i + 1; j < samples; j++)
            {
                if (distances[j] < distances[i])
                {
                    std::swap(distances[i], distances[j]);
                    std::swap(labels[i], labels[j]);
                }
            }
        }

        // Count votes
        int best_label = -1;
        int max_votes = 0;

        // Simple voting algorithm using arrays
        for (int i = 0; i < k; i++)
        {
            int votes = 0;
            int current_label = labels[i];

            for (int j = 0; j < k; j++)
            {
                if (labels[j] == current_label)
                    votes++;
            }

            if (votes > max_votes)
            {
                max_votes = votes;
                best_label = current_label;
            }
        }

        delete[] distances;
        delete[] labels;
        return best_label;
    }

public:
    KNN(int k, int samples, int features, int tests)
    {
        this->k = k;
        this->samples = samples;
        this->features = features;
        this->tests = tests;

        // Allocate memory for training data
        X_train = new double *[samples];
        for (int i = 0; i < samples; i++)
        {
            X_train[i] = new double[features];
        }
        Y_train = new int[samples];

        // Allocate memory for test data
        X_test = new double *[tests];
        for (int i = 0; i < tests; i++)
        {
            X_test[i] = new double[features];
        }
        Y_test = new int[tests];

        // Read training data
        std::ifstream finX, tfinX, finY, tfinY;
        finX.open("./data/train.txt");
        finY.open("./data/labeltrain.txt");

        if (!finX || !finY)
        {
            std::cout << "Cannot open the training data files" << std::endl;
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

        // Read test data
        tfinX.open("./data/test.txt");
        tfinY.open("./data/labeltest.txt");

        if (!tfinX || !tfinY)
        {
            std::cout << "Cannot open the test data files" << std::endl;
            exit(1);
        }

        for (int i = 0; i < tests; i++)
        {
            for (int j = 0; j < features; j++)
            {
                tfinX >> X_test[i][j];
            }
            tfinY >> Y_test[i];
        }
    }

    int *predict() const
    {
        int *predictions = new int[tests];

        for (int i = 0; i < tests; i++)
        {
            predictions[i] = predictOne(X_test[i]);
        }

        return predictions;
    }

    double score() const
    {
        int correct = 0;
        int *preds = predict();

        for (int i = 0; i < tests; i++)
        {
            if (preds[i] == Y_test[i])
            {
                correct++;
            }
        }

        double accuracy = static_cast<double>(correct) / tests;
        delete[] preds;
        return accuracy;
    }
};

int main(int argc, char **argv)
{
    int num_tests = std::stoi(argv[1]);
    KNN knn(3, 4000, 8, num_tests);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    double acc = knn.score();
    gettimeofday(&end, NULL);
    double milliseconds = (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    std::cout << milliseconds << std::endl;
    return 0;
}
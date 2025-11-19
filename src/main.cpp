#include <iostream>
#include <vector>

#include "GroupedRegressionTree.h"

int main() {
    int n_samples = 5000000;
    std::vector<std::vector<double>> X(n_samples, std::vector<double>(2));
    std::vector<double> y(n_samples, 0.0);
    std::vector<int> g(n_samples, 0);

    for(int i=0; i<n_samples; ++i) {
        X[i][0] = (rand() % 100) / 10.0;
        X[i][1] = (rand() % 100) / 10.0;

        g[i] = rand() % 3;
        if (g[i] == 0) {
            y[i] = 2 * X[i][0] + 5 * X[i][1];
        } else if (g[i] == 1) {
            y[i] = 7 * X[i][0] + 3 * X[i][1];
        } else {
            y[i] = 8 * X[i][0] - 6 * X[i][1];
        }
    }

    std::cout << "Training on " << n_samples << " samples..." << std::endl;

    GroupedRegressionTree tree(10, 5, 3);
    tree.fit(X, y, g);

    std::vector<double> sample = {1.0, 4.0};
    std::vector<double> prediction = tree.predict_single(sample);

    std::cout << "Input: [7.0, 4.0]" << std::endl;
    std::cout << "Predicted: ";
    for (double val : prediction) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    std::cout << "Actual Target: [" << (2 * sample[0] + 5 * sample[1])
        << ", " << (7 * sample[0] + 3 * sample[1]) << ", " << (8 * sample[0] - 6 * sample[1])
        << "]" << std::endl;

    return 0;
}

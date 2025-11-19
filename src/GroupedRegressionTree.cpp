#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <memory>

#include "GroupedRegressionTree.h"

void GroupedRegressionTree::fit(
    const std::vector<std::vector<double>>& X, 
    const std::vector<double>& y,
    const std::vector<int>& g
) {
    std::vector<int> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    root = build_tree(X, y, g, indices, 0);
}

std::vector<double> GroupedRegressionTree::predict_single(
    const std::vector<double>& sample
) const {
    Node* node = root.get();
    while (!node->is_leaf) {
        if (sample[node->feature_index] <= node->threshold) {
            node = node->left.get();
        } else {
            node = node->right.get();
        }
    }
    return node->values;
}

std::vector<std::vector<double>> GroupedRegressionTree::predict(
    const std::vector<std::vector<double>>& X
) const {
    std::vector<std::vector<double>> predictions;
    predictions.reserve(X.size());
    for (const auto& row : X) {
        predictions.push_back(predict_single(row));
    }
    return predictions;
}

std::vector<double> GroupedRegressionTree::calculate_leaf_values(
    const std::vector<double>& y, 
    const std::vector<int>& g,
    const std::vector<int>& indices
) {
    std::vector<double> sum(output_size, 0.0);
    std::vector<int> count(output_size, 0);
    for (int idx : indices) {
        sum[g[idx]] += y[idx];
        count[g[idx]] += 1;
    }

    for (int k = 0; k < output_size; ++k) {
        sum[k] /= count[k];
    }
    return sum;
}

std::unique_ptr<Node> GroupedRegressionTree::build_tree(
    const std::vector<std::vector<double>>& X, 
    const std::vector<double>& y,
    const std::vector<int>& g,
    std::vector<int>& indices, 
    int depth
) {
    std::unique_ptr<Node> node = std::make_unique<Node>();

    // stopping criteria
    if (indices.size() < min_samples_split || depth >= max_depth) {
        node->is_leaf = true;
        node->values = calculate_leaf_values(y, g, indices);
        return node;
    }

    // find best split
    int best_feature = -1;
    double best_thresh = 0.0;
    double best_score = std::numeric_limits<double>::max();
    
    std::vector<double> sum_y(output_size, 0.0);
    std::vector<double> sum_yy(output_size, 0.0);

    for (int idx : indices) {
        double val = y[idx];
        sum_y[g[idx]] += val;
        sum_yy[g[idx]] += val * val;
    }

    int n_samples = indices.size();
    int n_features = X[0].size();

    for (int f = 0; f < n_features; ++f) {
        std::sort(indices.begin(), indices.end(), 
            [&X, f](int a, int b) { return X[a][f] < X[b][f]; });

        std::vector<double> left_sum_y(output_size, 0.0);
        std::vector<double> left_sum_yy(output_size, 0.0);

        for (int i = 0; i < n_samples - 1; ++i) {
            int idx = indices[i];
            double y_val = y[idx];
            
            left_sum_y[g[idx]] += y_val;
            left_sum_yy[g[idx]] += y_val * y_val;
            if (X[indices[i]][f] == X[indices[i+1]][f]) continue;

            int n_left = i + 1;
            int n_right = n_samples - n_left;

            // calculate score
            double current_total_score = 0.0;
            for (int k = 0; k < output_size; ++k) {
                double sse_left = left_sum_yy[k] 
                    - (left_sum_y[k] * left_sum_y[k] / n_left);

                double right_sum_y = sum_y[k] - left_sum_y[k];
                double right_sum_yy = sum_yy[k] - left_sum_yy[k];
                double sse_right = right_sum_yy 
                    - (right_sum_y * right_sum_y / n_right);

                current_total_score += sse_left + sse_right;
            }

            if (current_total_score < best_score) {
                best_score = current_total_score;
                best_feature = f;
                best_thresh = (X[indices[i]][f] + X[indices[i+1]][f]) / 2.0;
            }
        }
    }

    // If no split improved the score (or all features identical)
    if (best_feature == -1) {
        node->is_leaf = true;
        node->values = calculate_leaf_values(y, g, indices);
        return node;
    }

    // Perform the split
    std::vector<int> left_indices;
    std::vector<int> right_indices;
    left_indices.reserve(indices.size());
    right_indices.reserve(indices.size());

    for (int idx : indices) {
        if (X[idx][best_feature] <= best_thresh) {
            left_indices.push_back(idx);
        } else {
            right_indices.push_back(idx);
        }
    }

    node->feature_index = best_feature;
    node->threshold = best_thresh;
    node->left = build_tree(X, y, g, left_indices, depth + 1);
    node->right = build_tree(X, y, g, right_indices, depth + 1);

    return node;
}

int main() {
    int n_samples = 500000;
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

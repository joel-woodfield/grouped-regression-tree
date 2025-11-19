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
	// presort indices for each feature
	std::vector<std::vector<int>> sorted_indices(X[0].size(), std::vector<int>(X.size()));
	for (size_t f = 0; f < X[0].size(); ++f) {
		std::iota(sorted_indices[f].begin(), sorted_indices[f].end(), 0);
		std::sort(sorted_indices[f].begin(), sorted_indices[f].end(),
			[&X, f](int a, int b) { return X[a][f] < X[b][f]; });
	}
    
    root = build_tree(X, y, g, sorted_indices, 0);
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
) const {
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
    std::vector<std::vector<int>>& sorted_indices, 
    int depth
) {
    std::unique_ptr<Node> node = std::make_unique<Node>();
	int n_samples = sorted_indices[0].size();
	int n_features = sorted_indices.size();

    // stopping criteria
    if (n_samples < min_samples_split || depth >= max_depth) {
        node->is_leaf = true;
        node->values = calculate_leaf_values(y, g, sorted_indices[0]);
        return node;
    }

    // find best split
    int best_feature = -1;
    double best_thresh = 0.0;
    double best_score = std::numeric_limits<double>::max();
    
    std::vector<double> sum_y(output_size, 0.0);
    std::vector<double> sum_yy(output_size, 0.0);

    for (int idx : sorted_indices[0]) {
        double val = y[idx];
        sum_y[g[idx]] += val;
        sum_yy[g[idx]] += val * val;
    }

    for (int f = 0; f < n_features; ++f) {
        std::vector<double> left_sum_y(output_size, 0.0);
        std::vector<double> left_sum_yy(output_size, 0.0);

        for (int i = 0; i < n_samples - 1; ++i) {
            int idx = sorted_indices[f][i];
            double y_val = y[idx];
            
            left_sum_y[g[idx]] += y_val;
            left_sum_yy[g[idx]] += y_val * y_val;
            if (X[sorted_indices[f][i]][f] == X[sorted_indices[f][i+1]][f]) continue;

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
                best_thresh = (X[sorted_indices[f][i]][f] + X[sorted_indices[f][i+1]][f]) / 2.0;
            }
        }
    }

    // If no split improved the score (or all features identical)
    if (best_feature == -1) {
        node->is_leaf = true;
        node->values = calculate_leaf_values(y, g, sorted_indices[0]);
        return node;
    }

    // Perform the split
	std::vector<std::vector<int>> left_indices(n_features);
	std::vector<std::vector<int>> right_indices(n_features);

	for (int f = 0; f < n_features; ++f) {
		left_indices[f].reserve(n_samples);
		right_indices[f].reserve(n_samples);

		for (int idx : sorted_indices[f]) {
			if (X[idx][best_feature] <= best_thresh) {
				left_indices[f].push_back(idx);
			} else {
				right_indices[f].push_back(idx);
			}
		}
	}

    node->feature_index = best_feature;
    node->threshold = best_thresh;
    node->left = build_tree(X, y, g, left_indices, depth + 1);
    node->right = build_tree(X, y, g, right_indices, depth + 1);

    return node;
}


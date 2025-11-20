#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <sstream>
#include <fstream>

#include "GroupedRegressionTree.h"

inline double at(const double* X, int i, int j, int row_size) {
    return X[i * row_size + j];
}

inline double& at(double* X, int i, int j, int row_size) {
    return X[i * row_size + j];
}

void write_node(std::ostream& out, Node* node) {
    if (!node) return;

    // use the node's memory address as a unique ID
    unsigned long long id = reinterpret_cast<unsigned long long>(node);

    // construct label
    std::stringstream label;
    std::string shape;
    std::string color;

    if (node->is_leaf) {
        shape = "box";
        color = "lightblue";
        label << "LEAF\\nVal: [";
        for (size_t i = 0; i < node->values.size(); ++i) {
            label << node->values[i] << (i < node->values.size() - 1 ? ", " : "");
        }
        label << "]";
    } else {
        shape = "ellipse";
        color = "lightgrey";
        label << "Feature " << node->feature_index << "\\n<= " << node->threshold;
    }

    // node definition
    out << "    node_" << id << " [label=\"" << label.str() 
        << "\", shape=" << shape 
        << ", style=filled, fillcolor=" << color << "];\n";

    // children
    if (node->left) {
        unsigned long long left_id = reinterpret_cast<unsigned long long>(node->left.get());
        
        // Recursive call
        write_node(out, node->left.get());
        
        // Write Edge
        out << "    node_" << id << " -> node_" << left_id 
            << " [label=\"True\", fontsize=10];\n";
    }

    if (node->right) {
        unsigned long long right_id = reinterpret_cast<unsigned long long>(node->right.get());
        
        // Recursive call
        write_node(out, node->right.get());
        
        // Write Edge
        out << "    node_" << id << " -> node_" << right_id 
            << " [label=\"False\", fontsize=10];\n";
    }
}

void GroupedRegressionTree::fit(
    const double* X,
    const double* y,
    const int* g,
    int n_samples,
    int n_features
) {
	// presort indices for each feature
	std::vector<std::vector<int>> sorted_indices(n_features, std::vector<int>(n_samples));
	for (size_t f = 0; f < n_features; ++f) {
		std::iota(sorted_indices[f].begin(), sorted_indices[f].end(), 0);
		std::sort(sorted_indices[f].begin(), sorted_indices[f].end(),
			[&X, f, n_features](int a, int b) { return at(X, a, f, n_features) < at(X, b, f, n_features); });
	}
    
    root = build_tree(X, y, g, sorted_indices, 0);
}

void GroupedRegressionTree::fit_py(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& X,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& y,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& g
) {
    auto X_view = X.unchecked<2>();
    int n_samples = X_view.shape(0);
    int n_features = X_view.shape(1);

    fit(
        static_cast<const double*>(X.request().ptr),
        static_cast<const double*>(y.request().ptr),
        static_cast<const int*>(g.request().ptr),
        n_samples,
        n_features
    );
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
    const double* X,
    int n_samples,
    int n_features
) const {
    std::vector<std::vector<double>> predictions;
    predictions.reserve(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        std::vector<double> sample(n_features);

        for (int j = 0; j < n_features; ++j) {
            sample[j] = at(X, i, j, n_features);
        }
        predictions.push_back(predict_single(sample));
    }
    return predictions;
}

std::vector<std::vector<double>> GroupedRegressionTree::predict_py(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& X
) const {
    auto X_view = X.unchecked<2>();
    int n_samples = X_view.shape(0);
    int n_features = X_view.shape(1);

    return predict(static_cast<const double*>(X.request().ptr), n_samples, n_features);
}

std::unique_ptr<GroupedRegressionTree> GroupedRegressionTree::clone() const {
    auto new_tree = std::make_unique<GroupedRegressionTree>(
        max_depth, min_samples_split, output_size
    );

    if (root) {
        new_tree->root = std::make_unique<Node>(*root);
    }
    return new_tree;
}

void GroupedRegressionTree::export_tree(const std::string& filename) const {
    std::ofstream out(filename);
    out << "digraph DecisionTree {\n";
    out << "    node [fontname=\"Arial\"];\n";
    out << "    edge [fontname=\"Arial\"];\n";
    
    if (root) {
        write_node(out, root.get());
    }
    
    out << "}\n";
    out.close();
    std::cout << "DOT file generated: " << filename << std::endl;
}

std::vector<double> GroupedRegressionTree::calculate_leaf_values(
    const double* y,
    const int* g,
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
    const double* X,
    const double* y,
    const int* g,
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
            if (at(X, sorted_indices[f][i], f, n_features) == at(X, sorted_indices[f][i+1], f, n_features)) continue;

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
                best_thresh = (at(X, sorted_indices[f][i], f, n_features) + at(X, sorted_indices[f][i+1], f, n_features)) / 2.0;
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
            if (at(X, idx, best_feature, n_features) <= best_thresh) {
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


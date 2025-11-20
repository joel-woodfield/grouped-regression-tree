#pragma once
#include <memory>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct Node {
    bool is_leaf = false;
    std::vector<double> values;
    int feature_index = -1;
    double threshold = 0.0;
    std::unique_ptr<Node> left = nullptr;
    std::unique_ptr<Node> right = nullptr;

    Node(int size = 1) : values(size, 0.0) {}

    Node(const Node& other) {
        is_leaf = other.is_leaf;
        values = other.values;
        feature_index = other.feature_index;
        threshold = other.threshold;
        if (other.left) {
            left = std::make_unique<Node>(*other.left);
        }
        if (other.right) {
            right = std::make_unique<Node>(*other.right);
        }
    }
};


class GroupedRegressionTree {
public:
    GroupedRegressionTree(
        int max_depth = 10, 
        int min_samples_split = 2, 
        int output_size = 1
    ) :
        max_depth(max_depth),
        min_samples_split(min_samples_split),
        output_size(output_size)
    {}

    void fit(
        const double* X,
        const double* y,
        const int* g,
        int n_samples,
        int n_features
    );

    void fit_py(
        const py::array_t<double>& X,
        const py::array_t<double>& y,
        const py::array_t<int>& g
    );

    std::vector<double> predict_single(const std::vector<double>& sample) const;

    std::vector<std::vector<double>> predict(
        const double* X,
        int n_samples,
        int n_features
    ) const;

    std::vector<std::vector<double>> predict_py(
        const py::array_t<double>& X
    ) const;

    std::unique_ptr<GroupedRegressionTree> clone() const;

    void export_tree(const std::string& filename) const;

private:
    int max_depth;
    int min_samples_split;
    int output_size;
    std::unique_ptr<Node> root;

    std::vector<double> calculate_leaf_values(
        const double* y, 
        const int* g,
        const std::vector<int>& indices
    ) const;

    std::unique_ptr<Node> build_tree(
        const double* X,
        const double* y,
        const int* g,
        std::vector<std::vector<int>>& sorted_indices, 
        int depth
    );

};



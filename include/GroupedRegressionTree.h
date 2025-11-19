#include <memory>
#include <vector>

struct Node {
    bool is_leaf = false;
    std::vector<double> values;
    int feature_index = -1;
    double threshold = 0.0;
    std::unique_ptr<Node> left = nullptr;
    std::unique_ptr<Node> right = nullptr;

    Node(int size = 1) : values(size, 0.0) {}
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
        const std::vector<std::vector<double>>& X, 
        const std::vector<double>& y,
        const std::vector<int>& g
    );

    std::vector<double> predict_single(const std::vector<double>& sample) const;

    std::vector<std::vector<double>> predict(
        const std::vector<std::vector<double>>& X
    ) const;

private:
    int max_depth;
    int min_samples_split;
    int output_size;
    std::unique_ptr<Node> root;

    std::vector<double> calculate_leaf_values(
        const std::vector<double>& y, 
        const std::vector<int>& g,
        const std::vector<int>& indices
    ) const;

    std::unique_ptr<Node> build_tree(
        const std::vector<std::vector<double>>& X, 
        const std::vector<double>& y,
        const std::vector<int>& g,
        std::vector<std::vector<int>>& sorted_indices, 
        int depth
    );

};



import numpy as np
from grouped_regression_tree import GroupedRegressionTree

def main():
    n_samples = 1_000_000
    n_features = 2
    n_groups = 3

    # Generate synthetic data
    X = np.random.rand(n_samples, n_features)
    y = np.zeros(n_samples)
    g = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        group = i % n_groups
        g[i] = group

        if group == 0:
            y[i] = 5 * X[i, 0] + X[i, 1] + np.random.normal(0, 0.1)
        elif group == 1:
            y[i] = X[i, 0] - 3 * X[i, 1] + np.random.normal(0, 0.1)
        else:
            y[i] = -X[i, 0] + 4 * X[i, 1] + np.random.normal(0, 0.1)

    max_depth = 5
    min_samples_split = 2

    model = GroupedRegressionTree(max_depth, min_samples_split, n_groups)
    model.fit(X, y, g)

    # Make predictions
    sample = np.array([0.5, 0.5])

    print("Model predictions")
    print(model.predict_single(sample))

    print("True values")
    print([
        5 * sample[0] + sample[1],
        sample[0] - 3 * sample[1],
        -sample[0] + 4 * sample[1]
    ])



if __name__ == "__main__":
    main()


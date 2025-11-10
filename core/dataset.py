import numpy as np


def make_dataset():
    """Generates a dataset of 50 3D points from three distinct Gaussian distributions."""
    np.random.seed(42)  # for reproducibility

    mean1 = [0, 0, 0]
    cov1 = [[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]]
    cluster1 = np.random.multivariate_normal(mean1, cov1, 15)

    mean2 = [3, 3, 3]
    cov2 = [[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]]
    cluster2 = np.random.multivariate_normal(mean2, cov2, 15)

    mean3 = [0, 4, 6]
    cov3 = [[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]]
    cluster3 = np.random.multivariate_normal(mean3, cov3, 20)

    dataset = np.vstack([cluster1, cluster2, cluster3])

    points = [tuple(int(round(c)) for c in point) for point in dataset]
    return points
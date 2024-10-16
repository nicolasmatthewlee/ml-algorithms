import numpy as np


class KNN:
    """
    An implementation of the K-Nearest Neighbors (KNN) algorithm for binary classification.

    Parameters:
    k (int): The number of nearest neighbors to consider.
    train_x (numpy.ndarray): The training dataset, where each row is a data point.
    train_y (list): The labels corresponding to each point in the training dataset.
    """

    def __init__(self, k, train_x, train_y):
        """
        Initialize the KNN classifier with the given training data.

        Parameters:
        k (int): The number of neighbors to consider for classification.
        train_x (numpy.ndarray): The training data points.
        train_y (list): The labels associated with the training data points.
        """
        self.k = k
        self.train_x = train_x
        self.train_y = train_y

    def predict(self, x):
        """
        Predict the label for a given data point using the K-Nearest Neighbors algorithm.

        Parameters:
        x (numpy.ndarray): The data point to classify.

        Returns:
        int: The predicted label (0 or 1).
        """
        # 1. Calculate the Euclidean distance between the input point and all training points
        distances = []
        for tx, ty in zip(self.train_x, self.train_y):
            distances.append((np.linalg.norm(tx - x), ty))

        # 2. Sort by distance and select the k nearest neighbors
        k_nearest = sorted(distances, key=lambda x: x[0])[: self.k]

        # 3. Predict the majority class among the k nearest neighbors
        if sum([d[1] for d in k_nearest]) > self.k / 2:
            return 1
        else:
            return 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1. generate dataset
    np.random.seed(1)
    x = np.random.rand(1000, 2)
    y = [abs(np.sin(e[0] * 2 * np.pi) / 4 + 0.5 - e[1]) > 0.2 for e in x]

    # 2. create KNN classifier and predict labels for new points
    knn = KNN(10, x, y)
    test_x = np.random.rand(50, 2)
    test_y = [knn.predict(x) for x in test_x]

    # 3. set up the figure and axis for plotting
    fig, ax = plt.subplots(tight_layout=True)
    fig.canvas.manager.set_window_title("k-nearest neighbors")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # 4. plot the original dataset and test points
    c = ["#d62728" if d == True else "#1f77b4" for d in y]
    ax.scatter(x[:, 0], x[:, 1], color=c, alpha=0.1, s=20)
    c = ["#d62728" if d == True else "#1f77b4" for d in test_y]
    ax.scatter(test_x[:, 0], test_x[:, 1], color=c, alpha=0.7, s=20)

    # 5. save figure
    plt.savefig(
        "assets/knn_binary_classification.png",
        format="png",
        dpi=300,
    )

    plt.show()

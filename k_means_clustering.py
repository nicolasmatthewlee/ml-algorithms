import numpy as np


def k_means_clustering(k, x, steps=100):
    """
    Perform K-Means clustering on a given dataset.

    Parameters:
    k (int): The number of clusters.
    x (numpy.ndarray): The dataset to cluster, where each row is a data point.
    steps (int, optional): The number of iterations to perform. Default is 100.

    Returns:
    list: A list containing the history of the clustering process. Each element is a tuple, where:
        - means (numpy.ndarray): The centroids of the clusters at that step.
        - clusters (list of numpy.ndarray): The points assigned to each cluster at that step.
    """
    # 1. set initial clusters and means
    clusters = np.array_split(x, k)
    means = np.array([np.mean(c, axis=0) for c in clusters])
    history = []

    # 2. perform iterations
    for _ in range(0, steps):
        # 2.1. reset clusters
        clusters = [[] for _ in range(k)]

        # 2.2. assign each point to the cluster with the closest centroid
        for e in x:
            l2 = np.sum((means - e) ** 2, axis=1)
            clusters[np.argmin(l2)].append(e)

        # 2.3. save clustering to history
        history.append((means, [np.array(c) for c in clusters]))

        # 2.4. calculate means for next iteration
        means = np.array([np.mean(c, axis=0) for c in clusters])

    return history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # 1. generate dataset
    np.random.seed(1)
    x = np.random.rand(1000, 2)

    # 2. perform clustering
    history = k_means_clustering(4, x, 30)

    # 3. Set up the figure and axis for plotting
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("k-means clustering")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    scatter_clusters = [
        ax.scatter([], [], s=20, alpha=0.5) for _ in range(len(history[0][1]))
    ]
    scatter_means = ax.scatter([], [], color="red", s=50, marker="+")

    # 4. define animation
    def update(frame):
        """
        Update the scatter plot for the current frame in the animation.

        Parameters:
        frame (int): The current frame number in the animation, corresponding to the iteration of K-Means.

        Returns:
        list: A list containing the updated scatter plot for the means and clusters.
            - scatter_means (matplotlib.collections.PathCollection): The updated scatter plot of the centroids.
            - scatter_clusters (list of matplotlib.collections.PathCollection): The updated scatter plots of the points in each cluster.
        """
        means, clusters = history[frame]
        scatter_means.set_offsets(means)

        for i, c in enumerate(clusters):
            if len(c) > 0:
                scatter_clusters[i].set_offsets(c)
            else:
                scatter_clusters[i].set_offsets([])
        return [scatter_means] + scatter_clusters

    animation = FuncAnimation(fig, update, frames=len(history), interval=100)

    # 5. save animation
    animation.save("assets/k_means_clustering.gif", writer="pillow", fps=10)

    plt.show()

import numpy as np


def multiple_linear_regression_ols(x, y):
    """
    Perform multiple linear regression using Ordinary Least Squares (OLS).

    Parameters:
    x (numpy.ndarray): The input features matrix (independent variables),
                       shape (n_samples, n_features).
    y (numpy.ndarray): The target vector (dependent variable), shape (n_samples,).

    Returns:
    numpy.ndarray: The coefficients of the regression model, shape (n_features + 1,).
                   The first coefficient is the intercept term.
    """
    # add column of ones for intercept
    ones = np.ones((x.shape[0], 1))
    x = np.hstack((ones, x))
    return np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)


if __name__ == "__main__":
    # 1. generate sample data
    np.random.seed(0)
    n_samples = 50
    x1 = np.random.uniform(50, 80, n_samples)
    x2 = np.random.uniform(10, 30, n_samples)
    noise = np.random.normal(0, 5, n_samples)
    y = 3.5 * x1 + 2 * x2 + 10 + noise
    x = np.column_stack((x1, x2))

    # 2. perform regression
    beta = multiple_linear_regression_ols(x, y)

    # 3. plot data and regression
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x[:, 0], x[:, 1], y, color="dodgerblue", label="actual")
    x1_surf, x2_surf = np.meshgrid(
        np.linspace(x[:, 0].min(), x[:, 0].max(), 10),
        np.linspace(x[:, 1].min(), x[:, 1].max(), 10),
    )
    y_surf = beta[0] + beta[1] * x1_surf + beta[2] * x2_surf
    ax.plot_surface(
        x1_surf,
        x2_surf,
        y_surf,
        color="red",
        alpha=0.5,
        label=f"y = {beta[0]:.2f} + {beta[1]:.2f}x_1 + {beta[2]:.2f}x_2",
    )
    ax.legend()
    plt.show()

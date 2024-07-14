import numpy as np


def simple_linear_regression_ols(data):
    """
    Perform simple linear regression using the Ordinary Least Squares (OLS) method.

    Parameters:
    data (numpy.ndarray): A 2D array where each row represents a data point and the columns
                          represent the variables x and y respectively.
                          data[:, 0] should be the independent variable (x)
                          data[:, 1] should be the dependent variable (y)

    Returns:
    tuple: A tuple containing:
        - intercept (float): The y-intercept of the regression line.
        - slope (float): The slope of the regression line.
    """
    # Calculate the covariance between x and y
    cov = np.cov(data, rowvar=False, bias=True)[0, 1]

    # Calculate the variance of x
    var = np.var(data[:, 0])

    # Calculate slope (quotient of covariance and variance)
    b1 = cov / var

    # Calculate the means of x and y
    x_bar = np.mean(data[:, 0])
    y_bar = np.mean(data[:, 1])

    # Calculate the intercept
    b0 = y_bar - b1 * x_bar

    return b0, b1


if __name__ == "__main__":
    # 1. generate sample data
    np.random.seed(0)
    x = np.linspace(0, 10, 50)
    y = 2.5 * x + np.random.normal(size=x.size)
    data = np.column_stack((x, y))

    # 2. perform regression
    b0, b1 = simple_linear_regression_ols(data)

    # 3. plot data and regression
    x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100)
    y = b0 + b1 * x
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], color="dodgerblue", label="actual", alpha=0.5)
    ax.plot(x, y, label=f"y = {b0:.2f} + {b1:.2f}x", color="red", alpha=0.5)
    ax.legend()
    plt.show()

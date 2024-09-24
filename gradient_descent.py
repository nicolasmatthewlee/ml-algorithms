import numpy as np


def gradient_descent(x_0, f_prime, alpha, max_steps=1000):
    """
    Perform gradient descent to minimize a function.

    Parameters:
    x_0 (float): Initial guess for the variable x.
    alpha (float): Learning rate, i.e., the step size.
    f_prime (function): The derivative of the function being minimized.
    max_steps (int, optional): Maximum number of iterations (steps). Default is 1000.

    Returns:
    numpy.ndarray: An array containing the history of x values through the iterations.
    """
    history = [x_0]
    x = x_0
    for _ in range(1, max_steps):
        history.append(x)
        x = x - alpha * f_prime(x)
    return np.array(history)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # 1. Define the function f(x) and its derivative f'(x)
    f = lambda x: x**2 + 8
    f_prime = lambda x: x * 2

    # 2. Perform gradient descent starting at x=10, with a learning rate of 0.1, for 30 steps
    results_x = gradient_descent(10, f_prime, alpha=0.1, max_steps=30)
    results_y = f(results_x)

    # 3. Generate the values for plotting the function f(x) = x^2 + 8
    f_x = np.arange(-10, 10, 0.1)
    f_y = list(map(f, f_x))

    # 4. Set up the figure and axis for plotting
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.plot(f_x, f_y, color="royalblue")
    scatter = ax.scatter(
        [], [], color="royalblue", s=50, alpha=0.8, linewidth=1, edgecolor="royalblue"
    )

    # 5. define animation
    def update(frame):
        """
        Update the scatter plot for each frame of the animation.

        Parameters:
        frame (int): The current frame number.

        Returns:
        scatter (matplotlib.collections.PathCollection): The updated scatter plot.
        """
        scatter.set_offsets(np.c_[results_x[:frame], results_y[:frame]])
        return scatter

    animation = FuncAnimation(fig, update, frames=len(results_x), interval=100)
    plt.show()

import numpy as np


def gradient_descent_2d(x_0, f_prime, alpha, max_steps=1000):
    """
    Perform gradient descent to minimize a function of a single variable.

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


def gradient_descent_3d(x_0, gradient, alpha, max_steps=1000):
    """
    Perform gradient descent to minimize a function.

    Parameters:
    x_0 ((float, float)): Initial guess for the variable x.
    alpha (float): Learning rate, i.e., the step size.
    gradient (function): The gradient of the function being minimized.
    max_steps (int, optional): Maximum number of iterations (steps). Default is 1000.

    Returns:
    numpy.ndarray: An array containing the history of x values through the iterations.
    """
    history = [x_0]
    x = x_0
    for _ in range(1, max_steps):
        history.append(x)
        x = np.array(x) - alpha * gradient(x)

    return np.array(history)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from scipy.optimize import rosen, rosen_der

    # 1. Perform gradient descent
    plot_max = 5
    n = 100
    max_steps = 60
    results_x = [
        gradient_descent_3d(
            (x[0] * plot_max, x[1] * plot_max),
            rosen_der,
            alpha=0.00001,
            max_steps=max_steps,
        )
        for x in np.random.uniform(-1, 1, (n, 2))
    ]
    results_y = [[rosen(x) for x in res_x] for res_x in results_x]

    # 2. Set up the figure and axis for plotting
    fig = plt.figure(tight_layout=True)
    fig.canvas.manager.set_window_title("gradient descent")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.xaxis.line.set_lw(0.0)
    ax.yaxis.line.set_lw(0.0)
    ax.zaxis.line.set_lw(0.0)

    # 3. plot surface
    x = np.linspace(-(plot_max), plot_max, 400)
    y = np.linspace(-(plot_max), plot_max, 400)
    X, Y = np.meshgrid(x, y)
    Z = rosen((X, Y))
    surface = ax.plot_surface(
        X, Y, Z, cmap="spring", edgecolor="none", alpha=0.5, zorder=1
    )
    scatters = [
        ax.scatter(
            [], [], [], color="dodgerblue", s=30, linewidth=1, zorder=10, alpha=1
        )
        for _ in results_x
    ]

    # 4. define animation
    def update(frame):
        """
        Update the scatter plot for each frame of the animation.

        Parameters:
        frame (int): The current frame number.

        Returns:
        scatter (matplotlib.collections.PathCollection): The updated scatter plot.
        """
        for i, scatter in enumerate(scatters):
            scatter._offsets3d = (
                [results_x[i][:, 0][frame]],
                [results_x[i][:, 1][frame]],
                [results_y[i][frame]],
            )
        return scatters

    animation = FuncAnimation(fig, update, frames=len(results_x[0]), interval=100)

    # 5. save animation
    animation.save("assets/gradient_descent3d_multiple.gif", writer="pillow", fps=10)

    plt.show()

# ml-algorithms

## simple linear regression with ordinary least squares

Consider the following model of a perfect linear relationship between $x_i$ and $y_i$ where all variability in $y_i$ is explained by $x_i$

$y_i=\alpha+\beta{x_i}$

Now consider the following model of an imperfect linear relationship between $x_i$ and $y_i$ where some of the variability in $y_i$ is explained by $x_i$, and the remaining variability in $y_i$ is explained by $\epsilon_i$

$y_i=\alpha+\beta{x_i}+\epsilon_i$

Assuming the relationship between $x_i$ and $y_i$ is perfectly linear, and that the error $\epsilon_i$ is i.i.d. with mean zero and constant variance, the residual $y_i-(\alpha+\beta{x_i})$ will follow the distribution of $\epsilon_i$ independent of $x_i$

Let $\{(x_i,y_i),i=1,...,n\}$ be the collection of observations

Let $\hat{y}_1,...\hat{y}_n$ be the collection of predictions

Under ordinary least squares the objective function $Q$ is defined as follows:

$$
\begin{align}
Q(\alpha,\beta)&=\sum_{i=1}^n{(y_i-\hat{y})^2} \\
&=\sum_{i=1}^n{(y_i-\alpha-\beta{x_i})^2}
\end{align}
$$

Given that $Q$ is a strictly convex function, $Q$ has exactly one critical point where the gradient is zero, which is also the global minimum.

To find the $(\alpha,\beta)$ that minimizes $Q$, solve $\nabla{Q}=0$ by evaluating the following system of equations:

$$
\begin{align}
\frac{\partial{Q}}{\partial{\alpha}}&=-2\sum_{i=1}^n{\left(y_i-\alpha-\beta{x_i}\right)}=0 \\
\frac{\partial{Q}}{\partial{\beta}}&=-2x_i\sum_{i=1}^n{\left(y_i-\alpha-\beta{x_i}\right)}=0
\end{align}
$$

which simplify to the following:

$$
\begin{align}
\alpha &= \bar{y}- \frac{\text{Cov(x,y)}}{\text{Var(x)}}\bar{x} \\
\beta &= \frac{\text{Cov}(x,y)}{\text{Var}(x)}
\end{align}
$$

## multiple linear regression with ordinary least squares

Given a dataset with $n$ observations and $p$ features, let:

- $y$ be an $n \times 1$ vector of target values
- $X$ be an $n \times (p+1)$ matrix (with a column of ones for the intercept)
- $\beta$ be a $(p+1) \times 1$ vector of coefficients
- $\epsilon$ be an $n \times 1$ vector of residuals

$$
\begin{align}
\left[
\begin{matrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{matrix}\right]
&=
\left[\begin{matrix}
1 & x_{11} & x_{12} & \dots & x_{1p} \\
1 & x_{21} & x_{22} & \dots & x_{2p} \\
\vdots & \ddots & \ddots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \dots & x_{np}
\end{matrix}\right]
\cdot
\left[\begin{matrix}
\beta_0 \\
\beta_1 \\
\vdots \\
\beta_p
\end{matrix}\right]
+
\left[\begin{matrix}
\epsilon_1 \\
\epsilon_2 \\
\vdots \\
\epsilon_n
\end{matrix}\right]
\end{align}
$$

which can be written as follows:
$$y=X\beta+\epsilon$$

Under ordinary least squares the objective function $Q$ is defined as follows:

$$
\begin{align}
Q(\alpha,\beta)&=\sum_{i=1}^n{(y_i-\hat{y})^2} \\
&=\sum_{i=1}^n{(y_i-\alpha-\beta{x_i})^2} \\
&=\sum_{i=1}^n{e_i^2} \\
&=\epsilon^T\epsilon \\
\end{align}
$$

The global minimum of $Q$ (with some assumptions about $X$) is found by first computing the gradient of $Q$ with respect to $\beta$:

$$
\begin{align}
\nabla_\beta{Q}&=\frac{\partial{Q}}{\partial{\beta}} \\
&=\frac{\partial{\epsilon^T\epsilon}}{\partial{\beta}} \\
&= \frac{\partial{(y-X\beta)^T(y-X\beta)}}{\partial{\beta}} \\
&= \frac{\partial{(y^T-\beta^TX^T)(y-X\beta)}}{\partial{\beta}} \\
&= \frac{\partial{(y^Ty-y^TX\beta-\beta^TX^Ty+\beta^TX^TX\beta)}}{\partial{\beta}} \\
&= \frac{\partial{(y^Ty-2y^TX\beta+\beta^TX^TX\beta)}}{\partial{\beta}} \\
&= -2X^Ty+2X^TX\beta \\
\end{align}
$$

Note: the simplification occurring at equation $(16)$ follows from the fact that $y^TX\beta$ and $\beta^TX^Ty$ are $1\times1$ matrices, the transpose of any $1\times1$ matrix is itself, and $(y^TX\beta)^T=\beta^TX^Ty$

And then setting the gradient equal to zero:

$$
\begin{align}
0 &= -2X^Ty+2X^TX\beta \\
X^TX\beta &= X^Ty \\
\beta &= (X^TX)^{-1}X^Ty
\end{align}
$$

## multiple linear regression with stochastic gradient descent

resume `multiple_linear_regression_ols.py` implementation

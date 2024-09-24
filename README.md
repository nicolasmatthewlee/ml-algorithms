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

## gradient descent

Let $f:\mathbb{R}\to\mathbb{R}$ be a twice-differentiable, strongly convex function with a Lipschitz continuous derivative.

A Lipschitz continuous derivative implies the following:
$$\exists_{L>0}\forall_{x_1,x_2\in\mathbb{R}}|f'(x_1)-f'(x_2)|\le L|x_1-x_2|\land \forall_{x\in\mathbb{R}}f''(x)\le L$$

Let $\alpha$ be a constant satisfying the following:
$$0<\alpha<\frac{1}{L}$$

The gradient descent algorithm is defined as follows:
$$x_{k+1}=x_k-\alpha f'(x_k)$$

Let $x^*$ be the unique minimum of $f$.

Let the error at step $k$, $e_k$ be defined as follows:
$$e_k=x_k-x^*$$

The error at step $k+1$ can be written as follows:

$$
\begin{align}
e_{k+1}&=x_{k+1}-x^* \\
&=x_k-\alpha f'(x_k)-x^* \\
&=x_k-x^*-\alpha f'(x_k) \\
&=e_k-\alpha f'(x_k)
\end{align}
$$

Strong convexity implies the following:
$$\exists_{\mu>0}\forall_{x\in\mathbb{R}}f''(x)\ge\mu$$

Which implies that $f'(x)$ is strictly increasing. Apply the Mean Value Theorem to $f'(x)$ as follows:
$$\exists_{c \in [x_k, x^{*}]}\frac{f'(x_k) - f'(x^*)}{x_k - x^*} = f''(c)$$

Given that $\forall_{x\in\mathbb{R}}f''(x)\ge\mu$ and $f'(x^*)=0$,

$$
\begin{align}
\frac{f'(x_k)-f'(x^*)}{x_k-x^*}& \ge \mu \\
f'(x_k)-f'(x^*)& \ge \mu(x_k-x^*) \\
f'(x_k)& \ge \mu e_k \\
\end{align}
$$

Note that $\forall_{k}$, $f'(x_k)$ and $e_k$ have the same sign. Applying the defintion of Lipschitz continuity between $x_k$ and $x^*$ gives the following:

$$
\begin{align}
\left| f'(x_k) - f'(x^*) \right| &\leq L \left| x_k - x^* \right| \\
\left| f'(x_k) \right| &\leq L \left| e_k \right| \\
f'(x_k) &\leq L e_k
\end{align}
$$

Recall the following result derived earlier:
$$e_{k+1} = e_k-\alpha f'(x_k)$$

Now observe the following bounds on $e_{k+1}$:

$$
\begin{align}
e_{k+1}&\le e_k-\alpha \mu e_k \\
&= e_k(1-\alpha\mu)\\
e_{k+1} &\ge e_k-\alpha Le_k \\
&= e_k(1-\alpha L) \\
&\therefore \\
e_k(1-\alpha L) &\le e_{k+1} \le e_k(1-\alpha\mu) \\
\end{align}
$$

Recall the following constraints and note the bounds that follow:

$$
\begin{align}
L &>0 \\
0&<\alpha<\frac{1}{L} \\
0&<\mu\le f''(x)\le L \\
\therefore \\
\alpha L &< \frac{1}{L}L = 1 \\
\alpha\mu &< \frac{1}{L} = 1 \\
\end{align}
$$

Therefore, $e_{k+1}$ is bounded within positive scalar multiples of $e_k$. Now observe the following recurrence relation:

$$
\begin{align}
e_{k+1} &\le e_k(1-\alpha\mu) \\
e_{k} &\le e_{k-1}(1-\alpha\mu) \\
&\therefore \\
e_{k+1} &\le e_{k-1}(1-\alpha\mu)^2 \\
\end{align}
$$

Which can be simplified to the following:

$$
\begin{align}
e_{k} &\le e_{0}(1-\alpha\mu)^{k} \\
\end{align}
$$

Thereby demonstrating the convergence $e_k\to0$, or $x_k\to x^*$, as $k\to \infty$

---

- gradient descent in multiple dimensions
- multiple linear regression with stochastic gradient descent
- k-means clustering

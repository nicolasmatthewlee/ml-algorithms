# ml-algorithms

## linear regression with ordinary least squares

Consider the following model of a perfect linear relationship between $x_i$ and $y_i$ where all variability in $y_i$ is explained by $x_i$

$y_i=\alpha+\beta{x_i}$

Now consider the following model of an imperfect linear relationship between $x_i$ and $y_i$ where some of the variability in $y_i$ is explained by $x_i$, and the remaining variability in $y_i$ is explained by $\epsilon_i$

$y_i=\alpha+\beta{x_i}+\epsilon_i$

Assuming the relationship between $x_i$ and $y_i$ is perfectly linear, and that the error $\epsilon_i$ is i.i.d. with mean zero and constant variance, the observed data will scatter around the regression line $\alpha+\beta{x_i}$

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
\Beta &= \frac{\text{Cov}(x,y)}{\text{Var}(x)}
\end{align}
$$

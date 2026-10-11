"""Logistic regression with NumPy (Task 3)."""
import numpy as np


def add_intercept(X):
    """Prepend the column x0 = 1."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def sigmoid(z):
    """g(z) = 1 / (1 + e^-z). Clipping z only prevents overflow in exp."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def hypothesis(X, theta):
    """h_theta(x) = g(theta^T x) for every row of X."""
    return sigmoid(X @ theta)


def cost(X, y, theta):
    """J(theta) = -(1/n) * sum[ y log h + (1 - y) log(1 - h) ]."""
    h = np.clip(hypothesis(X, theta), 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))


def fit_logistic(X, y, alpha, n_iters):
    """Batch gradient descent from theta = 0. Returns theta and J after every iteration."""
    n, d = X.shape
    theta = np.zeros(d)
    history = np.zeros(n_iters)
    for t in range(n_iters):
        h = hypothesis(X, theta)
        theta = theta + (alpha / n) * (X.T @ (y - h))
        history[t] = cost(X, y, theta)
    return theta, history


def predict(X, theta):
    """1 if h_theta(x) >= 0.5, else 0."""
    return (hypothesis(X, theta) >= 0.5).astype(int)

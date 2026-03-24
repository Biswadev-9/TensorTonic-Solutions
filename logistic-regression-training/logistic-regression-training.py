import numpy as np

def train_logistic_regression(X, y, lr=0.1, steps=500):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(steps):
        # prediction
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))   # sigmoid

        # gradients
        error = p - y
        w = w - lr * (X.T @ error) / n
        b = b - lr * np.mean(error)

    return w, b
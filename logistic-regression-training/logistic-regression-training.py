import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    m,n = X.shape
    w = np.zeros(n)
    b = 0.0

    for i in range(steps):
        z = X @ w + b
        y_pred = _sigmoid(z)

        dw = (X.T @ (y_pred-y))/m
        db = np.sum(y_pred - y)/m

        w-= lr*dw
        b-= lr*db

    return w,b
    pass
import numpy as np

class Dense:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim)
        self.b = np.zeros(out_dim)

    def forward(self, x):
        self.x = x
        return x @ self.W +self.b

    def backward(self, dout, learning_rate):
        dW = self.x.T @ dout
        db = np.sum(dout, axis=0, keepdims=True)
        dx = dout @ self.W.T

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dx


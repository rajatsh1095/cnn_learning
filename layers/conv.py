import numpy as np

class Conv2D():
    def __init__(self, in_channels, out_channels, kernal_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.kernal_size = kernal_size

        self.weights = np.random.randn(out_channels, in_channels, kernal_size, kernal_size)*0.01

        self.bias = np.zeros((out_channels, 1))

    def forward(self, x):
        self.x = x

        N, C, H, W = x.shape
        F, _, HH, WW = self.weights.shape
        H_out = (H + 2*self.padding - HH)//self.stride + 1
        W_out = (W + 2*self.padding - WW)//self.stride + 1

        out = np.zeros((N , F, H_out, W_out))

        x_padded = np.pad(x, ((0,0),(0,0),(self.padding, self.padding),(self.padding, self.padding)), mode='constant')

        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i*self.stride
                        w_start = j*self.stride

                        region = x_padded[n, : , h_start: h_start + HH, w_start: w_start + HH]

                        out[n, f, i ,j] = (np.sum(region * self.weights[f]) + self.bias[f])

        return out

    def backward(self, dout, learning_rate):
        N, C, H, W = self.x.shape
        F, _, HH, WW = self.weights.shape
        _, _, H_out, W_out = dout.shape
        dx = np.zeros_like(self.x)
        dW = np.zeros_like(self.weights)
        db = np.zeros_like(self.bias)
        x_padded = np.pad(self.x,((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),mode='constant')
        dx_padded = np.pad(dx, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),mode='constant')

        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        region = x_padded[n, :, h_start:h_start + HH, w_start:w_start + WW]
                        # upstream gradient
                        grad = dout[n, f, i, j]
                        # filter gradients
                        dW[f] += region * grad
                        # bias gradients
                        db[f] += grad
                        # input gradients
                        dx_padded[n, :, h_start:h_start + HH, w_start:w_start + WW] += self.weights[f] * grad

        if self.padding > 0:
            dx = dx_padded[ :, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded

        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db
        return dx

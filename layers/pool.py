import numpy as np

class MaxPool2D:
    def __init__(self, size = 2, stride = 2):
        self.size = size
        self.stride = stride

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape

        H_out = H // self.size
        W_out = W // self.size

        out = np.zeros((N, C, H_out, W_out))

        # store max locations
        self.max_indices = {}

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        region = x[n, c, i*self.size: (i+1)*self.size, j*self.size: (j+1)*self.size]
                        out[n, c, i, j] = np.max(region)

                        # location of max inside region
                        max_pos = np.unravel_index(np.argmax(region), region.shape)
                        self.max_indices[(n, c, i, j)] = (h_start + max_pos[0], w_start + max_pos[1])

            return out

    def backward(self, dout):

        N, C, H_out, W_out = dout.shape
        dx = np.zeros_like(self.x)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_max, w_max = self.max_indices[(n, c, i, j)]
                        dx[n, c, h_max, w_max] = dout[n, c, i, j]

        return dx
import numpy as np

class SoftmaxCrossEntropy:
    def forward(self, logits, y):
        logits = logits - np.max(logits, axis =1, keepdims=True)
        exp = np.exp(logits)

        self.probs = exp / np.sum(exp, axis =1, keepdims=True)

        loss = -np.mean(np.log(probs[np.arange(len(y)), y]))

        return loss

    def backward(self):
        batch_size = len(self.y)
        dx = self.probs.copy()
        dx[np.arange(batch_size), self.y] -= 1
        dx /= batch_size

        return dx
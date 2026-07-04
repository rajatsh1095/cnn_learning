from layers.conv import Conv2D
from layers.relu import ReLU
from layers.pool import MaxPool2D
from layers.dense import Dense
from layers.losses import SoftmaxCrossEntropy

class SimpleCNN:
    def __init__(self, num_classes):
        self.conv1 = Conv2D(3, 8, 3, padding = 1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D()

        self.conv2 = Conv2D(8, 16, 3, padding = 1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D()

        self.fc = Dense(16*56*56, num_classes)

        self.loss_fn = SoftmaxCrossEntropy()


    def forward(self, x, y):
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)

        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)

        self.before_flatten_shape = out.shape

        out = out.reshape(out.shape[0], -1)

        logits = self.fc.forward(out)

        loss = self.loss_fn.forward(logits, y)

        return loss

    def backward(self, learning_rate):
        dout = self.loss_fn.backward()
        dout = self.fc.backward(dout, learning_rate)
        dout = dout.reshape(self.before_flatten_shape)

        dout = self.relu2.backward(dout)
        dout = self.pool2.backward(dout)
        dout = self.conv2.backward(dout, learning_rate)

        dout = self.relu1.backward(dout)
        dout = self.pool1.backward(dout)
        dout = self.conv1.backward(dout, learning_rate)

        return dout
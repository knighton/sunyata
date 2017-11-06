import torch
from torch.autograd import Variable


class Layer(object):
    pass


class Dense(Layer):
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, x):
        return x.mm(self.kernel)


class ReLU(Layer):
    def forward(self, x):
        return x.clamp(min=0)


class Sequence(Layer):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


dtype = torch.FloatTensor

batch_size, in_dim, hidden_dim, num_classes = 64, 1000, 100, 10

x = Variable(torch.randn(batch_size, in_dim).type(dtype), requires_grad=False)
y = Variable(torch.randn(batch_size, num_classes).type(dtype),
             requires_grad=False)

w1 = Variable(torch.randn(in_dim, hidden_dim).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(hidden_dim, num_classes).type(dtype),
              requires_grad=True)

model = Sequence([
    Dense(w1),
    ReLU(),
    Dense(w2),
])

learning_rate = 1e-6
for t in range(500):
    y_pred = model.forward(x)

    loss = (y_pred - y).pow(2).sum()
    print(t, loss.data[0])

    loss.backward()

    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()

import numpy as np

class Module:
    def __init__(self):
        self.training = True

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        return self.forward(*args)

class Linear(Module):
    def __init__(self, in_features, out_features, weight_init='he'):
        super().__init__()
        if weight_init == 'he':
            self.W = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)
        elif weight_init == 'xavier':
            self.W = np.random.randn(in_features, out_features) * np.sqrt(1. / in_features)
        else:
            self.W = np.random.randn(in_features, out_features) * 0.01
            
        self.b = np.zeros((1, out_features))
        
        self.dW = None
        self.db = None
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b

    def backward(self, dY):
        # dY: (N, out_features)
        # self.X: (N, in_features)
        self.dW = np.dot(self.X.T, dY)
        self.db = np.sum(dY, axis=0, keepdims=True)
        dX = np.dot(dY, self.W.T)
        return dX

class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, X):
        self.mask = (X > 0)
        return X * self.mask

    def backward(self, dY):
        return dY * self.mask

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, X):
        self.out = 1 / (1 + np.exp(-np.clip(X, -500, 500)))
        return self.out

    def backward(self, dY):
        return dY * self.out * (1 - self.out)

class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, X):
        self.out = np.tanh(X)
        return self.out

    def backward(self, dY):
        return dY * (1 - self.out**2)

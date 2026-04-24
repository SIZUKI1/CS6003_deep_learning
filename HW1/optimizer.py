class SGD:
    def __init__(self, model, lr=0.01, weight_decay=0.0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        params = self.model.get_params()
        for p in params:
            # W = W - lr * (dW + weight_decay * W)
            p['W'] -= self.lr * (p['dW'] + self.weight_decay * p['W'])
            # b = b - lr * db (bias usually not decayed)
            p['b'] -= self.lr * p['db']

class StepLR:
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step % self.step_size == 0:
            self.optimizer.lr *= self.gamma

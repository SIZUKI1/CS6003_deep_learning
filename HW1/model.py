import numpy as np
from layers import Linear, ReLU, Sigmoid, Tanh

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu'):
        # 3-layer MLP: Input -> H1 -> H2 -> Output
        # (Alternatively, if 3-layer means Input -> Hidden -> Output, it would be 2 weight layers)
        # We'll use 2 hidden layers to be safe for "3-layer" (3 weight transitions)
        
        self.layers = []
        
        # Layer 1
        self.layers.append(Linear(input_dim, hidden_dim, weight_init='he' if activation == 'relu' else 'xavier'))
        self.layers.append(self._get_activation(activation))
        
        # Layer 2
        self.layers.append(Linear(hidden_dim, hidden_dim, weight_init='he' if activation == 'relu' else 'xavier'))
        self.layers.append(self._get_activation(activation))
        
        # Layer 3 (Output)
        self.layers.append(Linear(hidden_dim, output_dim, weight_init='xavier'))
        
    def _get_activation(self, name):
        if name.lower() == 'relu':
            return ReLU()
        elif name.lower() == 'sigmoid':
            return Sigmoid()
        elif name.lower() == 'tanh':
            return Tanh()
        else:
            raise ValueError(f"Unknown activation: {name}")

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_params(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                params.append({'W': layer.W, 'b': layer.b, 'dW': layer.dW, 'db': layer.db})
        return params

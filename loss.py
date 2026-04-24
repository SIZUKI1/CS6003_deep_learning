import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.labels = None

    def __call__(self, logits, labels):
        # logits: (N, C)
        # labels: (N,) - class indices
        N = logits.shape[0]
        
        # Softmax for stability
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        self.labels = labels
        
        # Cross entropy loss
        correct_logprobs = -np.log(self.probs[range(N), labels] + 1e-12)
        loss = np.sum(correct_logprobs) / N
        return loss

    def backward(self):
        N = self.probs.shape[0]
        dZ = self.probs.copy()
        dZ[range(N), self.labels] -= 1
        dZ /= N
        return dZ

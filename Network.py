import numpy as np

class Rede(object):

    def __init__(self, sizes):
        self.num_camadas = len(sizes)
        self.sizes = sizes
        self.vies = [np.random.randn(y, 1) for y in sizes[1:]]
        self.pesos = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for v, p in zip(self.vies, self.pesos):
            a = sigmoid(np.dot(p, a) + v)
        return a
    

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

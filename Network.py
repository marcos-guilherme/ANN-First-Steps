import numpy as np
import random

class Rede(object):

    def __init__(self, sizes):
        self.num_camadas = len(sizes)
        self.sizes = sizes

        #Aqui é criada uma coluna de valores aleatórios para vieses,
        #para cada camada da rede, exceto a primeira.
        self.vies = [np.random.randn(y, 1) for y in sizes[1:]]

        #Aqui são criadas matrizes coluna para cada camada, representando os pesos
        #que ligam os neurônios entre as camadas
        self.pesos = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    #Essa função é responsável por receber um vetor de ativação (vetor com os valores de ativação de neurônios)
    #a partir deste vetor de ativação calculamos, com base nos pesos da camada atual, o novo vetor de ativação
    #que será passado para a próxima camada, lembrando que a função de ativação usada é a sigmoide.
    def feedforward(self, a):
        for v, p in zip(self.vies, self.pesos):
            a = sigmoid(np.dot(p, a) + v)
        return a
    
    #É usada uma variante do método de minimizar a função de custo, onde 
    #selecionamos mini-lotes aleatórios do conjunto de treinamento para ajustar os parâmetros
    #da rede, ao invés de passar por todos os dados de treinamento.
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):

        test_data = list(test_data)
        training_data = list(training_data)
        if test_data: n_test = len(test_data)
        n = len(training_data)
        
        
    #Para cada epoch, escolhemos o conjunto de mini-lotes,
    #e então ajustamos os parâmetros da rede em cada passagem por mini-lote.
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
    
    def update_mini_batch(self, mini_batch, eta):
        nabla_p = [np.zeros(p.shape) for p in self.pesos]
        nabla_v = [np.zeros(v.shape) for v in self.vies]

        for x, y in mini_batch:
            delta_nabla_v, delta_nabla_p = self.backprop(x, y)
            nabla_v = [nv + dnv for nv, dnv in zip(nabla_v, delta_nabla_v)]
            nabla_p = [np + dnp for np, dnp in zip(nabla_p, delta_nabla_p)]
        self.pesos = [p - (eta/len(mini_batch))*np for p, np in zip(self.pesos, nabla_p)]

        self.vies = [v - (eta/len(mini_batch))*nv for v, nv in zip(self.vies, nabla_v)]


    def backprop(self, x, y):

        nabla_p = [np.zeros(p.shape) for p in self.pesos]
        nabla_v = [np.zeros(v.shape) for v in self.vies]


        #feedforward
        activation = x

        activations = [x]

        zs = []

        for v, p in zip(self.vies, self.pesos):
        
            z = np.dot(p, activation) + v
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_v[-1] = delta
        nabla_p[-1] = np.dot(delta, activations[-2].transpose())


        for l in range(2, self.num_camadas):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.pesos[-l + 1].transpose(), delta) * sp
            nabla_v[-l] = delta
            nabla_p[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_v, nabla_p)
    
    def evaluate(self, test_data):

        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        return(output_activations - y)


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

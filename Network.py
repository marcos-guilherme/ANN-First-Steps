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

        

        
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


net = Rede([1, 3, 3, 2])

print(net.pesos)
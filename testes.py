from Network import Rede
from mnist_loader import load_data_wrapper

training_data, valid_data, test_data = load_data_wrapper()

net = Rede([784, 30, 10])

net.SGD(training_data=training_data, epochs=50, mini_batch_size=10, eta=3, test_data=test_data)
import numpy as np
import network as net
import mnist_loader as load

training_data, validation_data, test_data = load.load_data_wrapper()

net_1 = net.Network([784, 30, 10])  # 28x28 image, 10 output digits
net_1.SGD(training_data, 30, 10, 3)

number_successes = net_1.evaluate(test_data)
percentage_success = number_successes/len(test_data)
print(percentage_success)

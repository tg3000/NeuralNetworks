from matplotlib import pyplot
from nn import MLP
import sys
from engine import Value
import numpy as np

# Your Path of mnist.npz here
path = "training_data/Mnist/mnist.npz"

# Load dataset
with np.load(path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

x_train = x_train / 255
x_test = x_test / 255

y_train_update = np.zeros((y_train.shape[0], 10))
y_test_update = np.zeros((y_test.shape[0], 10))

for i in range(y_train.shape[0]):
    y_train_update[i][y_train[i]] = 1
for i in range(y_test.shape[0]):
    y_test_update[i][y_test[i]] = 1
y_train = y_train_update
y_test = y_test_update


def get_loss(inputs, labels):
    out_of_nn = myNN(inputs)
    loss = Value.cross_entropy_loss(labels, out_of_nn)
    return loss


# Create Neural Network
myNN = MLP(np.array([784, 30, 10]), [False, True, False])

# Inputs
batchSize = 100
alpha = 0.1

# Time for training
for i in range(x_train.shape[0]):
    sys.stdout.write("\r" + str((i / x_train.shape[0]) * 100) + "% done")
    sys.stdout.flush()
    '''
    for j in range(100):
        currIndex = j + (100 * i)
        out += get_loss(x_train[currIndex], y_train[currIndex])
    '''
    currIndex = i
    out = get_loss(x_train[currIndex], y_train[currIndex])
    out.grad = 1
    out.backward()
    myNN.learn(alpha)
    myNN.zero_grad()
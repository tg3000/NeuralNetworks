from matplotlib import pyplot
from nn import MLP
from engine import Value
import numpy as np

'''
# Your Path of mnist.npz here
path = "training_data/Mnist/mnist.npz"

# Load dataset
with np.load(path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

# Print shapes
print('X_train: ' + str(x_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  '  + str(x_test.shape))
print('Y_test:  '  + str(y_test.shape))

# Plot mnist dataset
for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()
'''


def get_loss(inputs, labels):
    out_of_nn = myNN(inputs)
    loss = Value.cross_entropy_loss(labels, out_of_nn)
    loss.grad = 1
    loss.backward()
    return loss

# Create Neural Network
myNN = MLP(np.array([1, 4, 3]), [False, True, False])

# Inputs
inputs = np.array([1])
labels = np.array([1, 0, 0])
alpha = 0.1

# Draw loss before training
out = get_loss(inputs, labels)
out.draw("before")
myNN.zero_grad()

# Time for training
epochs = 1000
for i in range(epochs):
    out = get_loss(inputs, labels)
    myNN.learn(alpha)
    myNN.zero_grad()

# Draw loss after training
out = get_loss(inputs, labels)
out.draw("after")

print(myNN)

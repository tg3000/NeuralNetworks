from matplotlib import pyplot
from nn import MLP
import sys
from engine import Value
import numpy as np
import signal

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


# Train Model
def update_current_input_vals(current: np.array, next_arr: np.array):
    for i, j in zip(current.flatten(), next_arr.flatten()):
        i.data = j

def save_mlp(signum, frame):
    myNN.save_to_file("FirstMlp")
    quit()


# Create Neural Network
#myNN = MLP(np.array([784, 30, 10]), [False, True, False])
myNN = MLP.load_from_file("FirstMlp")

# Inputs
batchSize = 100
alpha = 0.1

x_train_vals = np.array([Value(0) for i in range(784)])
y_train_vals = np.array([Value(0) for i in range(10)])

out = Value.cross_entropy_loss(y_train_vals, myNN(x_train_vals))


signal.signal(signal.SIGTERM, save_mlp)
signal.signal(signal.SIGINT, save_mlp)
# Time for training
while myNN.current_training_step < x_train.shape[0]:
    percentage = (myNN.current_training_step / x_train.shape[0]) * 100
    sys.stdout.write("\r %.4f done" % percentage)
    sys.stdout.flush()
    
    sys.stdout.write("   step: " + str(myNN.current_training_step))
    sys.stdout.write("   loss: " + str(out.data))
    update_current_input_vals(x_train_vals, x_train[myNN.current_training_step])
    update_current_input_vals(y_train_vals, y_train[myNN.current_training_step])
    out.recalc()
    out.zero_grad()
    out.grad = 1
    out.backward()
    myNN.learn(alpha)
    myNN.current_training_step += 1
myNN.save_to_file("FirstMlp")

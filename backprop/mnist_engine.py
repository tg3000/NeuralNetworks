from backprop import mnist_loader
from nn import MLP
import sys
from engine import Value
import numpy as np
import signal


def learn(my_nn: MLP, x_train: np.array, y_train: np.array, name: str, batch_size: int, alpha: float):
    # Train Model
    def update_current_input_vals(current: np.array, next_arr: np.array):
        for i, j in zip(current.flatten(), next_arr.flatten()):
            i.data = j

    def save_mlp(signum, frame):
        my_nn.save_to_file(name)
        quit()

    x_train_vals = np.array([[Value(0) for i in range(784)] for i in range(batch_size)])
    y_train_vals = np.array([[Value(0) for i in range(10)] for i in range(batch_size)])

    out = Value.cross_entropy_loss(y_train_vals[0], my_nn(x_train_vals[0].tolist()))
    for i in range(1, batch_size):
        out += Value.cross_entropy_loss(y_train_vals[i], my_nn(x_train_vals[i].tolist()))

    signal.signal(signal.SIGTERM, save_mlp)
    signal.signal(signal.SIGINT, save_mlp)

    # Time for training
    while my_nn.current_training_step * batch_size < x_train.shape[0]:
        # Print
        percentage = (my_nn.current_training_step * batch_size / x_train.shape[0]) * 100

        sys.stdout.write("\r %.4f done" % percentage)
        sys.stdout.flush()

        sys.stdout.write("   batch: " + str(my_nn.current_training_step))
        sys.stdout.write("   avg. loss: " + str(out.data))

        update_current_input_vals(x_train_vals,
                                  x_train[my_nn.current_training_step * batch_size:
                                          my_nn.current_training_step * batch_size + batch_size + 1])
        update_current_input_vals(y_train_vals,
                                  y_train[my_nn.current_training_step * batch_size:
                                          my_nn.current_training_step * batch_size + batch_size])
        out.recalc()
        out.zero_grad()
        out.grad = 1
        out.backward()
        my_nn.apply_grad(alpha)
        my_nn.current_training_step += 1

    my_nn.save_to_file(name)

path = "training_data/Mnist/mnist.npz"

loader = mnist_loader.DataLoader()
loader.load(path)

name = "RealTry"

my_nn = MLP(1, [784, 30, 10], [False, True, False])
#my_nn = MLP.load_from_file(name)

batch_size = 32
alpha = 0.1

learn(my_nn, loader.get_xtrain(), loader.get_ytrain(), name, batch_size, alpha)

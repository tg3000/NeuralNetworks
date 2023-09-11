import numpy as np


class DataLoader:
    def __init__(self):
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []

    def get_xtrain(self):
        return self.x_train

    def get_ytrain(self):
        return self.y_train

    def get_xtest(self):
        return self.x_test

    def get_ytest(self):
        return self.y_test

    def load(self, path: str):
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

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

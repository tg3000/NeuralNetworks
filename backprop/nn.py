import pickle
import random
import numpy as np
from engine import Value


class Neuron:
    def __init__(self, inputs: int, neuronindex: int, layerindex: int, use_tanh):
        self.li = layerindex
        self.ni = neuronindex
        self.inputs = inputs
        self.use_tanh = use_tanh

        # Initialize Weights and Biases
        self.weights = [Value(random.uniform(-1, 1), label=f"w[{str(self.li)}][{str(self.ni)}][{str(i)}]") for i in range(inputs)]
        self.bias = Value(0, label=f"b[{str(self.li)}][{str(self.ni)}]")

    def __call__(self, x):

        out = sum([wi * xi for wi, xi in zip(self.weights, x)], self.bias)
        if self.use_tanh:
            out = out.tanh()
        return out

    def apply_grad(self, alpha):
        for w in self.weights:
            w.data += -alpha * w.grad
        self.bias.data += -alpha * self.bias.grad

    def zero_grad(self):
        for w in self.weights:
            w.grad = 0
        self.bias.grad = 0

    def __params(self):
        s = ""
        for i in range(self.weights.size):
            s += "w[%d]: %.04f; " % (i, self.weights[i].data)
        s += "b: %0.4f; " % self.bias.data

    def __str__(self):
        return "Neuron[%d]: [%s]" % (self.ni, self.__params())


class Layer:
    def __init__(self, neurons: int, ins: int, layerindex: int, use_tanh):
        self.li = layerindex
        self.use_tanh = use_tanh

        self.neurons = [Neuron(ins, i, layerindex, use_tanh) for i in range(neurons)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out

    def __str__(self):
        return

    def apply_grad(self, alpha):
        for neuron in self.neurons:
            neuron.apply_grad(alpha)

    def zero_grad(self):
        for neuron in self.neurons:
            neuron.zero_grad()


class MLP:
    PATH_TO_NETWORKS = "runtime_saves/neural_networks/"
    current_training_step = 0

    def __init__(self, inputs, layout: list, use_tanh: list):
        # Init Layers
        layers = [Layer(layout[0], inputs, 0, use_tanh[0])]
        for i in range(1, len(layout)):
            layers.append(Layer(layout[i], layout[i - 1], i, use_tanh[i]))
        self.layers = layers

    def __call__(self, x: list):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def apply_grad(self, alpha=0.1):
        for layer in self.layers:
            layer.apply_grad(alpha)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def save_to_file(self, name: str):
        file = open(MLP.PATH_TO_NETWORKS + name, "wb")
        pickle.dump(self, file)

    @staticmethod
    def load_from_file(name: str):
        file = open(MLP.PATH_TO_NETWORKS + name, "rb")
        return pickle.load(file)

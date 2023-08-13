import random
import numpy as np
from engine import Value


class Neuron:
    def __init__(self, inputs: int, neuronindex: int, layerindex: int, use_relu):
        self.li = layerindex
        self.ni = neuronindex
        self.inputs = inputs
        self.use_relu = use_relu

        # Initialize Weights and Biases
        self.weights = np.array(
            [Value(random.uniform(-1, 1), label=f"w[{str(self.li)}][{str(self.ni)}][{str(i)}]") for i in range(inputs)])
        self.bias = Value(random.uniform(-1, 1), label=f"b[{str(self.li)}][{str(self.ni)}]")

    def __call__(self, x):
        out = np.sum(np.multiply(x, self.weights))
        out = out + self.bias
        if self.use_relu:
            if isinstance(out, Value):
                out = out.relu()
            else:
                for i in range(out.size):
                    out[i] = out[i].relu()
        return out

    def learn(self, alpha):
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
        s += "b: %0.4f; " % (self.bias.data)


    def __str__(self):
        return "Neuron[%d]: [%s]" % (self.ni, self.__params())


class Layer:
    def __init__(self, neurons: int, ins: int, layerindex: int, use_relu):
        self.li = layerindex
        self.use_relu = use_relu

        self.neurons = np.array([Neuron(ins, i, layerindex, use_relu) for i in range(neurons)])

    def __call__(self, x):
        if self.li == 0:
            out = np.array([n(x_single) for x_single, n in zip(x, self.neurons)])
        else:
            out = np.array([n(x) for n in self.neurons])
        return out

    def __str__(self):
        return

    def learn(self, alpha):
        for neuron in self.neurons:
            neuron.learn(alpha)

    def zero_grad(self):
        for neuron in self.neurons:
            neuron.zero_grad()


class MLP:
    def __init__(self, layout: np.ndarray, use_relu: list):
        # Neural Network layout may only be a 1-D array
        if layout.ndim != 1:
            print("We only accept 1-D arrays as layouts for NNs")
            return

        # Init Layers
        l = [Layer(layout[0], 1, 0, use_relu[0])]
        for i in range(1, layout.size):
            l.append(Layer(layout[i], layout[i - 1], i, use_relu[i]))
        self.layers = np.array(l)

    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def learn(self, alpha=0.1):
        for layer in self.layers:
            layer.learn(alpha)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

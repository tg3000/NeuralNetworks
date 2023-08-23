import math
import numpy as np
import graphviz


class Value:
    def __init__(self, val, op="", label=""):
        self.data = val
        self.grad = 0
        self.children = np.empty((0, 0))
        self.op = op
        self.label = label
        self.grad = 0
        self.__backward = lambda: None
        self.__calc = lambda: None

    def __getstate__(self):
        self.__backward = None
        self.__calc = None
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state
        self.__backward = lambda: None
        self.__calc = lambda: None


    def __str__(self):
        return f'Value({self.data}, Label: {self.label})'

    def __repr__(self):
        return f'Value({self.data}, Label: {self.label})'

    def __allchildren(self) -> list:
        queue = [self]
        children = []
        alr_queued = set()
        alr_queued.add(self)
        while len(queue) > 0:
            current_val = queue.pop(0)
            children.append(current_val)
            for i in current_val.children:
                if not alr_queued.__contains__(i):
                    queue.append(i)
                    alr_queued.add(i)
        return children

    def backward(self):
        allchildren = self.__allchildren()
        for i in allchildren:
            i.__backward()

    def recalc(self):
        allchildren = self.__allchildren()
        allchildren.reverse()
        for i in allchildren:
            i.__calc()

    def zero_grad(self):
        allchildren = self.__allchildren()
        for i in allchildren:
            i.grad = 0

    def draw(self, name=""):
        dot = graphviz.Digraph(name, format="pdf", graph_attr={'rankdir': 'LR', 'label': name})
        # Creates The Nodes themselves
        allchildren = self.__allchildren()
        for i in allchildren:
            dot.node(str(i.__hash__()), "%s | data: %.4f | grad: %.4f" % (i.label, i.data, i.grad), shape='record')
            if i.op != "":
                dot.node(str(i.__hash__()) + i.op, i.op)

        # Creates Arrows between nodes
        for i in allchildren:
            if len(i.children) != 0:
                dot.edge(str(i.__hash__()) + i.op, str(i.__hash__()))
            for j in i.children:
                dot.edge(str(j.__hash__()), str(i.__hash__()) + i.op)

        dot.render(directory="runtime_saves/graphs", view=True)

    def softmax(self, neuron_layer):
        # https://youtu.be/KpKog-L9veg?si=VhpMKIitEKhba-rm - Source
        out = Value(0, op="Softmax")
        out.children = [self]

        def calc():
            try:
                neuron_layer_exp = np.array([math.exp(neuron_layer[i].data) for i in range(neuron_layer.size)])
            except OverflowError:
                print(neuron_layer)
                quit()


            neuron_layer_exp_sum = np.sum(neuron_layer_exp)
            predicted = math.exp(self.data) / neuron_layer_exp_sum
            out.data = predicted
        calc()
        out.__calc = calc

        def backward():
            # self.grad += (predicted * (1 - predicted)) * out.grad
            # pytorch doesn't have backward for softmax but does make up for it in cross_entropy_loss
            self.grad += out.grad

        out.__backward = backward
        return out

    @staticmethod
    # Do not Softmax y_predicitions as method already does this
    def cross_entropy_loss(y_labels: np.ndarray, y_predictions: np.ndarray):
        # https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/ - Source
        softmaxed_predictions = np.array([x.softmax(y_predictions) for x in y_predictions])
        out = Value(0, op="CrossEntropyLoss")
        out.children = softmaxed_predictions.tolist()

        def calc():
            softmaxed_predictions_data = np.array([pred.data for pred in softmaxed_predictions])
            y_labels_data = np.array([label.data for label in y_labels])
            loss = -np.sum(y_labels_data * np.array([math.log(pred) for pred in softmaxed_predictions_data]))
            out.data = loss
        calc()
        out.__calc = calc

        def backward():
            softmaxed_predictions_data = np.array([pred.data for pred in softmaxed_predictions])
            for i in range(len(softmaxed_predictions_data)):
                y_predictions[i].grad += (softmaxed_predictions_data[i] - y_labels[i].data) * out.grad

        out.__backward = backward
        return out

    def relu(self):
        out = Value(max(0, self.data), op="ReLu")
        out.children = [self]

        def calc():
            out.data = max(0, self.data)

        out.__calc = calc

        def backward():
            self.grad += (0 if self.data < 0 else 1) * out.grad

        out.__backward = backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), op="exp")
        out.children = [self]

        def calc():
            out.data = math.exp(self.data)
        out.__calc = calc

        def backward():
            self.grad += out.data * out.grad

        out.__backward = backward
        return out

    def __add__(self, other):
        # In case other is not already a Value
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data + other.data, "+")
        out.children = np.array([self, other])

        def calc():
            out.data = self.data + other.data
        out.__calc = calc

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out.__backward = backward
        return out

    def __mul__(self, other):
        # In case other is not already a Value
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data * other.data, "*")
        out.children = np.array([self, other])

        def calc():
            out.data = self.data * other.data
        out.__calc = calc

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out.__backward = backward
        return out

    def __pow__(self, power, modulo=None):
        # In case other is not already a Value
        if not isinstance(power, Value):
            power = Value(power)

        out = Value(self.data ** power.data, f"**{str(power.data)}|{power.label}")
        out.children = np.array([self])

        def calc():
            out.data = self.data ** power.data
        out.__calc = calc

        def backward():
            self.grad += power.data * (self.data ** (power.data - 1)) * out.grad

        out.__backward = backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return -self + other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return self * (other ** -1)

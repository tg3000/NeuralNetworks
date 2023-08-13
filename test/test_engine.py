import unittest
from backprop.engine import Value
import torch
import numpy as np


class TestValueMethods(unittest.TestCase):

    def test_normal_ops(self):
        x = torch.Tensor([2.3]).double()
        x.requires_grad = True
        a = x + 2
        b = x ** 2
        c = a * b
        d = c - a.relu()
        f = -d / 3
        g = f.relu()
        g.backward()
        xTorch, gTorch = x, g

        x = Value(2.3)
        a = x + 2
        b = x ** 2
        c = a * b
        d = c - a.relu()
        f = -d / 3
        g = f.relu()
        g.grad = 1
        g.backward()
        xValue, gValue = x, g

        self.assertAlmostEqual(gTorch.data.item(), gValue.data, places=3)
        self.assertAlmostEqual(xTorch.grad.item(), xValue.grad, places=3)

    def test_cross_entropy_loss(self):
        test_val1 = 13
        test_val2 = 6

        # Torch Implementation
        a = torch.Tensor([test_val1]).double()
        b = torch.Tensor([test_val2]).double()
        a.requires_grad = True
        b.requires_grad = True

        loss = torch.nn.CrossEntropyLoss(reduction='none')
        inp = torch.softmax(torch.cat([a, b]), dim=0).requires_grad_(True)
        out = loss(torch.cat([a, b]), torch.Tensor([1, 0]).double())
        out.backward()

        # Torch outputs to test
        torch_a_grad = a.grad.item()
        torch_b_grad = b.grad.item()
        torch_loss = out.data.item()
        torch_a_softmax = inp[0].data.item()
        torch_b_softmax = inp[1].data.item()

        # Value Implementation
        inps = np.array([Value(test_val1), Value(test_val2)])

        softmaxes = np.array([inp.softmax(inps) for inp in inps])

        loss = Value.cross_entropy_loss(np.array([1.0, 0.0]), inps)
        loss.grad = 1
        loss.backward()

        # Value outputs to test
        value_a_grad = inps[0].grad
        value_b_grad = inps[1].grad
        value_loss = loss.data
        value_a_softmax = softmaxes[0].data
        value_b_softmax = softmaxes[1].data

        # Tests
        self.assertAlmostEqual(torch_a_grad, value_a_grad, places=3)
        self.assertAlmostEqual(torch_b_grad, value_b_grad, places=3)
        self.assertAlmostEqual(torch_loss, value_loss, places=3)
        self.assertAlmostEqual(torch_a_softmax, value_a_softmax, places=3)
        self.assertAlmostEqual(torch_b_softmax, value_b_softmax, places=3)

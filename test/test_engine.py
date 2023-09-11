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

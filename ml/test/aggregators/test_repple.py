from unittest import TestCase

import torch
from torch import nn, Tensor

from ml.aggregators.repple import Repple


class TestRepple(TestCase):
    def test_similar(self):
        a = nn.Linear(2, 2)
        b = nn.Linear(2, 2)
        c = nn.Linear(2, 2)
        with torch.no_grad():
            for i, p in enumerate(a.parameters()):
                p.mul_(0)
                p.add_(Tensor([0, 1]))
            for i, p in enumerate(b.parameters()):
                p.mul_(0)
                p.add_(Tensor([1, 0]))
            for i, p in enumerate(c.parameters()):
                p.mul_(0)
                p.add_(Tensor([1, -1]))

        res = Repple().aggregate([a, b, c], [a, b, c])

        expected = [
            Tensor([[0, 1], [0, 1]]),
            Tensor([0, 1]),
        ]

        for i, p in enumerate(res.parameters()):
            print(p)
            self.assertTrue(torch.equal(p, expected[i]))

    def test_different(self):
        a = nn.Linear(2, 2)
        b = nn.Linear(2, 2)
        c = nn.Linear(2, 2)
        with torch.no_grad():
            for i, p in enumerate(a.parameters()):
                p.mul_(0)
                p.add_(Tensor([0, 1]))
            for i, p in enumerate(b.parameters()):
                p.mul_(0)
                p.add_(Tensor([1, 0]))
            for i, p in enumerate(c.parameters()):
                p.mul_(0)
                p.add_(Tensor([0, -1]))

        res = Repple().aggregate([a, b, c], [a, b, c])

        expected = [
            Tensor([[1/3, 0], [1/3, 0]]),
            Tensor([1/3, 0]),
        ]

        for i, p in enumerate(res.parameters()):
            print(p)
            self.assertTrue(torch.equal(p, expected[i]))


    def test_history(self):
        a = nn.Linear(2, 2)
        b = nn.Linear(2, 2)
        c = nn.Linear(2, 2)
        with torch.no_grad():
            for i, p in enumerate(a.parameters()):
                p.mul_(0)
                p.add_(Tensor([0, 1]))
            for i, p in enumerate(b.parameters()):
                p.mul_(0)
                p.add_(Tensor([1, 0]))
            for i, p in enumerate(c.parameters()):
                p.mul_(0)
                p.add_(Tensor([1, -1]))

        res = Repple().aggregate([a, b], [a, b, c])

        expected = [
            Tensor([[0, 1], [0, 1]]),
            Tensor([0, 1]),
        ]

        for i, p in enumerate(res.parameters()):
            print(p)
            self.assertTrue(torch.equal(p, expected[i]))

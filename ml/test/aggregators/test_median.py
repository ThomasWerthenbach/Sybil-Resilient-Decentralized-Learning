from unittest import TestCase

import torch
from torch import nn, Tensor

from ml.aggregators.average import Average
from ml.aggregators.median import Median


class TestMedian(TestCase):
    def test_aggregate(self):
        a = nn.Linear(2, 2)
        b = nn.Linear(2, 2)
        c = nn.Linear(2, 2)
        with torch.no_grad():
            for i, p in enumerate(a.parameters()):
                p.mul_(0)
                p.add_(Tensor([i, i + 1]))
            for i, p in enumerate(b.parameters()):
                p.mul_(0)
                p.add_(Tensor([i + 2, i + 3]))
            for i, p in enumerate(c.parameters()):
                p.mul_(0)
                p.add_(Tensor([i + 4, i + 5]))

        agg = Median()
        res = agg.aggregate(None, None, [a, b, c], [])

        expected = [
            Tensor([[2, 3], [2, 3]]),
            Tensor([3, 4]),
        ]

        for i, p in enumerate(res.parameters()):
            self.assertTrue(torch.equal(p, expected[i]))

    def test_aggregate2(self):
        a = nn.Linear(2, 2)
        b = nn.Linear(2, 2)
        c = nn.Linear(2, 2)
        d = nn.Linear(2, 2)
        with torch.no_grad():
            for i, p in enumerate(a.parameters()):
                p.mul_(0)
                p.add_(Tensor([i, i + 1]))
            for i, p in enumerate(b.parameters()):
                p.mul_(0)
                p.add_(Tensor([i + 2, i + 3]))
            for i, p in enumerate(c.parameters()):
                p.mul_(0)
                p.add_(Tensor([i + 4, i + 5]))
            for i, p in enumerate(d.parameters()):
                p.mul_(0)
                p.add_(Tensor([i + 10001231243, i + 1325123213]))

        agg = Median()
        res = agg.aggregate(None, None, [a, b, c, d], [])

        # Pytorch median does not average, it takes the lower value
        expected = [
            Tensor([[2, 3], [2, 3]]),
            Tensor([3, 4]),
        ]

        for i, p in enumerate(res.parameters()):
            self.assertTrue(torch.equal(p, expected[i]))


    def test_aggregate3(self):
        a = nn.Linear(2, 2)
        b = nn.Linear(2, 2)
        c = nn.Linear(2, 2)
        with torch.no_grad():
            for i, p in enumerate(a.parameters()):
                p.mul_(0)
                p.add_(Tensor([i, i + 1]))
            for i, p in enumerate(b.parameters()):
                p.mul_(0)
                p.add_(Tensor([i + 2, i + 3]))
            for i, p in enumerate(c.parameters()):
                p.mul_(0)
                p.add_(Tensor([i + 34252345, i + 124351234]))

        agg = Median()
        res = agg.aggregate(None, None, [a, b, c], [])

        # Pytorch median does not average, it takes the lower value
        expected = [
            Tensor([[2, 3], [2, 3]]),
            Tensor([3, 4]),
        ]

        for i, p in enumerate(res.parameters()):
            self.assertTrue(torch.equal(p, expected[i]))


    def test_aggregate4(self):
        a = nn.Linear(2, 2)
        b = nn.Linear(2, 2)
        c = nn.Linear(2, 2)
        d = nn.Linear(2, 2)
        e = nn.Linear(2, 2)
        with torch.no_grad():
            for i, p in enumerate(a.parameters()):
                p.mul_(0)
                p.add_(Tensor([i, i + 1]))
            for i, p in enumerate(b.parameters()):
                p.mul_(0)
                p.add_(Tensor([i + 2, i + 3]))
            for i, p in enumerate(c.parameters()):
                p.mul_(0)
                p.add_(Tensor([i + 4, i + 5]))
            for i, p in enumerate(d.parameters()):
                p.mul_(0)
                p.add_(Tensor([i + 34252345, i + 124351234]))
            for i, p in enumerate(e.parameters()):
                p.mul_(0)
                p.add_(Tensor([i + 76543456543, i + 76543456543]))

        agg = Median()
        res = agg.aggregate(None, None, [a, b, c, d, e], [])

        # Pytorch median does not average, it takes the lower value
        expected = [
            Tensor([[4, 5], [4, 5]]),
            Tensor([5, 6]),
        ]

        for i, p in enumerate(res.parameters()):
            self.assertTrue(torch.equal(p, expected[i]))

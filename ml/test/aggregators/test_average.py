from unittest import TestCase

import torch
from torch import nn, Tensor

from ml.aggregators.average import Average


class TestAverage(TestCase):
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

        agg = Average()
        res = agg.aggregate([a, b, c], [])

        expected = [
            Tensor([[2, 3], [2, 3]]),
            Tensor([3, 4]),
        ]

        for i, p in enumerate(res.parameters()):
            self.assertTrue(torch.equal(p, expected[i]))

import jax
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod


class CompressionOperator(ABC):
    @abstractmethod
    def compress(self, vec):
        pass

    @abstractmethod
    def beta(self, *args):
        pass

    @abstractmethod
    def theta(self, *args):
        pass


class Top_k(CompressionOperator):
    def __init__(self, k):
        self.k = k

    def compress(self, vec):
        assert len(vec) >= self.k
        _, inds = jax.lax.top_k(jnp.abs(vec), self.k)
        inds = np.array(inds)
        mask = np.zeros_like(vec)
        mask[inds] = 1
        return np.multiply(vec, mask)

    def beta(self, d):
        alpha = float(self.k) / float(d)
        return (1 - alpha) / (1 - np.sqrt(1 - alpha))

    def theta(self, d):
        alpha = float(self.k) / float(d)
        return 1 - np.sqrt(1 - alpha)
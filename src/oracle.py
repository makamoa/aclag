import jax
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod


class Oracle(ABC):
    def __init__(self):
        self.grad = jax.jit(jax.grad(self.loss))

    @abstractmethod
    def loss(self, w):
        pass

    @abstractmethod
    def compute_smoothness(self):
        pass


class OracleContainer(ABC):
    def compute_grad(self, w):
        return jnp.array([oracle.grad(w) for oracle in self.oracles])

    def compute_distributed_smoothness(self):
        L_i = [oracle.compute_smoothness() for oracle in self.oracles]
        return np.sqrt((np.array(L_i) ** 2).mean())

    @abstractmethod
    def compute_smoothness(self):
        pass

    @abstractmethod
    def num_clients(self):
        pass


class LogReg(Oracle):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        self.n = self.X.shape[0]

    def loss(self, w):
        z = - self.X * self.y[:, None]
        exp = jnp.exp(z @ w)
        log = jnp.log(1 + exp)
        return log.mean()

    def compute_smoothness(self):
        xtx = self.X.T.dot(self.X)
        return np.max(np.linalg.eigvalsh(xtx)) / (4 * self.n)


class LogRegContainer(OracleContainer):
    def __init__(self, data):
        self.oracles = [LogReg(ft, l) for (ft, l) in data]
        self.data = data
        self.all_X = self.stack_features()
        self.n = self.all_X.shape[0]

    def stack_features(self):
        return np.vstack([ft for (ft, l) in self.data])

    def compute_smoothness(self):
        xtx = self.all_X.T.dot(self.all_X)
        return np.max(np.linalg.eigvalsh(xtx)) / (4 * self.n)

    def num_clients(self):
        return len(self.data)


class NoncvxLogReg(LogReg):
    def __init__(self, X, y, lambda_):
        super().__init__(X, y)
        self.lambda_ = lambda_

    def loss(self, w):
        reg = 1. - 1. / (w ** 2 + 1.)
        reg = reg.sum()
        return LogReg.loss(self, w) + self.lambda_ * reg

    def compute_smoothness(self):
        return LogReg.compute_smoothness(self) + 2 * self.lambda_


class NoncvxLogRegContainer(LogRegContainer):
    def __init__(self, data, lambda_):
        self.oracles = [NoncvxLogReg(ft, l, lambda_) for (ft, l) in data]
        self.data = data
        self.lambda_ = lambda_
        self.all_X = LogRegContainer.stack_features(self)
        self.n = self.all_X.shape[0]

    def compute_smoothness(self):
        return LogRegContainer.compute_smoothness(self) + 2 * self.lambda_
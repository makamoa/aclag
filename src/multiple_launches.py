import argparse
import numpy as np
import jax.numpy as jnp
import oracle
from data_processing import download_and_process_data
from compression_operator import Top_k
import algorithms
from copy import deepcopy
import pickle

# parse arguments
parser = argparse.ArgumentParser(description='Run CLAG for a grid of paramters.')
parser.add_argument('--dataset', action='store', dest='dataset_name', type=str,
                    help='Dataset name. \
                    Options are phishing, a9a, w6a, ijcnn1.')
parser.add_argument('--time_limit', action='store', dest='time_limit',
                    type=int, default=180, help='Computation time limit.')
args = parser.parse_args()


num_clients = 20
lambda_ = 0.1
X, y, data = download_and_process_data(args.dataset_name, num_clients)
d = X.shape[1]

oracle_container = oracle.NoncvxLogRegContainer(data, lambda_)
L = oracle_container.compute_smoothness()
L_tilde = oracle_container.compute_distributed_smoothness()
print('Oracle used is {}. Smoothness constant (L) is {}. Distributed \
smoothness (L_tilde) is {}.'.format(oracle_container.__class__.__name__, L,
                                    L_tilde))

stepsize_scales = 2 ** np.arange(0, 11, 2)
x_0 = jnp.zeros(d)
max_comm = 20000
zetas = np.geomspace(2 ** -8, 2 ** 8, 5)
zetas = np.insert(zetas, 0, 0)
ks = [1, int(d / 4), int(d/2), d]
histories = dict()
for k in ks:
    c_op = Top_k(k)
    beta = c_op.beta(d)
    theta = c_op.theta(d)
    for zeta in zetas:
        theoretical_stepsize = 1. / (L + L_tilde * np.sqrt(
            max(zeta, beta) / theta))
        history_best = [np.float32('inf')]
        history_comm_best = None
        best_scale = None
        for stepsize_scale in stepsize_scales:
            print('{}-{}-{}'.format(k, zeta, stepsize_scale))
            stepsize = stepsize_scale * theoretical_stepsize
            history, history_comm = algorithms.CLAG(x_0,
                                                    oracle_container,
                                                    c_op,
                                                    stepsize,
                                                    zeta,
                                                    max_comm,
                                                    args.time_limit)
            print('')
            if history[-1] < history_best[-1]:
                history_best = deepcopy(history)
                history_comm_best = deepcopy(history_comm)
                best_scale = stepsize_scale
        histories[(k, zeta)] = (history_comm_best, history_best, best_scale)
        with open('../results/histories_{}_plot.pickle'.format(
                args.dataset_name), 'wb') as file:
            pickle.dump(histories, file)
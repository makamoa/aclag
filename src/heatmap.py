import argparse
import numpy as np
import jax.numpy as jnp
import oracle
import algorithms
from data_processing import download_and_process_data

# parse arguments
parser = argparse.ArgumentParser(description='Compute heatmap table for CLAG.')
parser.add_argument('--k_num', action='store', dest='k_num', type=int,
                    help='Number of top-k operators. k for top-k operators\
                    is linearly interpolated between 1 and d.')
parser.add_argument('--max_pow', action='store', dest='max_pow', type=int,
                    help='The largest power of zeta. \
                    Zetas are powers of two including zero.')
parser.add_argument('--dataset', action='store', dest='dataset_name', type=str,
                    help='Dataset name. \
                    Options are phishing, a9a, w6a, ijcnn1.')
parser.add_argument('--stepsize_alignment', action='store',
                    dest='stepsize_alignment', type=str, default='EF21',
                    help='What algorithm sets the stepsize. When "EF21", \
                    the best stepsize for EF21 is kept for CLAG with \
                    non-zero zeta. When "CLAG", the best stepsize is \
                    computed for each pair of parameters (k and zeta).')
parser.add_argument('--time_limit', action='store', dest='time_limit',
                    type=int, default=180, help='Computation time limit.')
parser.add_argument('--tolerance', action='store', dest='tol',
                    type=float, default=0.0001, help='Tolerance level which \
                    sets a stop criterion.')
args = parser.parse_args()

# hard-coded experiment parameters; TODO: add these parameter to 'parser'
num_clients = 20
lambda_ = 0.1
stepsize_coefs = 2 ** jnp.arange(12)

# download data
X, y, data = download_and_process_data(args.dataset_name, num_clients)
d = X.shape[1]

# create an oracle container and compute its parameters;
# TODO: pass oracle itself to 'parser'
oracle_container = oracle.NoncvxLogRegContainer(data, lambda_)
L = oracle_container.compute_smoothness()
L_tilde = oracle_container.compute_distributed_smoothness()
print('Oracle used is {}. Smoothness constant (L) is {}. Distributed \
smoothness (L_tilde) is {}.'.format(oracle_container.__class__.__name__, L,
                                    L_tilde))

# set heatmap ranges
ks = np.linspace(1, d, args.k_num, endpoint=True, dtype=int)
trigger_betas = 2 ** jnp.arange(0, args.max_pow, dtype=np.float32)
trigger_betas = jnp.insert(trigger_betas, 0, 0)

# computation parameters
x_0 = jnp.zeros(d)
save_filepath = '../results/heatmap_{}_{}.npy'.format(
    args.stepsize_alignment, args.dataset_name)

# computation
if args.stepsize_alignment == 'EF21':
    heatmap = algorithms.heatmap_CLAG(x_0, oracle_container, ks, trigger_betas,
                                      args.tol, stepsize_coefs, args.time_limit,
                                      save_filepath)
elif args.stepsize_alignment == 'CLAG':
    heatmap = algorithms.heatmap_CLAG_full(x_0, oracle_container, ks,
                                           trigger_betas, args.tol, stepsize_coefs,
                                           args.time_limit, save_filepath)
else:
    print('Invalid value for stepsize_alignment argument.')
    raise
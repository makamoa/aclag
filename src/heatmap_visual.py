import argparse
import numpy as np
import jax.numpy as jnp
from sklearn.datasets import load_svmlight_file
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt

# parse arguments
parser = argparse.ArgumentParser(description='Compute heatmap table for CLAG.')
parser.add_argument('--dataset', action='store', dest='dataset_name', type=str,
                    help='Dataset name. \
                    Options are phishing, a9a, w6a, ijcnn1.')
parser.add_argument('--stepsize_alignment', action='store',
                    dest='stepsize_alignment', type=str, default='EF21',
                    help='What algorithm sets the stepsize. When "EF21", \
                    the best stepsize for EF21 is kept for CLAG with \
                    non-zero zeta. When "CLAG", the best stepsize is \
                    computed for each pair of parameters (k and zeta).')
args = parser.parse_args()
dataset_name = args.dataset_name
stepsize_alignment = args.stepsize_alignment

# get the number of dimensions
raw_data = load_svmlight_file('../data/' + dataset_name)
X, y = raw_data
d = X.shape[1]
del raw_data, X, y

# download heatmap
save_filepath = '../results/heatmap_{}_{}.npy'.format(
    stepsize_alignment, dataset_name)
heatmap = np.load(save_filepath)

# set heatmap ranges
k_num = heatmap.shape[0]
max_pow = heatmap.shape[1] - 1
ks = np.linspace(1, d, k_num, endpoint=True, dtype=int)
trigger_betas = 2 ** jnp.arange(0, max_pow, dtype=int)
trigger_betas = jnp.insert(trigger_betas, 0, 0)

# save heatmap as pandas.DataFrame
mask_ = heatmap != -1
max_heatmap = int(heatmap.max().item())
min_heatmap = int(heatmap.min(initial=max_heatmap, where=mask_).item())
df = pd.DataFrame(heatmap, index=ks, columns=trigger_betas, dtype=int)
plt.figure(figsize=(20, 10))
log_norm = LogNorm(vmin=min_heatmap, vmax=max_heatmap)
ax = sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu",
                 mask=pd.DataFrame(df == -1), norm=log_norm)
ax.set_xlabel('trigger constant')
ax.set_ylabel('compression level')
plt.title('{}, {}'.format(dataset_name, stepsize_alignment))
plt.tight_layout()
plt.savefig('../plots/heatmap_{}_{}.pdf'.format(
    stepsize_alignment, dataset_name))
import numpy as np
from sklearn.datasets import load_svmlight_file


def download_and_process_data(dataset_name: str, num_clients: int):
    # download data
    raw_data = load_svmlight_file('../data/' + dataset_name)
    X, y = raw_data
    print('Dataset is downloaded.')

    # labels (0, 1) are translated into (-1, 1)
    if ((y == 0).sum()) > 0:
        y = 2 * y - 1

    # cut unnecessary data
    residual = X.shape[0] % num_clients
    X = X[:-residual].toarray()
    y = y[:-residual]

    # divide data between clients
    inds = np.array_split(np.arange(X.shape[0]), num_clients)
    data = []
    for i in range(num_clients):
        data.append((X[inds[i]][:], y[inds[i]]))
    print('Data is evenly divided between {} clients'.format(num_clients))
    return X, y, data

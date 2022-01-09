import numpy as np
import scipy.sparse as sp
import tensorflow as tf

def normalized_adj_single(adj, method=1):
    print('normalized_adj_single method %d is functioning!' % method)
    rowsum = np.array(adj.sum(1), dtype=np.float32)

    if method == 1:
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
    elif method == 2:
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv).todense()
        print('d_mat_inv obtained, shape:', d_mat_inv.shape)
        norm_adj = np.dot(np.dot(d_mat_inv, adj), d_mat_inv)
    print('generate single-normalized adjacency matrix.')
    return norm_adj



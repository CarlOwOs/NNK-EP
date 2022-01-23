import torch
import torch_geometric
import numpy as np

from torch_geometric import utils

from utils.non_neg_qpsolver import non_negative_qpsolver
import utils.graph_utils as gutils

def nnk_prunning(x, edge_index, edge_attr, training=True):

    #if not training:
    #    return edge_index, edge_attr
    num_nodes = x.size(0)

    X = x.detach().numpy()

    reg = 1e-6
    mask = torch.empty(0, dtype=torch.bool)

    for node_i in range(num_nodes):
        neighbor_indices = gutils.get_neighbors(edge_index, node_i) # adjacency of that node
        nodes = np.append(node_i, neighbor_indices)
        X_i = X[nodes]

        D = gutils.create_distance_matrix(X_i, len(nodes))
        sigma = np.mean(np.amax(D, 0)) / 3

        if sigma != 0:

            G = np.exp(-D ** 2 / (2 * sigma ** 2))

            G_i = G[1:len(nodes), 1:len(nodes)] # neighbor kernel matrix
            g_i = G[1:len(nodes), 0] # node kernel vector
            x_opt, _ = non_negative_qpsolver(G_i, g_i, g_i, reg)

            mask_node = x_opt > reg

        else:

            mask_node = np.ones(len(nodes) - 1, dtype=bool)

        mask_node = torch.from_numpy(mask_node)
        mask = torch.cat((mask, mask_node), 0)

    row, col  = edge_index
    row, col, edge_attr = gutils.filter_adj(row, col, edge_attr, mask)

    edge_index = torch.stack([row, col], dim=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    return edge_index.to(torch.device(device)), None if edge_attr is None else edge_attr.to(torch.device(device))
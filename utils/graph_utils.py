import numpy as np

def get_neighbors(edge_index, node_i):
    idx = edge_index[0] == node_i
    node_neighbor_idx = edge_index[1][idx]
    return node_neighbor_idx


def create_distance_matrix(X, num_nodes):
    D = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            D[i, j] = np.linalg.norm(X[i] - X[j])
            D[j, i] = D[i, j]
    return D

def filter_adj(row, col, edge_attr, mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]
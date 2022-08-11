"""Various utility functions."""
import numpy as np
import scipy.sparse as sp
import torch
from typing import Union
import networkx as nx
import random
from sklearn.preprocessing import normalize, StandardScaler
from collections import Counter
import pandas as pd
from scipy.stats import norm


def load_dataset(dir_net, bin_num, header = False):

    with open(dir_net, mode='r') as f:
        if header:
            next(f)
        net_nA = []
        net_nB = []
        for line in f:
            na, nb = line.strip("\n").split("\t")
            net_nA.append(na)
            net_nB.append(nb)

    G = nx.Graph()
    G.add_edges_from(list(zip(net_nA, net_nB)))
    net_node = list(max(nx.connected_components(G), key = len))
    net_node = sorted(net_node)


    NODE2ID = dict()
    ID2NODE = dict()
    for idx in range(len(net_node)):
        ID2NODE[idx] = net_node[idx]
        NODE2ID[net_node[idx]] = idx

    row_idx = []
    col_idx = []
    val_idx = []
    for n_a, n_b in zip(net_nA, net_nB):
        if n_a in net_node and n_b in net_node:
            row_idx.append(NODE2ID[n_a])
            col_idx.append(NODE2ID[n_b])
            val_idx.append(1.0)
            row_idx.append(NODE2ID[n_b])
            col_idx.append(NODE2ID[n_a])
            val_idx.append(1.0)

    A = sp.csr_matrix((np.array(val_idx), (np.array(row_idx), np.array(col_idx))), shape=(len(net_node), len(net_node)))
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()

    G_lcc = nx.Graph()
    G_lcc.add_edges_from(list(zip(row_idx, col_idx)))
    node_degree = {n: G_lcc.degree[n] for n in G_lcc.nodes}
    sorted_node_degree = dict(sorted(node_degree.items(), key=lambda item: item[1]))
    node2bin = np.array_split(list(sorted_node_degree.keys()), bin_num)

    return A, G_lcc, NODE2ID, ID2NODE, node2bin



def l2_reg_loss(model, scale=1e-5):

    loss = 0.0
    for w in model.get_weights():
        loss += w.pow(2.).sum()
    return loss * scale



def to_sparse_tensor(matrix: Union[sp.spmatrix, torch.Tensor],
                     cuda: bool = True,
                     ) -> Union[torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor]:

    '''
    This function's code is borrowed from: https://github.com/shchur/overlapping-community-detection/blob/master/nocd/utils.py
    '''

    if sp.issparse(matrix):
        coo = matrix.tocoo()
        indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
        values = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    elif torch.is_tensor(matrix):
        row, col = matrix.nonzero().t()
        indices = torch.stack([row, col])
        values = matrix[row, col]
        shape = torch.Size(matrix.shape)
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    else:
        raise ValueError(f"matrix must be scipy.sparse or torch.Tensor (got {type(matrix)} instead).")
    if cuda:
        sparse_tensor = sparse_tensor.cuda()
    return sparse_tensor.coalesce()



def feature_generator(A, preprocess = None):
    assert preprocess in [None, "normalize", "standardscaler"]
    if preprocess == None:
        feat = A
    elif preprocess == "normalize":
        feat = normalize(A)  # adjacency matrix
    else:
        scaler = StandardScaler(with_mean = False)
        feat = scaler.fit_transform(A)
    return feat


def normalize_adj(adj, sparse=True):

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    if sparse:
        return res.tocoo()
    else:
        return res.todense()



def adj_polynomials(adj, k, sparse=True):

    adj_normalized = normalize_adj(adj, sparse=sparse)

    p_k = []
    if sparse:
        p_k.append(sp.eye(adj.shape[0]))
    else:
        p_k.append(np.eye(adj.shape[0]))

    p_k.append(adj_normalized)

    for p in range(2, k+1):
        p_k.append(sp.csr_matrix.power(adj_normalized, p))

    return p_k


def _volume(S, n, degG):
    '''
    volume(node) = sum(neighbors' degrees) + degree of itself
    paper link: https://arxiv.org/pdf/1112.0031.pdf
    :param S: node set S = node + node 1st-order neighbors
    :param degG: dictionary, key = node, value = node degree
    :return: volume(S) (numeric)
    '''

    S.add(n)
    vol = 0
    for m in S:
        vol += degG[m]
    return vol


def _edge(nxG, S, n):
    '''
    paper link: https://arxiv.org/pdf/1112.0031.pdf
    :param nxG:
    :param S: node set S = node + node 1st-order neighbors
    :return: edges(S)
    '''
    S.add(n)
    return 2 * len(nxG.subgraph(S).edges)


def graph_property(nxG):
    '''
    compute degree(node), volume(set), edges(set)
    input: networkx graph object
    :return: condG
    '''
    _degG = {n: len(set(nxG[n])) for n in set(nxG.nodes)} # node degree
    _volG = {n: _volume(set(nxG[n]), n, _degG) for n in set(nxG.nodes)} # volume(node's neighborhood)
    _edgeG = {n: _edge(nxG, set(nxG[n]), n) for n in set(nxG.nodes)} # edges(node's neighborhood)
    _cutG = {n: _volG[n] - _edgeG[n] for n in set(nxG.nodes)} # cut(node's neighborhood)
    _cvolG = {n: _volume(set(nxG.nodes) - set(nxG[n]), n, _degG) for n in set(nxG.nodes)} #vol(S bar)
    condG = {n: _cutG[n] / min(_volG[n], _cvolG[n]) for n in set(nxG.nodes)} # conductance(node's neighborhood)

    return condG



def cluster_number(G, rand_seed, ratio):
    '''
    determine the community number
    input: networkx object G
    output: community number
    '''
    nsplit_edges = int(len(G.edges) * ratio)
    random.seed(rand_seed)
    sub_edges = random.sample(list(G.edges), nsplit_edges)
    G_sub = nx.Graph()
    G_sub.add_edges_from(sub_edges)
    cond_Gsub = graph_property(G_sub)
    clustercenter = [n for n in set(G_sub.nodes) if
                      cond_Gsub[n] < min([cond_Gsub[m] for m in G_sub[n]])]
    return len(clustercenter)



def cluster_infer(Z_pred, ID2NODE):
    clust_results = dict()
    for idx in range(Z_pred.shape[0]):
        clust_results[ID2NODE[idx]] = []
        clust_sets = np.where(Z_pred[idx, ] > 0)[0]
        for cidx in clust_sets:
            clust_results[ID2NODE[idx]].append(cidx)
    return clust_results



def load_snp(dir_net, header = False):

    with open(dir_net, mode='r') as f:
        if header:
            next(f)
        snp_id = []
        gene_id = []
        for line in f:
            snp, entrez = line.strip("\n").split("\t")
            snp_id.append(snp)
            gene_id.append(entrez)

    return set(gene_id)




def null_dist_score(non_snp_genes, rand_num, node_num, curr_gene, cluster_results):
    rand_gene_score = []
    gene_clust = set(cluster_results[curr_gene])
    temp_non_snp_genes = non_snp_genes - set(curr_gene)
    for _ in range(rand_num):
        rand_set = random.sample(list(temp_non_snp_genes), node_num)
        rand_score = 0.0
        for rg in rand_set:
            rg_clust = set(cluster_results[rg])
            if len(rg_clust) > 0 and len(gene_clust) > 0:
                rand_score += len(gene_clust.intersection(rg_clust)) / len(rg_clust)
        rand_gene_score.append(rand_score)
    rand_mean = np.array(rand_gene_score).mean()
    rand_std = np.array(rand_gene_score).std()
    return rand_mean, rand_std




def BuildSumScore(cluster_results, snp_dict):
    all_genes = list(cluster_results.keys())
    all_genes_score = dict()
    for gwas_f in snp_dict:
        snp_clust = {gene: cluster_results[gene] for gene in snp_dict[gwas_f] if gene in cluster_results}
        non_snp_genes = set(all_genes) - set(snp_clust.keys())
        all_genes_score[gwas_f] = []
        for gene in all_genes:
            gene_clust = set(cluster_results[gene])
            gene_score = 0.0
            for sp in snp_clust:
                sp_clust = set(cluster_results[sp])
                if len(sp_clust) > 0 and len(gene_clust) > 0:
                    gene_score += len(gene_clust.intersection(sp_clust)) / len(sp_clust)

            rand_mean, rand_std = null_dist_score(non_snp_genes, 1000, len(snp_clust), gene, cluster_results)
            gene_zscore = (gene_score - rand_mean) / rand_std
            if 1 - norm.cdf(gene_zscore) < 0.05:
                all_genes_score[gwas_f].append(gene_score)
            else:
                all_genes_score[gwas_f].append(0)

    return all_genes, all_genes_score



def BuildIntegratedScore(all_genes, feature_score):
    normalized_feat_score = np.zeros((len(all_genes), len(feature_score)))
    count = 0
    for feat in feature_score:
        normalized_feat_score[:, count] = np.array(feature_score[feat])
        count += 1
    final_score = normalized_feat_score.sum(axis = 1)

    return all_genes, final_score




def null_dist_score1(non_snp_genes, rand_num, node_num, curr_gene, cluster_results):
    rand_gene_score = []
    gene_clust = set(cluster_results[curr_gene])
    temp_non_snp_genes = non_snp_genes - set(curr_gene)
    for _ in range(rand_num):
        rand_set = random.sample(list(temp_non_snp_genes), node_num)
        rand_score = 0.0
        for rg in rand_set:
            rg_clust = set(cluster_results[rg])
            if len(rg_clust) > 0 and len(gene_clust) > 0:
                rand_score += len(gene_clust.intersection(rg_clust)) / len(rg_clust)
        rand_gene_score.append(rand_score)
    rand_mean = np.array(rand_gene_score).mean()
    rand_std = np.array(rand_gene_score).std()
    return rand_mean, rand_std


def FilterLCC(dir_lcc, all_genes, integrated_score):

    with open(dir_lcc, 'rb') as handle:
        lcc = pickle.load(handle)
    handle.close()

    filter_genes = []
    filter_score = []

    for gene, score in zip(all_genes, integrated_score):
        if int(gene) in lcc:
            filter_genes.append(int(gene))
            filter_score.append(score)


    return filter_genes, filter_score
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sympy.tensor import tensor
from torch.nn import Linear

from layers import GraphConvolution
import torch
import sympy

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x




class decoder(torch.nn.Module):
    def __init__(self, nfeat,  nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]


class MLP_L(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_L, self).__init__()
        self.wl = Linear(n_mlp, 64)

    def forward(self, mlp_in):
        weight_output =self.wl(mlp_in)

        return weight_output


class MorphST(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(MorphST, self).__init__()
        
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.ZINB = decoder(nfeat, nhid1, nhid2)
        self.dropout = dropout
        self.meta = nn.Parameter(torch.Tensor([0.1]))
        self.meta.data.clamp_(0, 1)
        self.alpha = nn.Parameter(torch.Tensor([0.1]))
        self.alpha.data.clamp_(0,1)


        self.MLP = nn.Sequential(
            nn.Linear(256, 64)#192
        )
        self.MLP_L=MLP_L(64)
        self.MLP_I = nn.Sequential(
            nn.Linear(2048,256),
            nn.Linear(256,64)
        )


    def forward(self, x, sadj, fadj,img_fea,img_sim, epoch, pretrain):
        contrastive_loss1 = 0.0
        if epoch < 100 and pretrain:
            edge_probs = convert_edge_probabilities(sadj, img_sim)
            #x_1 = drop_feature(x, 0.1)
            #x_2 = drop_feature(x, 0.2)
            sadj1 = multiple_dropout_average(sadj, edge_probs)
            sadj2 = multiple_dropout_average(sadj, edge_probs)
            emb_1_1 = self.SGCN(x, sadj1)  # Spatial_GCN
            emb_1_2 = self.SGCN(x, sadj2)  # Spatial_GCN
            contrastive_loss1 = contrastive_loss(emb_1_1, emb_1_2, sadj)

        emb1 = self.SGCN(x, sadj)  # Spatial_GCN

        emb2 = self.FGCN(x, fadj)  # Feature_GCN
        conadj = float(self.meta) * fadj + (1 - float(self.meta)) * sadj
    
        img_emb = self.MLP_I(img_fea)
        conadj = image_guided_diffusion_refined(conadj,img_sim,self.alpha)

        com = self.CGCN(x, conadj)


        emb = torch.stack([emb1, com, emb2, img_emb], dim=1)
        a = self.MLP_L(emb)
        emb = F.normalize(a, p=2)

        emb = torch.cat((emb[:, 0].mul(emb1), emb[:, 1].mul(com), emb[:, 2].mul(emb2),emb[:, 3].mul(img_emb)), 1)
        emb = self.MLP(emb)

        [pi, disp, mean] = self.ZINB(emb)
        return emb, pi, disp, mean, emb1, emb2, com,img_emb, contrastive_loss1


def image_guided_diffusion_refined(adj, sim, alpha=0.1):
    
    if adj.is_sparse:
        adj = adj.to_dense()
    
    refined_sim = torch.pow(sim, 2)
    
    A_img = adj * refined_sim
    A_combined = (1 - alpha) * adj + alpha * A_img
    A_combined = A_combined#*mask
    
    row_sum = torch.sum(A_combined, dim=1)
    d_inv_sqrt = torch.pow(row_sum + 1e-6, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(d_inv_sqrt)

    norm_adj = torch.mm(torch.mm(D_inv_sqrt, A_combined), D_inv_sqrt)
    
    return norm_adj

def convert_edge_probabilities(adj_matrix, edge_prob_matrix):
    adj_matrix = adj_matrix.to_dense()
    edge_probs = torch.zeros_like(adj_matrix, dtype=torch.float)
    
    edge_probs[adj_matrix != 0] = edge_prob_matrix[adj_matrix != 0]

    return edge_probs

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        device=torch.device('cpu')).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def filter_adj(row, col, edge_attr,mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]


def dropout_adj(
        edge_index,
        edge_attr,
        force_undirected: bool = False,
        training: bool = True,):
    if not training:
        return edge_index, edge_attr

    row, col = edge_index

    if force_undirected:
        mask = row <= col
        row, col, edge_attr = row[mask], col[mask], edge_attr[mask]

    edge_attr_scaled = edge_attr
    edge_attr_scaled_cpu = edge_attr_scaled.to('cpu')


    mask = torch.rand(edge_attr_scaled.size(0), device=torch.device('cpu')) >= edge_attr_scaled_cpu

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)], dim=0)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr, mask


def multiple_dropout_average(conadj,
                                    edge_probs,
                                    num_trials: int = 10,
                                    threshold_ratio: float = 0.5,
                                    training: bool = True,
                                    device: str = 'cuda'):
    if not training:
        return conadj.to(device)

    conadj = conadj.to_dense().to(device)
    edge_probs = edge_probs.to(device)
    edge_count = torch.zeros_like(conadj)

    for _ in range(int(num_trials)):
        # guided dropout
        drop_mask = (torch.rand_like(conadj) < edge_probs).float()
        dropped_adj = conadj * drop_mask
        edge_count += (dropped_adj > 0).float()

    threshold = num_trials * threshold_ratio
    final_adj = (edge_count >= threshold).float()

    return final_adj


def random_dropout_adj(
    edge_index,
    edge_attr,
    p: float = 0.5,
    force_undirected: bool = False,
    num_nodes = None,
    training: bool = True,
):

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        return edge_index, edge_attr

    row, col = edge_index

    mask = torch.rand(row.size(0), device=torch.device('cpu')) >= p

    if force_undirected:
        mask[row > col] = False

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)], dim=0)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr

def sim(z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

def neighbor_readout(z: torch.Tensor, adj: torch.Tensor):
        """Aggregate neighbor embeddings using mean pooling"""
        #adj_no_self = adj - torch.diag_embed(adj.diag())  # remove self-loop
        adj_no_self = adj.clone()
        adj_no_self[adj_no_self > 0] = 1
        degree = adj_no_self.sum(1, keepdim=True) + 1e-8
        z_neigh = adj_no_self @ z / degree
        return z_neigh

def nei_con_loss(z1: torch.Tensor, z2: torch.Tensor, adj, mask=None):
        '''neighbor contrastive loss'''
        adj = adj.to_dense()
        adj = adj - torch.diag_embed(adj.diag())  # remove self-loop
        adj[adj > 0] = 1

        # add
        # Neighbor readout
        z1_readout = neighbor_readout(z1, adj)
        z2_readout = neighbor_readout(z2, adj)
        #

        nei_count = torch.sum(adj, 1) * 2 + 1  # intra-view nei+inter-view nei+self inter-view
        nei_count = torch.squeeze(torch.tensor(nei_count))

        f = lambda x: torch.exp(x / 15)
        
        if mask is None:
            intra_view_sim = f(sim(z1, z1_readout))
            inter_view_sim = f(sim(z1, z2_readout))
        else:
            intra_view_sim = f(sim(z1, z1_readout)) * mask
            inter_view_sim = f(sim(z1, z2_readout)) * mask

        loss = (inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)) / (
                intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())
        loss = loss / nei_count  # divided by the number of positive pairs for each node

        return -torch.log(loss)

def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, adj,mask=None,
                         mean: bool = True):
        l1 = nei_con_loss(z1, z2, adj, mask)
        l2 = nei_con_loss(z2, z1, adj, mask)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
from __future__ import division
from __future__ import print_function

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
from sklearn import metrics

from config import Config
from models import MorphST
from utils import *
pretrain = True

def load_data(dataset):
    print("load data:")
    path = "./generate_data/DLPFC/" + dataset + "/MorphST.h5ad"
    adata = sc.read_h5ad(path)
    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground']
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])
    img_fea = adata.obsm['image_feature']
    img_sim = adata.obsm['edge_probabilities']
    print("done")
    return adata, features, labels, nfadj, nsadj, graph_nei, graph_neg, img_fea,img_sim


def train(epoch):
    model.train()
    emb ,pi, disp, mean,emb1,emb2,com, img_emb,contrastiveloss= model(features, sadj, fadj,img_fea,img_sim,epoch,pretrain)
    if epoch < 100:
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=config.weight_decay)
        zinb_loss = 0.0
        reg_loss = 0.0
        dcir_loss = 0.0
        total_loss = contrastiveloss
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
        reg_loss = regularization_loss(emb, graph_nei, graph_neg)

        dcir_loss= dicr_loss(emb1, emb2)


        total_loss =  config.alpha * zinb_loss+config.gamma * reg_loss+0.1*dcir_loss #+ 0.01*dcir_loss2
    optimizer.zero_grad()
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values
    total_loss.backward()
    optimizer.step()
    return emb, mean, zinb_loss, reg_loss, dcir_loss, total_loss

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datasets = ['151507', '151508', '151509', '151510', '151669', '151670',
             '151671', '151672', '151673', '151674', '151675', '151676']
    #datasets = ['151507']
    for i in range(len(datasets)):
        dataset = datasets[i]
        config_file = './config/DLPFC.ini'
        print(dataset)
        adata, features, labels, fadj, sadj, graph_nei, graph_neg,img_fea,img_sim = load_data(dataset)
        #print(adata)

        plt.rcParams["figure.figsize"] = (3, 3)
        savepath = './results/DLPFC/' + dataset + '/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        title = "Manual annotation (slice #" + dataset + ")"
        sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], title=title,
                      show=False)
        plt.savefig(savepath + 'Manual Annotation.pdf', bbox_inches='tight', dpi=600)
        plt.show()

        config = Config(config_file)
        cuda = not config.no_cuda and torch.cuda.is_available()
        use_seed = not config.no_seed

        _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
        ground = torch.LongTensor(ground)
        config.n = len(ground)
        config.class_num = len(ground.unique())

        config.epochs = 400
        config.epochs = config.epochs + 1

        img_fea = torch.as_tensor(img_fea, dtype=torch.float)
        img_sim = torch.as_tensor(img_sim,dtype=torch.float)
        if cuda:
            features = features.cuda()
            sadj = sadj.cuda()
            fadj = fadj.cuda()
            graph_nei = graph_nei.cuda()
            graph_neg = graph_neg.cuda()
            img_fea = img_fea.cuda()
            img_sim = img_sim.cuda()

        import random

        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        if not config.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
        print(dataset, ' ', config.lr, ' ', config.alpha, ' ', config.beta, ' ', config.gamma)
        model = MorphST(nfeat=config.fdim,
                             nhid1=config.nhid1,
                             nhid2=config.nhid2,

                             dropout=config.dropout)
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        epoch_max = 0
        ari_max = 0
        v_measure_max = 0
        purity_max = 0
        Homogeneity_max = 0
        Completeness_max = 0
        idx_max = []
        mean_max = []
        emb_max = []

        for epoch in range(config.epochs):
            emb, mean, zinb_loss, reg_loss, dcir_loss,total_loss = train(epoch)
            print(dataset, ' epoch: ', epoch,' zinb_loss = {:.2f}'.format(zinb_loss),
                  ' reg_loss = {:.2f}'.format(reg_loss), ' dcir_loss= {:.2f}'.format(dcir_loss),


                  ' total_loss = {:.2f}'.format(total_loss))
            kmeans = KMeans(n_clusters=config.class_num).fit(emb)
            idx = kmeans.labels_
            ari_res = metrics.adjusted_rand_score(labels, idx)
            nmi_res = metrics.normalized_mutual_info_score(labels, idx)
            v_measure_res = metrics.v_measure_score(labels, idx)
            Homogeneity_res = metrics.homogeneity_score(labels, idx)
            Completeness_res = metrics.completeness_score(labels, idx)
            print(ari_res)
            if ari_res > ari_max:
                ari_max = ari_res
                epoch_max = epoch
                idx_max = idx
                mean_max = mean
                emb_max = emb
                nmi_max = nmi_res
                v_measure_max = v_measure_res
                Homogeneity_max = Homogeneity_res
                Completeness_max = Completeness_res

        print(dataset, ' ', ari_max)

        title = 'MorphST: ARI={:.2f}'.format(ari_max)
        adata.obs['idx'] = idx_max.astype(str)
        adata.obsm['emb'] = emb_max
        adata.obsm['mean'] = mean_max

        sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title, show=False)
        plt.savefig(savepath + 'MorphST.pdf', bbox_inches='tight', dpi=600)
        plt.show()

        sc.pp.neighbors(adata, use_rep='mean')
        sc.tl.umap(adata)
        plt.rcParams["figure.figsize"] = (3, 3)
        sc.tl.paga(adata, groups='idx')
        sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, title=title, legend_fontoutline=2,
                           show=False)
        plt.savefig(savepath + 'MorphST_umap_mean.pdf', bbox_inches='tight', dpi=600)
        plt.show()

        pd.DataFrame(emb_max).to_csv(savepath + 'MorphST_emb.csv')
        pd.DataFrame(idx_max).to_csv(savepath + 'MorphST_idx.csv')
        adata.layers['X'] = adata.X
        adata.layers['mean'] = mean_max
        adata.write(savepath + 'MorphST.h5ad')


        sc.pl.umap(adata, color=['idx'], frameon = False)
from __future__ import division
from __future__ import print_function

from utils import features_construct_graph, spatial_construct_graph1
import os
import argparse
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from config import Config
import ast
from tqdm import tqdm
from scipy.spatial.distance import cdist,euclidean,cosine
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def normalize(adata, highly_genes=3000):
    print("start select HVGs")
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata


def load_ST_file(dataset, highly_genes, k, radius):
    path = "./data/DLPFC/" + dataset + "/"
    labels_path = path + "metadata.tsv"

    labels = pd.read_table(labels_path, sep='\t')
    labels = labels["layer_guess_reordered"].copy()
    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])
    ground = labels.copy()
    ground.replace('WM', '0', inplace=True)
    ground.replace('Layer1', '1', inplace=True)
    ground.replace('Layer2', '2', inplace=True)
    ground.replace('Layer3', '3', inplace=True)
    ground.replace('Layer4', '4', inplace=True)
    ground.replace('Layer5', '5', inplace=True)
    ground.replace('Layer6', '6', inplace=True)

    adata1 = sc.read_visium(path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata1.var_names_make_unique()
    obs_names = np.array(adata1.obs.index)
    positions = adata1.obsm['spatial']

    data = np.delete(adata1.X.toarray(), NA_labels, axis=0)
    obs_names = np.delete(obs_names, NA_labels, axis=0)
    positions = np.delete(positions, NA_labels, axis=0)

    adata = ad.AnnData(pd.DataFrame(data, index=obs_names, columns=np.array(adata1.var.index), dtype=np.float32))

    adata.var_names_make_unique()
    adata.obs['ground_truth'] = labels
    adata.obs['ground'] = ground
    adata.obsm['spatial'] = positions
    adata.obs['array_row'] = adata1.obs['array_row']
    adata.obs['array_col'] = adata1.obs['array_col']
    adata.uns['spatial'] = adata1.uns['spatial']
    adata.var['gene_ids'] = adata1.var['gene_ids']
    adata.var['feature_types'] = adata1.var['feature_types']
    adata.var['genome'] = adata1.var['genome']
    adata.var_names_make_unique()
    adata = normalize(adata, highly_genes=highly_genes)
    fadj = features_construct_graph(adata.X, k=k)
    sadj, graph_nei, graph_neg = spatial_construct_graph1(adata, radius=radius)

    
    image_data = pd.read_csv(f"{path}features_512.csv", header=0, index_col=0)
    #adata = adata[adata.obs_names.isin(image_data.index.tolist())]
    image_data = image_data.reindex(adata.obs_names)

    image_data['feature_vector'] = image_data['feature_vector'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else None
    )

    sample_vec = next(v for v in image_data['feature_vector'] if v is not None)
    dim = len(sample_vec)

    image_data['feature_vector'] = image_data['feature_vector'].apply(
        lambda x: x if x is not None else [0]*dim
    )

    image_data_new = pd.DataFrame(
        image_data['feature_vector'].tolist(),
        index=image_data.index
    )

    image_data_new = image_data_new.fillna(0)

    adata.obsm['image_feature'] = image_data_new.values

    # ===== 标准化 + PCA =====
    scaler = StandardScaler()
    embedding = scaler.fit_transform(image_data_new.values)

    pca = PCA(n_components=16, random_state=42)
    embedding = pca.fit_transform(embedding)

    adata.obsm['img_emb'] = embedding
    adata.obsm['img_emb'] = embedding
    adata.obsm["fadj"] = fadj
    adata.obsm["sadj"] = sadj
    adata.obsm["graph_nei"] = graph_nei.numpy()
    adata.obsm["graph_neg"] = graph_neg.numpy()
    adata.var_names_make_unique()
    return adata

def calculate_edge_weights(adata):

    graph_neigh = adata.obsm['graph_nei']
    node_emb = adata.obsm['img_emb']
    edge_weights = np.zeros_like(graph_neigh)  

    euclidean_distances = cdist(node_emb, node_emb, metric='cosine')
    edge_weights = np.where(graph_neigh == 1, euclidean_distances, 0)


    edge_probabilities = np.zeros_like(edge_weights)
    for i in range(edge_weights.shape[0]):
        non_zero_indices = edge_weights[i] != 0
        non_zero_weights = edge_weights[i][non_zero_indices]
        softmax_weights = softmax(non_zero_weights)  
        edge_probabilities[i][non_zero_indices] = softmax_weights

    adata.obsm['edge_probabilities'] = edge_probabilities


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datasets = ['151507', '151508', '151509', '151510', '151669', '151670',
                '151671', '151672', '151673', '151674', '151675', '151676']
    #datasets = ['151507']
    for i in range(len(datasets)):
        dataset = datasets[i]
        print(dataset)
        if not os.path.exists("./generate_data/DLPFC/"):
            os.mkdir("./generate_data/DLPFC/")
        savepath = "./generate_data/DLPFC/" + dataset + "/"
        config_file = './config/DLPFC.ini'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        config = Config(config_file)
        adata = load_ST_file(dataset, config.fdim, config.k, config.radius)
        calculate_edge_weights(adata)
        print("saving")
        adata.write(savepath + 'MorphST.h5ad')
        print("done")
        print(adata.X)




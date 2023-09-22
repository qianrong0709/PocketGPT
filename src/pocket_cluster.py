"""
author: qian rong
date: 2023/09/22
parp1 PDB晶体结构的口袋聚类分析
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import umap
# import hdbscan  # 聚类方法，需要安装包
from sklearn.preprocessing import StandardScaler # 归一化
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from collections import Counter



# 数据处理，得到umap降维后的embedding
def data_process(path):
    df = pd.red_csv(path)
    df_data = df.drop(['smiles', 'file', 'LogS', 'logp', 'qed', 'tpsa', 'sas'], axis=1)
    print(df_data.values)
    # 数据归一化
    data_scaled = StandardScaler().fit_transform(df_data)
    embedding = umap.UMAP(n_neighbors=4, min_dist=0.01, metric='cosine',random_state=42).fit_transform(data_scaled)
    return embedding

# 保存图片
def save_picture(embedding):
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], cmap='Set2')
    plt.title('UMAP Visualization with Data Point Indices')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig(f'pocket_cluster.png', dpi=300)
    plt.show()

# 层次聚类
def agg_cluster(embedding):
    agg_clustering = AgglomerativeClustering(n_clusters=5)
    agg_clusters = agg_clustering.fit_predict(embedding)
    # 找到每个簇的原始向量
    clustered_vectors = {}
    for i, label in enumerate(agg_clusters):
        if label not in clustered_vectors:
            clustered_vectors[label] = []
        clustered_vectors[label].append(embedding[i])

    # 输出每个簇中的原始向量
    for label, vectors in clustered_vectors.items():
        print(f'Cluster {label} has {len(vectors)} vectors:')
        for vector in vectors:
            print(vector)
    plt.scatter(embedding[:, 0], embedding[:, 1], s=20, c=agg_clusters, cmap='Set2')
    plt.title('UMAP with Hierarchical Clustering')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.show()

    dendrogram = sch.dendrogram(sch.linkage(embedding, method='ward'))
    plt.title('Dendrogram')
    plt.xlabel('Data Points')
    plt.ylabel('Euclidean Distance')
    plt.show()

# HDBSCAN 聚类
def HDBscan(embedding):
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5)
    hdbscan_clusters = hdbscan_clusterer.fit_predict(embedding)
    # 找到每个簇的原始向量
    clustered_vectors = {}
    for i, label in enumerate(hdbscan_clusters):
        if label not in clustered_vectors:
            clustered_vectors[label] = []
        clustered_vectors[label].append(embedding[i])

    # 输出每个簇中的原始向量
    for label, vectors in clustered_vectors.items():
        print(f'Cluster {label} has {len(vectors)} vectors:')
        for vector in vectors:
            print(vector)
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=hdbscan_clusters, cmap='viridis', alpha=0.85)
    plt.title('Pocket cluster with HDBSCAN Clustering')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig('Pocket_cluster.png', dpi=300)


if __name__ == '__main__':
    path = '/Users/qianrong/Desktop/myProject/parp1/pdb_unprocess/parp1_pdb_final.csv'
    embedding = data_process(path)
    HDBscan(embedding)

"""
author: qian rong
date: 2023/09/19
生成分子分析
"""


from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import umap

from sklearn.metrics import pairwise_distances
import seaborn as sns


# 先对生成数据进行处理,按组别获得标签
def data_process(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates('smiles', keep='first')

    grouped = df.groupby('condition', group_keys=False)
    group_labels = {}
    for i, (group_name, _) in enumerate(grouped):
        group_labels[group_name] = i + 1
    df['label'] = df['condition'].map(group_labels)
    result = grouped.apply(lambda x: x.head(10000))

    return result


# 对对比数据进行处理,按组别获得标签
def contrast_data_process(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates('smiles',keep='first')

    return df


# 获得分子morgan指纹
def morgan_finger(data):
    data['mol'] = data['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    data['ecfp'] = data['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048))
    morgan = data['ecfp'].to_list()

    return morgan


# 计算指纹的相似度
def tanimoto(fp1, fp2):
    tanimoto_similarity = 1 - pairwise_distances(fp1, fp2, metric='jaccard')

    return tanimoto_similarity

# tsne降维
def tsne_reduce(data, n_components=2, perplexity=30, learning_rate=200.0, n_iter=1000, random_state=42):
    tsne = TSNE(n_components=2, random_state=42)
    embedding = tsne.fit_transform(data)
    return embedding

# umap降维
def umap_reduce(data):
    reducer = umap.UMAP(n_neighbors=15,min_dist=0.3,metric='euclidean',random_state=42)
    embedding = reducer.fit_transform(data)
    return embedding


def get_distance_matrix(self):
    if self.scaffold_cluster_flag:
        fps = self.scaffolds_fps
    else:
        fps = self.molecules_fps
    distance_matrix = []
    for i, fp in enumerate(fps):
        dist = DataStructs.BulkTanimotoSimilarity(fps[i], fps, returnDistance=1)
        distance_matrix.append(dist)
    # logg.info(f"get_distance_matrix, {len(distance_matrix)}")
    # logg.info(f"distance_matrix: {distance_matrix[0][0:10]}")
    return distance_matrix


if __name__ == '__main__':
    gen_path_more = '/home/wang/qianrong/final/src/test/parp1_v1/parp1_gen_second_more.csv'
    # gen_path_first = '/home/wang/qianrong/final/src/test/parp1_v1/parp1_gen_first_finetune.csv'
    # gen_path_59 = '/home/wang/qianrong/final/src/test/parp1_v1/parp1_gen_second_59.csv'

    # path_59 = '/home/wang/qianrong/final/data/parp1_finetune/parp1_pdb_final.csv'
    path_drug = '/home/wang/qianrong/final/data/parp1_finetune/drugbank_parp1_final.csv'
    # path_more = '/home/wang/qianrong/final/data/parp1_finetune/parp1_500_final.csv'

    gen_data = data_process(gen_path_more)
    drug_data = contrast_data_process(path_drug)

    gen_fps = morgan_finger(gen_data)
    drug_fps = morgan_finger(drug_data)

    distance_matrix = []
    for i, fp_gen in enumerate(drug_fps):
        # for j, fp_drug in enumerate(drug_fps):
        dist = DataStructs.BulkTanimotoSimilarity(drug_fps[i], drug_fps, returnDistance=1)
        print(dist)
        distance_matrix.append(dist)
        print(distance_matrix)
        # tsne = TSNE(n_components=2, perplexity=30, random_state=0)
        # embedded_clusters = tsne.fit_transform(distance_matrix)


























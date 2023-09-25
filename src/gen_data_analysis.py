"""
@author: qian rong
@date: 2023/09/19
生成分子分析
"""

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

# from sklearn.preprocessing import StandardScaler   # morgan指纹不需要归一化
# from sklearn.metrics import pairwise_distances
# import seaborn as sns
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# 对生成csv文件进行处理，得到smiles和morgan指纹
def gen_data_process(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates('smiles', keep='first')
    grouped = df.groupby('condition', group_keys=False)
    group_df = grouped.apply(lambda x: x.head(2000))     # 每个口袋取10000个分子，12个口袋，120000个生成分子
    group_df['mol'] = group_df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    group_df['ecfp'] = group_df['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048))
    morgan = group_df['ecfp'].to_list()
    return morgan

# 对原始csv文件进行处理，得到smiles和morgan指纹
def original_data_process(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates('smiles',keep='first')
    df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    df['ecfp'] = df['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048))
    morgan = df['ecfp'].to_list()
    return morgan

# tsne降维
def tsne_reduce(data):
    embedding = TSNE(n_components=2, perplexity=30, metric='jaccard', learning_rate=200.0, random_state=42).fit_transform(data)
    return embedding

# umap降维
def umap_reduce(data):
    embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='jaccard', random_state=42).fit_transform(data)
    return embedding

# tanimoto相似性矩阵计算
def get_distance_matrix(fps:list):
    distance_matrix = []
    for i,fp in enumerate(fps):
        dist = DataStructs.BulkTanimotoSimilarity(fps[i], fps, returnDistance=1)
        distance_matrix.append(dist)
    return distance_matrix

# 画散点图
def save_figure(embedding, ori_len, gen_len):
    fig, ax = plt.subplots(figsize=(14, 10))
    plt.xlim([np.min(embedding[:, 0]) - 0.5, np.max(embedding[:, 0]) + 1.5])
    plt.ylim([np.min(embedding[:, 1]) - 0.5, np.max(embedding[:, 1]) + 0.5])
    plt.xlabel('UMAP 1', fontsize=16)
    plt.ylabel('UMAP 2', fontsize=16)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.scatter(embedding[:ori_len, 0], embedding[:ori_len, 1], lw=0, c='blue', label='original space', alpha=0.85, s=50)
    plt.scatter(embedding[ori_len:gen_len, 0], embedding[ori_len:gen_len, 1], lw=0, c='red', label='generate space', alpha=0.85, s=5)
    leg = plt.legend(prop={'size': 16}, loc='upper right', markerscale=1.00)
    leg.get_frame().set_alpha(0.9)
    plt.setp(ax, xticks=[], yticks=[])
    plt.savefig(f'test1.png', dpi=150)
    plt.show()
    return None

def main(path1, path2):
    ori_fps = original_data_process(path1)
    gen_fps = gen_data_process(path2)
    ori_len = len(ori_fps)
    gen_len = len(gen_fps)
    all_fps =  ori_fps + gen_fps      # 注意顺序，先是原始分子，再是生成分子

    similarity_matrix = get_distance_matrix(all_fps)
    embedding = umap_reduce(similarity_matrix)
    # embedding = tsne_reduce(similarity_matrix)
    save_figure(embedding, ori_len,gen_len)


if __name__ == '__main__':
    # 先不做
    # path_59 = '/home/wang/qianrong/final/data/parp1_finetune/parp1_pdb_final.csv'

    # path_drug = '/home/wang/qianrong/final/data/parp1_finetune/drugbank_parp1_final.csv'
    path_more = '/home/wang/qianrong/final/data/parp1_finetune/parp1_500_final.csv'

    # 生成分子
    gen_path_first = '/home/wang/qianrong/final/src/test/parp1_v1/parp1_gen_first_finetune.csv'
    # gen_path_59 = '/home/wang/qianrong/final/src/test/parp1_v1/parp1_gen_second_59.csv'
    # gen_path_more = '/home/wang/qianrong/final/src/test/parp1_v1/parp1_gen_second_more.csv'

    # main(path_drug,gen_path_first)
    main(path_more,gen_path_first)

    # main(path_drug,gen_path_59)
    # main(path_more,gen_path_59)

    # main(path_drug,gen_path_more)
    # main(path_more,gen_path_more)
































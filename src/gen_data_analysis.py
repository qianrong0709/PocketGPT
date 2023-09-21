"""
author: qian rong
date: 2023/09/19
生成分子分析
"""

from rdkit import Chem, DataStructs
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler

# from sklearn.metrics import pairwise_distances
# import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



# 对生成csv文件进行处理，得到smiles和morgan指纹
def gen_data_process(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates('smiles', keep='first')
    grouped = df.groupby('condition', group_keys=False)
    group_df = grouped.apply(lambda x: x.head(10000))     # 每个口袋取10000个分子，12个口袋，120000个生成分子
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
    embedding = TSNE(n_components=2, perplexity=30, metric='correlation', learning_rate=200.0, random_state=42).fit_transform(data)
    return embedding


# umap降维
def umap_reduce(data):
    embedding = umap.UMAP(n_neighbors=15, min_dist=0.3, metric='jaccard', random_state=42).fit_transform(data)
    return embedding


def get_distance_matrix(fps:list):
    distance_matrix = []
    for i,fp in enumerate(fps):
        dist = DataStructs.BulkTanimotoSimilarity(fps[i], fps, returnDistance=1)
        distance_matrix.append(dist)
    return distance_matrix

def main(path1, path2):
    gen_fps = gen_data_process(path1)
    drug_fps = original_data_process(path2)
    gen_len = len(gen_fps)
    drug_len = len(drug_fps)
    all_fps = gen_fps + drug_fps
    similarity_matrix = get_distance_matrix(all_fps)
    print(similarity_matrix)

    # embedding = umap_reduce(similarity_matrix)
    # # embedding = tsne_reduce(all_fps)
    #
    # plt.scatter(embedding[:drug_len, 0], embedding[:drug_len, 1], lw=0, c='blue', label='gen space', alpha=0.85, s=50)
    # plt.scatter(embedding[drug_len:gen_len, 0], embedding[drug_len:gen_len, 1], lw=0, c='red', label='drug space', alpha=0.85, s=2)
    # plt.show()
    # plt.savefig(f'{save_path}umap_{temp}.png', dpi=150)



if __name__ == '__main__':
    gen_path_more = '/home/wang/qianrong/final/src/test/parp1_v1/parp1_gen_second_more.csv'
    # gen_path_first = '/home/wang/qianrong/final/src/test/parp1_v1/parp1_gen_first_finetune.csv'
    # gen_path_59 = '/home/wang/qianrong/final/src/test/parp1_v1/parp1_gen_second_59.csv'

    # path_59 = '/home/wang/qianrong/final/data/parp1_finetune/parp1_pdb_final.csv'
    path_drug = '/home/wang/qianrong/final/data/parp1_finetune/drugbank_parp1_final.csv'
    # path_drug = f'../data/parp1_finetune/drugbank_parp1_final.csv'
    # path_more = '/home/wang/qianrong/final/data/parp1_finetune/parp1_500_final.csv'

    main(gen_path_more,path_drug)



    # smiles1 = 'Cn1c(=O)c2[nH]cnc2n(C)c1=O'
    # smiles2 = ['Cc1nc(=O)c2cccc(O)c2[nH]1','Nc1ccc2c3c(cccc13)C(=O)NC2=O']
    #
    # mol1 = Chem.MolFromSmiles(smiles1)
    # mol2 = [Chem.MolFromSmiles(smiles) for smiles in smiles2]
    #
    # fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    # fp2 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mol2]
    #
    # fp2.append(fp1)
    # # fps = np.array(fp2)
    # # print(type(fps))
    # # distance_matrix = []
    # for i, fp in enumerate(fp2):
    # # # similarity = TanimotoSimilarity(fp1, fp2)
    #     print(fp2[i])
    #     dist = DataStructs.BulkTanimotoSimilarity(fp2[i], fp2, returnDistance=1)
    # #     # distance_matrix.append(dist)
    #     print(dist)





























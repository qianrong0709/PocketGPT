"""
@author: qian rong
@date: 2023/10/03

"""


import pandas as pd
from rdkit.Chem import PandasTools
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs



def file_process(path):
    df = pd.read_csv(path, delimiter='\t', header=None)
    df['ttt'] = df.apply(lambda x: x.str.split(':'))
    df['frag'] = df['ttt'].apply(lambda x: x[0])
    df['count'] = df['ttt'].apply(lambda x: x[1])
    df = df.drop('ttt', axis=1)
    # print(df)
    return df

def analysis(df1,df2):
    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df['count'] = merged_df['count'].astype(int)   # 不转换成int，sum的时候会变成拼接
    # merged_df.to_csv('merge3.csv',index=False)
    result_df = merged_df.groupby('frag', as_index=False)['count'].sum()
    # 降序排列，重置索引
    result_df = result_df.sort_values(by='count', ascending=False)
    result_df = result_df.reset_index(drop=True)
    print(result_df)
    result_df.to_csv('merge3.csv', float_format='%.10f',index=False)  # 防止数字变科学计数
    return result_df

def moltoImage(path):
    df = pd.read_csv(path)
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='smiles',
                                         molCol='ROMol')  ##smilesCol  simles格式列名称   molCol新增图像的列名称
    PandasTools.SaveXlsxFromFrame(df, 'sample_5.xlsx', molCol="ROMol")


def substruct(path):
    df = pd.read_excel(path)
    df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    molecules = df['mol'].tolist()

    # 准备已知子结构的分子对象
    known_substructure = Chem.MolFromSmiles('O=c1cccn[nH]1')  # 你的已知子结构的SMILES表示法

    # 准备一个空列表用于存储包含已知子结构的分子
    molecules_with_substructure = []

    # 准备一个空字典用于存储分子与相似分子的对应关系
    similarity_clusters = {}

    # 遍历2000多个分子
    for mol in molecules:
        if mol.HasSubstructMatch(known_substructure):
            molecules_with_substructure.append(mol)
        else:
            # 计算分子与已知子结构的相似性
            fp1 = AllChem.GetMorganFingerprint(mol, radius=2)
            fp2 = AllChem.GetMorganFingerprint(known_substructure, radius=2)
            # fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            # fp2 = AllChem.GetMorganFingerprintAsBitVect(known_substructure, radius=2, nBits=2048)
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

            # 将分子添加到相似性聚类中
            if similarity in similarity_clusters:
                similarity_clusters[similarity].append(mol)
            else:
                similarity_clusters[similarity] = [mol]

    # 在这里，你可以使用相似性聚类算法对分子进行聚类，具体方法取决于你的需求。

    # 输出包含已知子结构的分子
    print("分子中包含已知子结构的数量:", len(molecules_with_substructure))
    print(len(similarity_clusters))
    print(sorted(similarity_clusters.keys()))
    return None




if __name__ == '__main__':
    # path1 = 'frag_vocab.txt'
    # path2 = 'parp1_frag_vocab.txt'
    # path3 = 'deeply_frag_vocab.txt'
    # df1 = file_process(path1)
    # df2 = file_process(path2)
    # merge_df = analysis(df1, df2)
    #
    # df3 = file_process(path3)
    # final_df = analysis(merge_df, df3)

    # 对生成的片段文件进行可视化处理
    # path = 'merge_frag.csv'
    # path = 'frag_molgpt/frag_test_5.csv'
    # moltoImage(path)

    path = '/home/qianrong/final/src/parp1_vs/vs_parp1.xlsx'
    res = substruct(path)



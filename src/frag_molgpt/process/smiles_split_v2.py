"""
@author: qian rong
@date: 2023/10/17
切分smiles的方法
"""

# from rdkit.Chem import Recap
# from rdkit.Chem import AllChem as Chem
#
# m = Chem.MolFromSmiles('N=C(N)c1ccc(CNC(=O)Cn2c(-c3ccc(-c4ccccc4)cc3)c(Cl)nc(NCCc3ccccc3)c2=O)cc1')
# hierarch = Recap.RecapDecompose(m)
#
# # 叶子节点函数：hierarch.GetLeaves()
# print(hierarch.GetLeaves().keys())
#
# # 子孙节点函数：hierarch.GetAllChildren()
# print(hierarch.GetAllChildren().keys())
#
# # 祖先节点函数，返回列表：getUltimateParents()
# print(hierarch.getUltimateParents()[0].smiles)


from rdkit.Chem import BRICS
from rdkit import Chem
import re
import pandas as pd
from multiprocessing import Pool


def mol_split(smiles):
    modified_smi = []
    try:
        mol = Chem.MolFromSmiles(smiles)
        fragments = BRICS.BRICSDecompose(mol, allNodes=None, minFragmentSize=1, onlyUseReactions=None,
                                         silent=True, keepNonLeafNodes=True, singlePass=True, returnMols=False)

        # modified_smi = [re.sub(r'\[\d+\*\]', '', i) for i in fragments]
        for i in fragments:
            if '*' not in i:
                continue
            else:
                mol = Chem.MolFromSmiles(i)
                num = mol.GetNumAtoms()
                if num >= 10:
                    f = re.sub(r'\[\d+\*\]', '', i)
                    modified_smi.append(f)
                else:
                    continue
    except Exception as e:
        print (e, smiles)

    return modified_smi



if __name__ == '__main__':
    ##############################Test##############################

    # smiles = 'O=C(NCCNc1ncccn1)N1CC=C(c2ccccc2)CC1'
    # res = mol_split(smiles)
    # print(res)

    ##################################################################
    """
    对gua.csv进行片段切分
    v1版本：切出来的所有片段都要        27661908
    v2版本：原子数小于10的小片段不要    19108303
    
    对gua_moses_merge.csv进行片段切分
    v1版本：切出来的所有片段都要        27513212   27453733(删除空值)
    v2版本：原子数小于10的小片段不要    18988771   18915277(删除空值)   18915264(删除 %)
    
    """
    path = 'deeply_all_duplicate_v1.csv'
    path = ''
    df = pd.read_csv(path)
    smiles_data = df['smiles'].tolist()
    with Pool(processes=64) as pool:
        results = pool.map(mol_split, smiles_data)
    df['frag'] = results
    df = df.dropna(subset=['frag'])
    df_exploded = df.explode('frag').reset_index(drop=True)
    df_exploded = df_exploded.dropna(subset=['frag'])
    df_exploded.to_csv('deeply_all_duplicate_v1_one_frag.csv', index=False)

    ##################################################################
    """
    对切分完的文件进行片段分析
    v1版本：片段共有5557676个
    v2版本：片段共有5531234个
    
    对gua_moses_merge.csv切分完的文件进行片段分析
    v1版本：片段共有5536017个    
    v2版本：片段共有5508789个
    
    """
    # path = 'gua_merge_frag_final_v2.csv'
    # df  = pd.read_csv(path)
    # # frag_list = set(df['frag'].tolist())
    # frag_counts = df['frag'].value_counts()
    # print(frag_counts)
    # # frag_counts.reset_index().to_csv('count_merge_result.csv', header=['frag', 'count'], index=False)
    # print('finish!')

    ##################################################################
    """
    查看文件
    """
    # path = 'gua_merge_frag_final_v2.csv'
    # df = pd.read_csv(path)
    # print(df)

    ##################################################################
    """
    整理文件
    """
    # path = 'gua_merge_frag_final_v2.csv'
    # df = pd.read_csv(path)
    # df = df[~df['frag'].str.contains('%')]
    # print(df)
    # df.to_csv('gua_merge_frag_final_v2.csv')





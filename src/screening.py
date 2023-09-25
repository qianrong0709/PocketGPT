"""
@author: qian rong
@date: 2023/09/24
parp1分子库进行筛选
"""

import pandas as pd
from rdkit import Chem

# import sys
# from rdkit import RDConfig
# from rdkit.Chem import AllChem as Chem
# from rdkit.Chem import Descriptors, Lipinski, QED
# from rdkit.Chem.Scaffolds import MurckoScaffold
# sys.path.append(RDConfig.RDContribDir)

from scopy.ScoDruglikeness import PC_rules
from scopy.ScoDruglikeness import PC_properties
from scopy.ScoFH import FHfilter
from scopy.ScoTox import Toxfilter

from rdkit.Chem.Scaffolds import MurckoScaffold


# 读取文件，返回smiles
def read_csv(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates('smiles',keep='first')
    smiles = df['smiles']

    return smiles

# 对合并后的所有smiles进行过滤
def scopy_filer(mols):
    """
    # 使用以下RO5规则logP < 5, MW < 500, HBA < 10， HBD < 5, and NRB < 10（类药性）
    # 使用警报子结构去除潜在的含有毒性和不良基团的化合物
    # 计算分子的可合成性得分  (<3.5)
    # 混乱化合物筛选
    # 计算分子的综合得分
    # ROA小鼠急性毒性 （>0.5)
    # 使用PARP分类模型对化合物的活性进行排序，选择被模型预测为抑制剂的分子 （>0.5)
    # 使用PARP回归模型，logS模型预测分子的PARP活性和水溶性，pki > 7.2,logS < -3.6
    # 重打分，对接模型

    """
    rules = PC_rules(mols, n_jobs=100, detail=True)
    fh = FHfilter(mols, n_jobs=100, detail=True)
    pc = PC_properties(mols, n_jobs=100)
    tox = Toxfilter(mols, n_jobs=100)


    summary_fh = pd.DataFrame(
        {
            'smiles': [Chem.MolToSmiles(mol) for mol in mols],
            # 'lipinski': [item['Disposed'] for item in rules.CheckLipinskiRule()],
            'Acute': [item['Disposed'] for item in tox.Check_Acute_Aquatic_Toxicity()],
            'sa': pc.CalculateSAscore(),
            'pains': [item['Disposed'] for item in fh.Check_PAINS()],
            'bRo5': [item['Disposed'] for item in rules.CheckBeyondRo5()]
        }
    )

    # 去除SA大于3的分子
    print(len(summary_fh))
    summary_fh = summary_fh[summary_fh['sa'] <= 3]
    print(len(summary_fh))
    # 去除不符合上述条件的分子
    summary_fh = summary_fh[~summary_fh.apply(lambda row: row.str.contains('Rejected').any(), axis=1)]
    print(len(summary_fh))
    return summary_fh

def murcko_filter(df):
    df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    df['scaffolds'] = df['mol'].apply(lambda x: MurckoScaffold.GetScaffoldForMol(x))
    df['scaffold_smiles'] = df['scaffolds'].apply(lambda x: Chem.MolToSmiles(x))

    # scaffold_count = data.groupby(["scaffold_smiles"]).count()
    # scaffold = pd.DataFrame({"murcko": scaffold_count.index, "count": scaffold_count[self.smiles_field]})
    # print('scaffold-len:', len(scaffold))





if __name__ == '__main__':
    path1 = f'test/parp1_v1/parp1_gen_first_finetune.csv'
    path2 = f'test/parp1_v1/parp1_gen_second_59.csv'
    path3 = f'test/parp1_v1/parp1_gen_second_more.csv'

    path_59 = '../data/parp1_finetune/parp1_pdb_final.csv'
    path_drug = '../data/parp1_finetune/drugbank_parp1_final.csv'
    path_more = '../data/parp1_finetune/parp1_500_final.csv'

    pre_train = '../gua_moses_merge.csv'

    smiles1 = read_csv(path1)
    smiles2 = read_csv(path2)
    smiles3 = read_csv(path3)
    gen_df = pd.concat([smiles1,smiles2,smiles3], ignore_index=True)
    gen_df = gen_df.drop_duplicates()   # 1944524

    ori_smiles1 = read_csv(path_59)
    ori_smiles2  = read_csv(path_drug)
    ori_smiles3 = read_csv(path_more)
    ori_df = pd.concat([ori_smiles1,ori_smiles2,ori_smiles3],ignore_index=True)
    ori_df = ori_df.drop_duplicates()   # 3338

    pre_smiles = read_csv(pre_train)
    pre_df = pre_smiles.drop_duplicates()

    merge_df = pd.merge(gen_df,ori_df,how='inner')         # 580
    gen_df = gen_df[~gen_df.index.isin(merge_df.index)]    # 1943944

    merge_df2 = pd.merge(gen_df,pre_df,how='inner')        # 13821
    gen_all_df = gen_df[~gen_df.index.isin(merge_df2.index)]   # 1930703
    # gen_all_df.to_csv('p.csv',index=False)


    gen_all_df['mol'] = gen_all_df.apply(lambda x: Chem.MolFromSmiles(x))
    mol_list = gen_all_df['mol'].tolist()
    res_df = scopy_filer(mol_list)
    res_df.to_csv('scopy_filter_v3.csv', index=False)



    # res = murcko_filter(ori_df)




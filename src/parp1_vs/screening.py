"""
@author: qian rong
@date: 2023/09/24
parp1分子库进行筛选
"""

import pickle
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs

# import sys
# from rdkit import RDConfig
# from rdkit.Chem import AllChem
# from rdkit.Chem import Descriptors, Lipinski, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
# sys.path.append(RDConfig.RDContribDir)

from scopy.ScoDruglikeness import PC_rules
from scopy.ScoDruglikeness import PC_properties
from scopy.ScoFH import FHfilter
from scopy.ScoTox import Toxfilter

from rdkit import Chem, RDLogger
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import MolStandardize
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')

from multiprocessing import Pool   # 多进程


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

    # scaffold_count = df.groupby(["scaffold_smiles"]).count()
    # scaffold = pd.DataFrame({"murcko": scaffold_count.index, "count": scaffold_count[self.smiles_field]})
    # print('scaffold-len:', len(scaffold))
    return df

def cluster_saffold(mol_list):
    scaffolds = {}
    clusters_list = []

    idx = 1
    for mol in mol_list:
        scaffold_smi = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if scaffold_smi not in scaffolds.keys():
            scaffolds[scaffold_smi] = idx
            idx += 1

        cluster_id = scaffolds[scaffold_smi]
        clusters_list.append(cluster_id)

def standardize_and_get_scaffold(smiles,basicClean=True,clearCharge=True, clearFrag=True, canonTautomer=True, isomeric=False):
    try:
        clean_mol = Chem.MolFromSmiles(smiles)
        # 除去氢、金属原子、标准化分子
        if basicClean:
            clean_mol = rdMolStandardize.Cleanup(clean_mol)
        if clearFrag:
        #  仅保留主要片段作为分子
            clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        # 尝试中性化处理分子
        if clearCharge:
            uncharger = rdMolStandardize.Uncharger()
            clean_mol = uncharger.uncharge(clean_mol)
        # 处理互变异构情形，这一步在某些情况下可能不够完美
        if canonTautomer:
            te = rdMolStandardize.TautomerEnumerator() # idem
            clean_mol = te.Canonicalize(clean_mol)
        #set to True 保存立体信息，set to False 移除立体信息，并将分子存为标准化后的SMILES形式
        stan_smiles=Chem.MolToSmiles(clean_mol, isomericSmiles=isomeric)
        new_mol = Chem.MolFromSmiles(stan_smiles)
        scaffold = MurckoScaffold.GetScaffoldForMol(new_mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
    except Exception as e:
        print (e, smiles)
        return None
    return stan_smiles, scaffold_smiles



if __name__ == '__main__':
    # path1 = f'test/parp1_v1/parp1_gen_first_finetune.csv'
    # path2 = f'test/parp1_v1/parp1_gen_second_59.csv'
    # path3 = f'test/parp1_v1/parp1_gen_second_more.csv'

    # path_59 = '../file/parp1_finetune/parp1_pdb_final.csv'
    # path_drug = '../file/parp1_finetune/drugbank_parp1_final.csv'
    # path_more = '../file/parp1_finetune/parp1_500_final.csv'

    # pre_train = '../gua_moses_merge.csv'

    ####################################################################################################
    # 过滤生成分子
    ####################################################################################################

    # smiles1 = read_csv(path1)
    # smiles2 = read_csv(path2)
    # smiles3 = read_csv(path3)
    # gen_df = pd.concat([smiles1,smiles2,smiles3], ignore_index=True)
    # gen_df = gen_df.drop_duplicates()   # 1944524

    # ori_smiles1 = read_csv(path_59)
    # ori_smiles2  = read_csv(path_drug)
    # ori_smiles3 = read_csv(path_more)
    # ori_df = pd.concat([ori_smiles1,ori_smiles2,ori_smiles3],ignore_index=True)
    # ori_df = pd.DataFrame(ori_df.drop_duplicates())  # 3338

    # pre_smiles = read_csv(pre_train)
    # pre_df = pre_smiles.drop_duplicates()

    # merge_df = pd.merge(gen_df,ori_df,how='inner')         # 580
    # gen_df = gen_df[~gen_df.index.isin(merge_df.index)]    # 1943944
    #
    # merge_df2 = pd.merge(gen_df,pre_df,how='inner')        # 13821
    # gen_all_df = gen_df[~gen_df.index.isin(merge_df2.index)]   # 1930703
    # # gen_all_df.to_csv('p.csv',index=False)

    # gen_all_df['mol'] = gen_all_df.apply(lambda x: Chem.MolFromSmiles(x))
    # mol_list = gen_all_df['mol'].tolist()
    # res_df = scopy_filer(mol_list)
    # res_df.to_csv('scopy_filter_v3.csv', index=False)

    ####################################################################################################
    # 过滤后的分子转换成smi文件
    # df = pd.read_csv(path_filter,sep=' ',header=None)
    # df_smiles = df['smiles']
    # output_file = 'file/output.smi'
    # with open(output_file, 'w') as smi_file:
    #     for smiles in df_smiles:
    #         smi_file.write(smiles + '\n')
    ####################################################################################################




    ####################################################################################################
    # 骨架分析
    ####################################################################################################

    """
    # 旧版分子骨架提取
    # res = murcko_filter(ori_df)   # 3338
    # res = res.drop_duplicates('scaffold_smiles')
    # res.to_csv('scaffold_v1.csv',index=False)   # 1577
    # filter_smiles = pd.DataFrame(df,columns=['smiles','source'])  # 920785         46033
    # filter_scaffold = murcko_filter(df)
    # filter_scaffold = filter_scaffold.drop_duplicates('scaffold_smiles') # 534603    29363
    # print(filter_scaffold)
    # filter_scaffold.to_csv('file/filter_scaffold.csv',index=False)
    """


    # 过滤VS之后的分子骨架
    # path_filter = 'file/scopy_filter_v3.csv'
    # path_filter = 'file/results.smi'   # 虚拟筛选之后得到的分子数量 46034
    # df = pd.read_csv(path_filter, sep=' ',header=None)
    # df.columns = ['smiles','source']

    # # （parp1活性小于500nm）source分子骨架 3322个
    # path_source = '/home/qianrong/final/data/parp1_finetune/parp1_500_final.csv'
    # df = pd.read_csv(path_source)

    # smiles_data = df['smiles'].tolist()
    # with Pool(processes=128) as pool:
    #     results = pool.map(standardize_and_get_scaffold, smiles_data)
    # standardized_smiles, scaffold_smiles = zip(*results)
    # df['Standardized_SMILES'] = standardized_smiles
    # df['Scaffold_SMILES'] = scaffold_smiles
    # # df.to_csv('file/vs_smiles_scaffold.csv', index=False)  # 输出到新的csv文件
    # df.to_csv('file/source_smiles_scaffold.csv', index=False)  # 输出到新的csv文件


    ####################################################################################################
    path1 = 'file/source_smiles_scaffold.csv'
    path2 = 'file/vs_smiles_scaffold.csv'
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    # df2 = df2.drop_duplicates('Standardized_SMILES')
    # print(df2)    # 45625

    # list1 = df1['Scaffold_SMILES'].drop_duplicates().tolist()
    # print(len(list1))   # 1562
    # list2 = df2['Scaffold_SMILES'].drop_duplicates().tolist()
    # print(len(list2))    # 28752

    # 使用groupby函数按'Scaffold_SMILES'列分组，并将'smiles'列的值合并到一个列表中
    df_grouped1 = df1.groupby('Scaffold_SMILES')['Standardized_SMILES'].apply(list)
    df_grouped2 = df2.groupby('Scaffold_SMILES')['Standardized_SMILES'].apply(list)
    # 将分组后的DataFrame转换为字典
    dictionary1 = df_grouped1.to_dict()
    dictionary2 = df_grouped2.to_dict()
    same_result = {}
    diff_result = {}
    for key in dictionary2.keys():
        if key in dictionary1:
            # same_result[key] = {'source': dictionary1[key], 'vs': dictionary2[key]}
            same_result[key] = dictionary2[key]
        else:
            diff_result[key] = dictionary2[key]
    print(len(same_result))   # 222 都有的共同骨架
    print(len(diff_result))   # 28530 生成的新骨架
    same_df = pd.DataFrame(list(same_result.items()),columns=['Scaffold_SMILES','Standardized_SMILES'])
    diff_df = pd.DataFrame(list(diff_result.items()),columns=['Scaffold_SMILES','Standardized_SMILES'])

    same_df_exploded = same_df.explode('Standardized_SMILES').reset_index(drop=True)
    diff_df_exploded = diff_df.explode('Standardized_SMILES').reset_index(drop=True)

    ####### 相同骨架的分子
    # same_df.to_csv('file/same_scaffold.csv', index=False)
    ####### 不同骨架的分子
    # diff_df.to_csv('file/diff_scaffold.csv', index=False)

    print(same_df_exploded)   # 2389 分子
    print(diff_df_exploded)   # 43645 分子




    ####################################################################################################
    # 骨架相似性的计算
    # smiles = 'c1ccc2c(c1)C(=NNC2=O)Cc3ccc(c(c3)C(=O)N4CCN(CC4)C(=O)C5CC5)F'    # Olaparib
    # scaff = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smiles))
    # scaff_smiles = Chem.MolToSmiles(scaff)
    # print(scaff_smiles)


















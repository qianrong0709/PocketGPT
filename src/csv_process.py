"""
@author:qian rong
@date:2023/07/12
multi-target processing

"""

import re
import os
import csv
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import MolStandardize


def processing(path):
    all_target = []
    df = pd.read_csv(path)
    with open('/Users/qianrong/Desktop/smiles-target.csv', 'w', encoding='utf-8', newline='') as cf:
        header = ['smiles', 'target_one', 'target_two']
        a = csv.writer(cf)
        a.writerow(header)
        for index, cont in df.iterrows():  # 读取每一行数据
            line = []
            smiles = cont['smiles']  # 获得每一行的smiles
            final_smiles = smiles_canonical(smiles)  # 对每一行的smiles进行标准化
            line.append(final_smiles)
            targets = cont['targets']  # 获取每一行的target
            final_target = target_process(targets)
            for each_target in final_target:
                all_target.append(each_target)
                line.append(each_target)
                a.writerow(line)
    return all_target


def smiles_canonical(path):
    print('开始计算')
    df = pd.read_csv(path)
    df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    df['mol2'] = df['mol'].apply(lambda x: MolStandardize.normalize.Normalizer().normalize(x))
    df['mol2'] = df['mol2'].apply(lambda x: MolStandardize.fragment.LargestFragmentChooser().choose(x))
    df['mol2'] = df['mol2'].apply(lambda x: MolStandardize.charge.Uncharger().uncharge(x))
    df['s'] = df['mol2'].apply(lambda x: Chem.MolToSmiles(x, isomericSmiles=False, canonical=True))
    df.to_csv('/home/wang/qianrong/final/gua_and_moses_test.csv', index=False)
    print('计算结束')
    return None


def target_process(targets):
    protein = []
    targets = targets[2:-2].split(',')
    for target in targets:
        target = target.replace("'", " ").strip(" ")
        protein.append(target)
    return protein


def regular_expression(string):  # 对smiles进一步处理，正则匹配｜｜之间的内容，并删除
    pattern = r"(?<=\|).+?(?=\|)"
    result = re.sub(pattern, '', string)
    smiles = result.replace('|', '').strip()  # 删除｜｜，去除空格
    return smiles



if __name__ == '__main__':
    root_path = '/home/wang/qianrong/final/gua_and_moses.csv'
    # target = processing(root_path)
    # print(target)
    smiles_canonical(root_path)

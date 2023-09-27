"""
@author: qian rong
@date: 2023/09/26
smiles切分
"""


import pandas as pd

from rdkit import Chem
from rdkit.Chem import BRICS

import os
from rdkit.Chem import Draw
from rdkit.Chem import RDConfig
from rdkit.Chem import FragmentCatalog

def read_csv(path):
    df = pd.read_csv(path)
    pass


def remove_dummy(smiles):
    try:
        stripped_smi=smiles.replace('*','[H]')
        mol=Chem.MolFromSmiles(stripped_smi)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        print (e)
        return None





if __name__ == '__main__':
    # path = ''


    # aspirin = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(O)=O')
    # fragments = BRICS.BRICSDecompose(aspirin, allNodes=None, minFragmentSize=1,
    #                                  onlyUseReactions=None, silent=True, keepNonLeafNodes=False, singlePass=False,
    #                                  returnMols=False)
    # print(fragments)
    # for i in fragments:
    #     mol = Chem.MolFromSmiles(i)



    # fragments = ['*C(C)=O', '*O*', '*c1ccccc1*', '*C(=O)O']
    # clean_fragments = [remove_dummy(smi) for smi in fragments]
    # print(clean_fragments)

    # 定义一个分子的SMILES表示
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)OCC"

    # 将SMILES表示转化为分子对象
    mol = Chem.MolFromSmiles(smiles)

    # 使用Chem.FragmentOnBonds函数进行自动分割
    fragments = Chem.FragmentOnBonds(mol,[0,1], addDummies=True)
    print(fragments)

    # 打印生成的碎片
    for fragment in fragments:
        fragment_smiles = Chem.MolToSmiles(fragment)
        print("Fragment SMILES:", fragment_smiles)

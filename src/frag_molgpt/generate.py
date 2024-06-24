from utils import check_novelty, sample, canonic_smiles
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit import Chem
import math
from tqdm import tqdm
import argparse
from model_1 import GPT, GPTConfig
import pandas as pd
import torch
import numpy as np
# from moses.utils import get_mol
import re
from rdkit.Chem import RDConfig
import json

import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

import sascorer
from rdkit.Chem.rdMolDescriptors import CalcTPSA


def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weight', type=str, help="path of model weights", required=False)
    parser.add_argument('--scaffold', action='store_true', default=True, help='condition on scaffold')
    parser.add_argument('--lstm', action='store_true', default=True, help='use lstm for transforming scaffold')
    parser.add_argument('--csv_name', type=str, help="name to save the generated mols in csv format", required=False)
    parser.add_argument('--batch_size', type=int, default=32, help="batch size", required=False)
    parser.add_argument('--gen_size', type=int, default=100, help="number of times to generate from a batch", required=False)
    parser.add_argument('--vocab_size', type=int, default=91, help="number of layers",   # 32 94
                        required=False)  # previously 28 .... 26 for moses. 94 for guacamol
    parser.add_argument('--block_size', type=int, default=100, help="number of layers",
                        required=False)  # previously 57... 54 for moses. 100 for guacamol.
    parser.add_argument('--num_props', type=int, default=0, help="number of properties to use for condition",   # coulomb
                        required=False)
    parser.add_argument('--n_layer', type=int, default=8, help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8, help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256, help="embedding dimension", required=False)
    parser.add_argument('--lstm_layers', type=int, default=1, help="number of layers in lstm", required=False)
    args = parser.parse_args()

    context = "C"
    # file = pd.read_csv('/home/wang/qianrong/final/file/parp1_finetune/parp1_pdb_final.csv')   # train 文件用来评价新颖性
    # file = pd.read_csv('/home/wang/qianrong/final/file/finetune/deeply_all_duplicate_v1.csv')
    # data = pd.read_csv('/home/wang/qianrong/final/file/parp1_finetune/parp1_500_train.csv')
    # file = pd.read_csv('/home/wang/qianrong/final/gua_moses_admet.csv')
    data = pd.read_csv('process/gua_merge_frag_final_v3.csv')


    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()


    # data_test = pd.read_csv('/home/wang/qianrong/final/file/parp1_finetune/parp1_pdb_cluster_5.csv')   # test文件的口袋条件用来生成分子
    # data_test = data_test.dropna(axis=0).reset_index(drop=True)
    # data_test.columns = data_test.columns.str.lower()


    smiles = data['smiles']  # train文件里的smiles用来评价新颖性

    # smiles_test = data_test['smiles']
    # prop = data_test.drop(['smiles','file','logp','qed','tpsa','sas'], axis=1).values.tolist()  # test文件里的条件用来输入模型
    # prop = data_test.drop(['smiles', 'file','logp','qed','tpsa'],axis=1).values.tolist()
    # prop = data_test.drop(['file'], axis=1).values.tolist()
    # print('条件维度:',len(prop[0]) ) # 二维的（1，128）


    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)


    stoi = json.load(open('vocab_stoi.json', 'r'))   # 更改词表
    itos = {i: ch for ch, i in stoi.items()}
    print(itos)
    print(len(itos))
    scaffold_max_len = 203    # 102


    num_props = args.num_props
    mconf = GPTConfig(args.vocab_size, args.block_size, num_props=num_props, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      scaffold=args.scaffold, scaffold_maxlen=scaffold_max_len, lstm=args.lstm, lstm_layers=args.lstm_layers)
    model = GPT(mconf)
    model.load_state_dict(torch.load('weight/pretrain_wang_two_v1.pt'))
    model.to('cuda')
    print('Model loaded')
    gen_iter = math.ceil(args.gen_size / args.batch_size)

    # # 无条件时的代码
    # prop_condition = None
    # all_dfs = []
    # all_metrics = []
    # count = 0
    # print("无条件")
    # if prop_condition is None:
    #     # for s in smiles:
    #     molecules = []
    #     count += 1
    #     for i in tqdm(range(gen_iter)):
    #         x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
    #         p = None
    #         y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=None, prop = p)   # 0.7 for guacamol
    #         for gen_mol in y:
    #             completion = ''.join([itos[int(i)] for i in gen_mol])
    #             completion = completion.replace('<', '')
    #             # gen_smiles.append(completion)
    #             mol = get_mol(completion)
    #             if mol:
    #                 molecules.append(mol)
    #
    #     "Valid molecules % = {}".format(len(molecules))
    #
    #     mol_dict = []
    #
    #     for i in molecules:
    #         mol_dict.append({'molecule' : i, 'smiles': Chem.MolToSmiles(i)})
    #     results = pd.DataFrame(mol_dict)
    #     # results['object'] = s
    #     canon_smiles = [canonic_smiles(s) for s in results['smiles']]
    #     unique_smiles = list(set(canon_smiles))
    #     novel_ratio = check_novelty(unique_smiles, set(file['smiles']))
    #     print('Valid ratio: ', np.round(len(results)/(args.batch_size*gen_iter), 3))
    #     print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
    #     print('Novelty ratio: ', np.round(novel_ratio/100, 3))
    #     results['qed'] = results['molecule'].apply(lambda x: QED.qed(x) )
    #     results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
    #     results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x) )
    #     results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x) )
    #     # results['temperature'] = temp
    #     results['validity'] = np.round(len(results)/(args.batch_size*gen_iter), 3)
    #     results['unique'] = np.round(len(unique_smiles)/len(results), 3)
    #     results['novelty'] = np.round(novel_ratio/100, 3)
    #     all_dfs.append(results)
    #
    # results = pd.concat(all_dfs)
    # results.to_csv('/home/wang/qianrong/final/src/deeply_random_10000.csv', index=False)
    # unique_smiles = list(set(results['smiles']))
    # canon_smiles = [canonic_smiles(s) for s in results['smiles']]
    # unique_smiles = list(set(canon_smiles))
    # novel_ratio = check_novelty(unique_smiles, set(file['smiles']))
    # print('Valid ratio: ', np.round(len(results)/(args.batch_size*gen_iter*count), 3))
    # print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
    # print('Novelty ratio: ', np.round(novel_ratio/100, 3))

    # 有条件时的代码
    # prop_condition = [[5.0,0.7,120.0,4.0]]

    # prop_condition = prop  # 条件信息

    # admet = [2.5, 0.6, 55.0, 4.0 ]  # 增加的成药性的条件信息
    # admet = [-3.5, 2.8]  # 增加的成药性的条件信息
    # admet = [-2.5,2.8]
    # admet_df = pd.DataFrame(np.array(admet).reshape(1, -1))
    # prop = pd.DataFrame(prop)
    # result = pd.concat([admet_df] * prop.shape[0], ignore_index=True)
    # prop_condition = pd.concat([result,prop], axis=1).values.tolist()  # 将成药性条件与口袋信息合并在一起，
    # print('pocket + admet:',prop_condition)
    # print('最后的条件维度是:',len(prop_condition[0]))

    # prop_condition = [[2.0,0.3,40.0,2.0],[4.0,0.5,80.0,3.0],[6.0,0.7,120.0,4.0]]
    # prop_condition = [[-2.9841, 0.246448286, 166.1, 4.282610861]] #-2.9841, 0.246448286, 166.1, 4.282610861

    scaf_condition = None
    scaf_condition = ['c1ccc(-c2ccccc2-c2nnn[nH]2)cc1<C(=O)OC1CCCCC1']  # c1ccc2c(c1)OCO2    N1CCCC(C(=O)Nc2cccc(F)c2)C1  NC(=O)C(C)NC(=O)C(CC(C)C)NC(=O)C(CO)NC(=O)C(CO)NC(=O)C(CO)NC(=O)C(N)CC(N)=O
    scaf_condition = [i + str('<') * (scaffold_max_len - len(regex.findall(i))) for i in scaf_condition]
    print(scaf_condition)

    all_dfs = []
    count = 0

    # # 性质条件不为空与骨架条件为空时
    # if prop_condition is not None:  # prop_condition 条件不为空
    #     count = 0
    #     for c in prop_condition:
    #         print(c)
    #         molecules = []
    #         # mol_dict = []
    #         count += 1
    #         for i in tqdm(range(gen_iter)):  # gen_iter = 2
    #         # for i in tqdm(range(math.ceil((args.gen_size - len(mol_dict))/ args.batch_size))):
    #             p = None
    #             x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None, ...].repeat(args.batch_size, 1).to('cuda')
    #
    #             # x = torch.tensor([stoi[s] for s in context], dtype=torch.long)[None, ...].repeat(args.batch_size, 1).to('cuda')
    #             # print(x)
    #             # content是起始字符，repeat更改形状，形成一个batch_size的起始字符，一次生成一个batch_size的smiles
    #             # print(x.size())  # 256,1
    #             # p = torch.tensor(prop_condition).repeat(args.batch_size, 1).to('cuda')
    #
    #             p = torch.tensor(c).repeat(args.batch_size, 1).unsqueeze(1).to('cuda')
    #             print(p.size())   # 256,1,128
    #             y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=None, prop=p)  # 0.7 for guacamol    值越大越多样，值越低越单一
    #             for gen_mol in y:
    #                 completion = ''.join([itos[int(i)] for i in gen_mol])
    #                 completion = completion.replace('<', '')
    #                 mol = get_mol(completion)
    #                 if mol:
    #                     molecules.append(mol)
    #         "Valid molecules % = {}".format(len(molecules))
    #         mol_dict = []
    #         for i in molecules:
    #             mol_dict.append({'molecule': i, 'smiles': Chem.MolToSmiles(i)})
    #
    #
    #         # final_length = len(set(mol_dict))
    #         # if final_length<100000:
    #         #     for i in tqdm(range(math.ceil((args.gen_size - len(mol_dict))/ args.batch_size))):
    #         #         p = None
    #         #         x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None, ...].repeat(
    #         #             args.batch_size, 1).to('cuda')
    #         #
    #         #         # x = torch.tensor([stoi[s] for s in context], dtype=torch.long)[None, ...].repeat(args.batch_size, 1).to('cuda')
    #         #         # print(x)
    #         #         # content是起始字符，repeat更改形状，形成一个batch_size的起始字符，一次生成一个batch_size的smiles
    #         #         # print(x.size())  # 256,1
    #         #         # p = torch.tensor(prop_condition).repeat(args.batch_size, 1).to('cuda')
    #         #
    #         #         p = torch.tensor(c).repeat(args.batch_size, 1).unsqueeze(1).to('cuda')
    #         #         print(p.size())  # 256,1,128
    #         #         y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=None,
    #         #                    prop=p)  # 0.7 for guacamol    值越大越多样，值越低越单一
    #         #         for gen_mol in y:
    #         #             completion = ''.join([itos[int(i)] for i in gen_mol])
    #         #             completion = completion.replace('<', '')
    #         #             mol = get_mol(completion)
    #         #             if mol:
    #         #                 molecules.append(mol)
    #         #
    #         #     for i in molecules:
    #         #         mol_dict.append({'molecule': i, 'smiles': Chem.MolToSmiles(i)})
    #
    #
    #         results = pd.DataFrame(mol_dict)
    #         # results['object'] = smiles_test[count - 1]
    #         results['condition'] = str(c)
    #         canon_smiles = [canonic_smiles(s) for s in results['smiles']]
    #         unique_smiles = list(set(canon_smiles))                          # 生成分子的唯一性
    #         novel_ratio = check_novelty(unique_smiles, set(data['smiles']))  # 新颖性，生成的分子不在训练集中
    #         print(f'Condition: {c}')
    #         print('Valid ratio: ', np.round(len(results) / (args.batch_size * gen_iter), 3))
    #         print('Unique ratio: ', np.round(len(unique_smiles) / len(results), 3))
    #         print('Novelty ratio: ', np.round(novel_ratio / 100, 3))
    #         # results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x))
    #         results['qed'] = results['molecule'].apply(lambda x: QED.qed(x))
    #         # results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x))
    #         results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
    #         # results['temperature'] = temp
    #         results['validity'] = np.round(len(results) / (args.batch_size * gen_iter), 3)
    #         results['unique'] = np.round(len(unique_smiles) / len(results), 3)
    #         results['novelty'] = np.round(novel_ratio / 100, 3)
    #         all_dfs.append(results)

    for j in scaf_condition:
        print(j)
        molecules = []
        count += 1
        for i in tqdm(range(gen_iter)):
            x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None, ...].repeat(
                args.batch_size, 1).to('cuda')
            p = None
            sca = torch.tensor([stoi[s] for s in regex.findall(j)], dtype=torch.long)[None, ...].repeat(args.batch_size,
                                                                                                        1).to('cuda')
            y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=None, prop = p, scaffold=sca)  # 0.7 for guacamol
            for gen_mol in y:
                completion = ''.join([itos[int(i)] for i in gen_mol])
                completion = completion.replace('<', '')
                # gen_smiles.append(completion)
                mol = get_mol(completion)
                if mol:
                    molecules.append(mol)

        "Valid molecules % = {}".format(len(molecules))

        mol_dict = []

        for i in molecules:
            mol_dict.append({'molecule': i, 'smiles': Chem.MolToSmiles(i)})

        # for i in gen_smiles:
        #       mol_dict.append({'temperature' : temp, 'smiles': i})

        results = pd.DataFrame(mol_dict)

        # metrics = moses.get_all_metrics(gen_smiles)
        # metrics['temperature'] = temp

        # with open(f'gen_csv/moses_metrics_7_top10.json', 'w') as file:
        #       json.dump(metrics, file)

        canon_smiles = [canonic_smiles(s) for s in results['smiles']]
        unique_smiles = list(set(canon_smiles))
        novel_ratio = check_novelty(unique_smiles, set(data['smiles']))  # 新颖性，生成的分子不在训练集中

        print(f'Scaffold: {j}')
        print('Valid ratio: ', np.round(len(results) / (args.batch_size * gen_iter), 3))
        print('Unique ratio: ', np.round(len(unique_smiles) / len(results), 3))
        print('Novelty ratio: ', np.round(novel_ratio / 100, 3))

        results['scaffold_cond'] = j
        results['qed'] = results['molecule'].apply(lambda x: QED.qed(x))
        results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
        results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x))
        results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x))
        # results['temperature'] = temp
        results['validity'] = np.round(len(results) / (args.batch_size * gen_iter), 3)
        results['unique'] = np.round(len(unique_smiles) / len(results), 3)
        results['novelty'] = np.round(novel_ratio / 100, 3)
        all_dfs.append(results)

    results = pd.concat(all_dfs)
    results.to_csv('frag_case_one.csv', index=False)
    unique_smiles = list(set(results['smiles']))
    canon_smiles = [canonic_smiles(s) for s in results['smiles']]
    unique_smiles = list(set(canon_smiles))
    novel_ratio = check_novelty(unique_smiles, set(data['smiles']))
    print('Valid ratio: ', np.round(len(results) / (args.batch_size * gen_iter * count), 3))
    print('Unique ratio: ', np.round(len(unique_smiles) / len(results), 3))
    print('Novelty ratio: ', np.round(novel_ratio / 100, 3))

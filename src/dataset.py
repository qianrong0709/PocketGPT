import torch
from torch.utils.data import Dataset
from utils import SmilesEnumerator
import numpy as np
import re


class SmileDataset(Dataset):

    def __init__(self, args, data, content, block_size, aug_prob=0.5, prop=None):
        chars = sorted(list(set(content)))
        print(chars)
        data_size, vocab_size = len(data), len(chars) 
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))
        
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.prop = prop
        self.tfm = SmilesEnumerator()
        self.aug_prob = aug_prob

    def __len__(self):
        # if self.debug:
        #     return math.ceil(len(self.data) / (self.max_len + 1))
        # else:
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]  # self.prop.iloc[idx, :].values  --> if multiple properties
        prop = self.prop[idx]  # 无条件时需要注释掉
        # print('dataset里面的条件维度', prop)
        smiles = smiles.strip()

        p = np.random.uniform()
        if p < self.aug_prob:
            smiles = self.tfm.randomize_smiles(smiles)

        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        smiles += str('<')*(self.max_len - len(regex.findall(smiles)))
        
        if len(regex.findall(smiles)) > self.max_len:
            smiles = smiles[:self.max_len]
        smiles = regex.findall(smiles)

        dix = [self.stoi[s] for s in smiles]
        x = torch.tensor(dix[:-1], dtype=torch.long)   # x 没有最后一个值  184维
        y = torch.tensor(dix[1:], dtype=torch.long)    # y 没有第一个值   184维   
        # prop = torch.tensor([prop], dtype=torch.long)  # long类型不行
        prop = torch.tensor([prop], dtype=torch.float)  # 无条件时需要注释掉 [prop]与prop是不一样的，维度不同

        
        return x, y, prop  # 这里有条件
        # return x, y         # 这里无条件

"""
@author: qian rong
@date: 2023/11/01
ori_model code
"""
import re
import os
import pandas as pd
import argparse
import torch
from utils import set_seed
from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
import wandb
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_props', type=int, default=130, help="number of properties to use for condition", required=False)
    parser.add_argument('--n_layer', type=int, default=8, help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8, help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256, help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=5, help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=256, help="batch size", required=False)
    parser.add_argument('--learning_rate', type=int, default=6e-4, help="learning rate", required=False)

    args = parser.parse_args()
    set_seed(42)
    wandb.init(project="pocket", name="drug_generate")


    # path = '/home/wang/qianrong/final/gua_and_moses.csv'
    # path = '/home/wang/qianrong/final/gua_moses_admet.csv'
    # path = '/home/wang/qianrong/final/gua_moses_merge.csv'
    # path = '/home/wang/qianrong/final/file/finetune/deeply_all_duplicate_v1.csv'
    # path = '/home/wang/qianrong/final/file/parp1_finetune/parp1_pdb_final.csv'
    path = '/home/wang/qianrong/final/file/parp1_finetune/parp1_500_train.csv'
    df = pd.read_csv(path)
    train_data, val_data = train_test_split(df, train_size=0.8, test_size=0.2, random_state=42)



    # train_data = pd.read_csv('/home/wang/qianrong/final/file/finetune/deeply_train_more_v1.csv')
    train_data = train_data.dropna(axis=0).reset_index(drop=True)
    train_data.columns = train_data.columns.str.lower()
    print(train_data)


    # val_data = pd.read_csv('/home/wang/qianrong/final/file/finetune/deeply_valid_v3_1_merge.csv')
    val_data = val_data.dropna(axis=0).reset_index(drop=True)
    val_data.columns = val_data.columns.str.lower()
    print(val_data)


    smiles = train_data['smiles']
    vsmiles = val_data['smiles']

    prop = train_data.drop(['smiles', 'file', 'logp', 'qed'], axis=1).values.tolist()
    vprop = val_data.drop(['smiles', 'file', 'logp', 'qed'], axis=1).values.tolist()

    # prop = train_data.drop(['smiles'], axis=1).values.tolist()
    # vprop = val_data.drop(['smiles'], axis=1).values.tolist()

    num_props = args.num_props
    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    lens = [len(regex.findall(i.strip())) for i in (list(smiles.values) + list(vsmiles.values))]
    max_len = max(lens)
    print(max_len)
    max_len = 100                  # 固定预训练的smiles最长长度 185  215 130
    print('Max len: ', max_len)

    smiles = [i + str('<') * (max_len - len(regex.findall(i.strip()))) for i in smiles]
    vsmiles = [i + str('<') * (max_len - len(regex.findall(i.strip()))) for i in vsmiles]

    whole_string = ['#', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[C+]', '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[Cl+]', '[Cl-]', '[F+]', '[F-]', '[H]', '[I+]', '[I-]', '[IH2]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[P-]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH-]', '[SH]', '[Se+]', '[Se-]', '[SeH2]', '[SeH]', '[Se]', '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[se+]', '[se]', 'b', 'c', 'n', 'o', 'p', 's']
    train_dataset = SmileDataset(args, smiles, whole_string, max_len, prop=prop, aug_prob=0)
    valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, prop=vprop, aug_prob=0)
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, num_props=num_props, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)


    model = GPT(mconf)
    uncond_model = torch.load('/home/wang/qianrong/final/weights/parp1_v1/second_finetune_v2.pt')
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k,v in uncond_model.items() if k in model_dict and (v.shape==model_dict[k].shape)}
    # pretrained_dict = {k: v for k, v in uncond_model.items() if k in model_dict.keys()}
    print(pretrained_dict)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                          lr_decay=True, warmup_tokens=0.1 * len(train_data) * max_len,
                          final_tokens=args.max_epochs * len(train_data) * max_len,
                          num_workers=10, ckpt_path=f'/home/wang/qianrong/final/weights/parp1/second_finetune_v1_500.pt',
                          block_size=train_dataset.max_len, generate=False)
    trainer = Trainer(model, train_dataset, valid_dataset, tconf, train_dataset.stoi, train_dataset.itos)
    df = trainer.train(wandb)

    # df.to_csv(f'df.csv', index=False)


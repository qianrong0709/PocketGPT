"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        num = int(bool(config.num_props)) + int(bool(config.scaffold_maxlen))   #int(config.lstm_layers)    #  int(config.scaffold)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + num, config.block_size + num))
                             .view(1, 1, config.block_size + num, config.block_size + num))  # (1,1,186,186) 有条件时
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        # print(B, T, C)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        # print(att.size())
        # print(att)
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.resid_drop(self.proj(y))
        return y, attn_save


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        y, attn = self.attn(self.ln1(x))
        x = x + y                        # attention之后的残差连接
        x = x + self.mlp(self.ln2(x))    # feed-forward之后的残差连接
        return x, attn


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.type_emb = nn.Embedding(2, config.n_embd)


        self.scaff_mlp = nn.Sequential(
            # nn.Linear(config.n_embd*203,2048),
            nn.Linear(config.n_embd * self.config.scaffold_maxlen, 2048),
            nn.GELU(),
            nn.Linear(2048, config.n_embd),
        )


        if config.num_props:
            self.prop_nn = nn.Linear(config.num_props, config.n_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size

        if config.lstm:
            self.lstm = nn.LSTM(input_size = config.n_embd, hidden_size = config.n_embd, num_layers = config.lstm_layers, dropout = 0.3, bidirectional = False)

        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias') or ('bias' in pn):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, prop=None, scaffold=None):
        # print(idx)         # 是一个tensor，smiles按照词表转换成index的tensor，184维，对应的是x
        b, t = idx.size()  # b是batch size; t是输入长度，block_size-1
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # t是每个smiles的减去最后一个字符的长度，应该小于block_size，也就是max_len

        if self.config.num_props:
            assert prop.size(-1) == self.config.num_props, "Num_props should be equal to last dim of property vector"

            # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        # print('token_embeddings的形状：', token_embeddings.shape)
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        type_embeddings = self.type_emb(torch.ones((b, t), dtype=torch.long, device=idx.device))
        x = self.drop(token_embeddings + position_embeddings + type_embeddings)
        # print('x输入的形状：', x.shape)

        if self.config.num_props:
            type_embd = self.type_emb(torch.zeros((b, 1), dtype=torch.long, device=idx.device))  # torch.zeros((b, 128)
            if prop.ndim == 2:
                p = self.prop_nn(prop.unsqueeze(1))  # for single property  增加维度
            else:
                p = self.prop_nn(prop)  # for multiproperty
            p += type_embd
            x = torch.cat([p, x], 1)  # 把条件向量1，1，256加到了整个smiles embbeding向量的最前面

        if self.config.scaffold:
            type_embd = self.type_emb(torch.zeros((b, 1), dtype=torch.long, device=idx.device))
            # print('type_embd的形状',type_embd.shape)

            scaffold_embeds = self.tok_emb(scaffold)  # .mean(1, keepdim = True)
            # print('scaffold_embeds的形状：',scaffold_embeds.shape)
            if self.config.lstm:
                # scaffold_embeds = self.lstm(scaffold_embeds.permute(1, 0, 2))[1][0]
                # print('第一个scaffold_embeds的形状：',scaffold_embeds.shape)
                # # scaffold_embeds = scaffold_embeds.reshape(scaffold_embeds.shape[1], scaffold_embeds.shape[0], 2, self.config.n_embd).mean(2)
                # scaffold_embeds = scaffold_embeds.permute(1, 0, 2)  # mean(0, keepdim = True)
                # # print('第二个scaffold_embeds的形状：',scaffold_embeds.shape)
                #
                # # scaffold_embeds = scaffold_embeds.reshape(self.config.lstm_layers, 1, -1, self.config.n_embd)[-1].permute(1,0,2)
                # # scaffold_embeds = scaffold_embeds.reshape(scaffold_embeds.shape[1], scaffold_embeds.shape[0], self.config.n_embd)


                #### test one
                scaffold_embeds = scaffold_embeds.reshape(scaffold_embeds.size(0), 1, -1)
                # print(scaffold_embeds.device)
                # print('修改后的输入前向量',scaffold_embeds.shape)

                scaffold_embeds = self.scaff_mlp(scaffold_embeds)
                # print('修改后的输入后向量',scaffold_embeds.shape)

            scaffold_embeds += type_embd
            # print('最后拼接前的scaffold_embeds的形状：', scaffold_embeds.shape)

            x = torch.cat([scaffold_embeds, x], 1)
            # print('x的形状',x.size())

        # x = self.blocks(x)
        attn_maps = []

        for layer in self.blocks:  # 8个blocks
            x, attn = layer(x)
            attn_maps.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)

        #
        if self.config.num_props and self.config.scaffold:
            num = int(bool(self.config.num_props)) + int(bool(self.config.scaffold_maxlen))
            # print(num)
        elif self.config.num_props:
            num = int(bool(self.config.num_props))
        elif self.config.scaffold:
            num = int(self.config.scaffold_maxlen)
        else:
            num = 0

        # print(num)

        # print(logits.size())
        logits = logits[:, num:, :]  # 去掉第一维，条件向量
        # print(logits.size())
        loss = None

        # print(logits.reshape(-1, logits.size(-1)).shape)
        # print(targets.view(-1).shape)


        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1))
            # print(loss)
        return logits, loss, attn_maps  # (num_layers, batch_size, num_heads, max_seq_len, max_seq_len)



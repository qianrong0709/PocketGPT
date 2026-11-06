import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = config.dropout
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x):
        b, t, c = x.shape
        def heads(layer):
            return layer(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            heads(self.query), heads(self.key), heads(self.value),
            dropout_p=self.dropout if self.training else 0.0, is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.proj(out))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.n_embd
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(config.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class PocketGPT(nn.Module):
    """One shared decoder; an explicitly masked, gated prefix for four modalities.

    The frozen DeeplyTough adapter supplies the pocket vector. This model never
    fabricates a pocket vector for a missing encoder or missing structure.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.n_embd
        self.tok_emb = nn.Embedding(config.vocab_size, d, padding_idx=0)
        self.type_emb = nn.Embedding(2, d)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, d))
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, config.vocab_size, bias=False)
        self.pocket_projector = nn.Sequential(nn.Linear(config.pocket_dim, d), nn.LayerNorm(d))
        self.property_projector = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
        self.fragment_projector = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        if config.fragment_encoder == "transformer-v1":
            self.fragment_pos_emb = nn.Embedding(config.block_size, d)
            fragment_layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=config.n_head, dim_feedforward=4 * d,
                dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.fragment_sequence_encoder = nn.TransformerEncoder(
                fragment_layer, num_layers=config.fragment_layers,
                norm=nn.LayerNorm(d), enable_nested_tensor=False,
            )
        self.ifp_projector = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.modality_emb = nn.Parameter(torch.zeros(4, d))
        self.gate = nn.Linear(d, 1)
        self.null_prefix = nn.Parameter(torch.zeros(1, d))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.MultiheadAttention):
            nn.init.normal_(module.in_proj_weight, std=0.02)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode_fragment(self, token_ids, token_mask):
        """Encode only available rows, never run attention over an all-padding row."""
        if self.config.fragment_encoder == "mean-v1":
            return ((self.tok_emb(token_ids) * token_mask.unsqueeze(-1)).sum(1)
                    / token_mask.sum(1, keepdim=True).clamp_min(1))
        output = self.tok_emb.weight.new_zeros((token_ids.size(0), self.config.n_embd))
        available = token_mask.any(1).nonzero(as_tuple=True)[0]
        if available.numel() == 0:
            return output
        if token_ids.size(1) > self.config.block_size:
            raise ValueError("Fragment exceeds checkpoint context length")
        tokens = token_ids.index_select(0, available)
        mask = token_mask.index_select(0, available)
        positions = torch.arange(tokens.size(1), device=tokens.device)
        encoded = self.tok_emb(tokens) + self.fragment_pos_emb(positions)[None]
        encoded = self.fragment_sequence_encoder(encoded, src_key_padding_mask=~mask)
        pooled = (encoded * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return output.index_copy(0, available, pooled)

    def make_prefix(self, batch, dropout=0.0):
        available = batch["condition_mask"].bool()
        mask = available.clone()
        if self.training and dropout:
            mask &= torch.rand(mask.shape, device=mask.device) >= dropout
        # Sanitizing before projection ensures a masked NaN cannot contaminate fusion.
        pocket = torch.where(mask[:, 0:1], batch["pocket"], 0.0)
        pmask = batch["property_mask"].bool() & mask[:, 1:2]
        props = torch.where(pmask, batch["properties"], 0.0)
        fmask = batch["fragment_tokens"].ne(0) & mask[:, 2:3]
        ftokens = torch.where(fmask, batch["fragment_tokens"], 0)
        frag = self.encode_fragment(ftokens, fmask)
        ifp = torch.where(mask[:, 3:4], batch["ifp"], 0.0)
        encoded = torch.stack([
            self.pocket_projector(pocket),
            self.property_projector(torch.cat([props, pmask.float()], -1)),
            self.fragment_projector(frag),
            self.ifp_projector(ifp),
        ], 1) + self.modality_emb
        scores = self.gate(encoded).squeeze(-1).masked_fill(~mask, -torch.inf)
        none = ~mask.any(1)
        scores = scores.masked_fill(none[:, None], 0.0)
        weights = scores.softmax(-1) * mask
        fused = (encoded * weights.unsqueeze(-1)).sum(1)
        fused = torch.where(none[:, None], self.null_prefix.expand_as(fused), fused)
        return fused[:, None, :], weights

    def forward(self, input_ids, batch, targets=None, condition_dropout=0.0):
        b, t = input_ids.shape
        if t > self.config.block_size:
            raise ValueError("Input exceeds checkpoint context length")
        prefix, weights = self.make_prefix(batch, condition_dropout)
        x = self.tok_emb(input_ids) + self.pos_emb[:, :t] + self.type_emb.weight[1]
        prefix = prefix + self.type_emb.weight[0]
        x = self.drop(torch.cat([prefix, x], dim=1))
        logits = self.head(self.ln_f(self.blocks(x)))[:, 1:, :]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=0)
        return {"logits": logits, "loss": loss, "gate_weights": weights}

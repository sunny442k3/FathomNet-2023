import torch
from torch import nn, optim

class MLP(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1, activation=nn.GELU()):
        super().__init__()
        self.c_fc    = nn.Linear(d_model, dim_feedforward, bias=False)
        self.c_proj  = nn.Linear(dim_feedforward, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation=nn.GELU(),
                 layer_norm_eps=1e-6) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True, bias=False)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.activation = activation
        self.mlp = MLP(d_model, dim_feedforward, dropout, activation)

    def forward(self, tgt, memory):
        tgt = tgt + self.multihead_attn(self.norm1(tgt), memory, memory)[0]
        tgt = tgt + self.mlp(self.norm2(tgt))
        return tgt

class Query2label(nn.Module):
    def __init__(self, backbone, n_classes, d_model, spatial_dim):
        super().__init__()
        if spatial_dim != d_model:
            self.proj_layer = nn.Linear(spatial_dim, d_model)
        else:
            self.proj_layer = nn.Identity()
        self.query_embeds = nn.Embedding(n_classes, d_model).weight
        self.backbone = backbone
        dim_feedforward=2048
        decoder_layer = TransformerDecoderLayer(d_model=d_model, dim_feedforward=dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.classifier = nn.Linear(d_model, 1, bias=True)

    def forward(self, img_feats):
        if len(img_feats.shape) == 4:  # [bs,2048, 7,7]
            embedding_spatial = img_feats.flatten(2).transpose(1, 2)
        else:  # [bs, 197,468]
            embedding_spatial = img_feats
        embedding_spatial = self.proj_layer(embedding_spatial)
        bs = embedding_spatial.shape[0]
        query_embed = self.query_embeds.weight
        tgt = query_embed.unsqueeze(0).expand(bs, -1, -1)  # no allocation of memory with expand
        out = self.decoder(tgt, embedding_spatial)  # [embed_len_decoder, batch, 768]
        logits = self.classifier(out).squeeze() # [batch size, num_query]
        return logits
import logging
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from fairseq.modules import SinusoidalPositionalEmbedding, LearnedPositionalEmbedding

logger = logging.getLogger(__name__)


class MemoryEmbedding(nn.Module):
    def __init__(self, d_model, memory_len, emb_dropout, scale_embedding=True, num_pitch_type=3):
        super(MemoryEmbedding, self).__init__()
        num_pitch_type += 2  # Add [cls] and [pad]
        self.memory_len = memory_len
        self.emb_dropout = emb_dropout
        self.emb_scale = math.sqrt(d_model) if scale_embedding else 1.0
        # sinusoidal이 아닌 positional embedding 사용
        self.pos = nn.Embedding(memory_len, d_model)
        self.pitch_embedding = nn.Embedding(num_pitch_type, d_model)
        self.label_embedding = nn.Embedding(5, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x_pitch, x_label):
        """
        Args
            x_ptich: Memory of pitch types thrown last L times (B x L)
            x_ptich: Memory of labels last L times (B x L)
        """
        # Positional Encoder for Inputs -> (B, L) => (B, L, d)
        x_pitch = self.emb_scale * self.pitch_embedding(x_pitch) + self.pos(
            torch.arange(self.memory_len).to(x_pitch.device)
        )
        x_label = self.emb_scale * self.label_embedding(x_label) + self.pos(
            torch.arange(self.memory_len).to(x_pitch.device)
        )

        x = self.layernorm(x_pitch + x_label)
        # (B, L, d) => (L, B, d)
        x = F.dropout(x, self.emb_dropout, self.training).transpose(0, 1)
        return x


class IntegratedEmbedding(nn.Module):
    def __init__(self, d_model, n_continuous_feature, grouping=None, *n_each_discrete_feature):
        """ 
        Args:
            grouping: whether each pitcher & hitter stats embedding is grouping(total, season, short) or not
        """
        super(IntegratedEmbedding, self).__init__()
        self.cont_emb = ContinuousEmbedding(n_continuous_feature, d_model, grouping)
        self.disc_emb = DiscreteEmbedding(d_model, *n_each_discrete_feature)

    def forward(self, x_disc, x_cont):
        x_disc = self.disc_emb(x_disc)
        x_cont = self.cont_emb(x_cont)
        return torch.cat((x_disc, x_cont), dim=1).transpose(1, 0)


class ContinuousEmbedding(nn.Module):
    def __init__(self, n_feature, d_model, grouping):
        """ 
        Project scalar feature to vector space by matrix multiplication
        (N x n_feature) -> (N x n_feature x d_model)    
        """
        super(ContinuousEmbedding, self).__init__()
        self.n_feature = n_feature
        self.grouping = grouping

        if self.grouping is None or self.grouping == "state":
            self.weights = nn.Parameter(torch.randn((n_feature, d_model)))
        elif self.grouping == "pitcher":
            # pitcher grouping
            # self.total_weights = nn.Linear(20, d_model)
            # self.season_weights = nn.Linear(20, d_model)
            # short만 사용
            self.short_weights = nn.Linear(19, d_model)
            self.ratio_weights = nn.Linear(9, d_model)
        elif self.grouping == "hitter":
            # hitter grouping
            # self.total_weights = nn.Linear(18, d_model)
            # self.season_weights = nn.Linear(18, d_model)
            self.short_weights = nn.Linear(18, d_model)
            self.ratio_weights = nn.Linear(9, d_model)
        else:
            raise ValueError("Invalid parameter name for grouping(use pitcher, hitter, or state)")

    def forward(self, x):
        if self.grouping is None or self.grouping == "state":
            return x.unsqueeze(-1).mul(self.weights)
        elif self.grouping == "pitcher":
            # pitcher grouping
            # total = self.total_weights(x[:, :20]).unsqueeze(1)
            # season = self.season_weights(x[:, 20:40]).unsqueeze(1)
            short = self.short_weights(x[:, :-9]).unsqueeze(1)
            ratio = self.ratio_weights(x[:, -9 : self.n_feature]).unsqueeze(1)
            # return torch.cat((total, season, short, ratio), dim=1)
            return torch.cat((short, ratio), dim=1)
        elif self.grouping == "hitter":
            # hitter grouping
            # total = self.total_weights(x[:, :18]).unsqueeze(1)
            # season = self.season_weights(x[:, 18:36]).unsqueeze(1)
            short = self.short_weights(x[:, :-9]).unsqueeze(1)
            ratio = self.ratio_weights(x[:, -9 : self.n_feature]).unsqueeze(1)
            # return torch.cat((total, season, short, ratio), dim=1)
            return torch.cat((short, ratio), dim=1)


class DiscreteEmbedding(nn.Module):
    def __init__(self, d_model, *num_embedding_features):
        """ 
        Project Discrete feature to vector space by Embedding layers
        (N x n_feature) -> (N x n_feature x d_model)    
        
        Args:
            num_embedding_features : The number of unique values of each discrete features
        """
        super(DiscreteEmbedding, self).__init__()
        self.embedding_layers = nn.ModuleList([])
        for i in range(len(num_embedding_features)):
            new_layer = nn.Embedding(num_embedding_features[i], d_model)
            self.embedding_layers.append(new_layer)

    def forward(self, x):
        out = []
        for i in range(len(self.embedding_layers)):
            out_ = self.embedding_layers[i](x[:, i])  # N x d
            out.append(out_)  # N x d
        return torch.stack(out, dim=1)  # N x n_feature x d


class CrossmodalTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, n_layer):
        super(CrossmodalTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        for layer in range(n_layer):
            new_layer = TransformerDecoderBlock(d_model, nhead, dim_feedforward, dropout)
            self.layers.append(new_layer)

    def forward(self, src, tgt, src_key_padding_mask=None):
        weights = []
        for layer in self.layers:
            tgt, w = layer(src, tgt, src_key_padding_mask=src_key_padding_mask)
            weights.append(w)
        return tgt, weights


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, n_layer):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        for layer in range(n_layer):
            new_layer = TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            self.layers.append(new_layer)

    def forward(self, src, x_key_padding_mask=None):
        weights = []
        for layer in self.layers:
            src, w = layer(src, x_key_padding_mask=x_key_padding_mask)
            weights.append(w)
        return src, weights


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        """
        Args:
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multiheadattention models (required).
            dim_feedforward: the dimension of the feedforward network model (required).
            dropout: the dropout value (required).
        """
        super(TransformerEncoderBlock, self).__init__()
        self.transformer = TransformerBlock(d_model, nhead, dropout)
        self.feedforward = FeedForwardBlock(d_model, dim_feedforward, dropout)

    def forward(self, x, x_key_padding_mask=None, x_attn_mask=None):
        """
        x : input of the encoder layer
        """
        x, weights = self.transformer(
            x, x, x, key_padding_mask=x_key_padding_mask, attn_mask=x_attn_mask
        )
        x = self.feedforward(x)
        return x, weights


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        """
        Args:
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multiheadattention models (required).
            dim_feedforward: the dimension of the feedforward network model (required).
            dropout: the dropout value (required).
        """
        super(TransformerDecoderBlock, self).__init__()
        # self.transformer1 = TransformerBlock(d_model, nhead, dropout)
        self.transformer2 = TransformerBlock(d_model, nhead, dropout)
        self.feedforward = FeedForwardBlock(d_model, dim_feedforward, dropout)

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        src_key_padding_mask=None,
        tgt_mask=None,
        tgt_key_padding_mask=None,
    ):
        """
        src : output from the encoder layer(query)
        tgt : input from the decoder layer(key, value)
        """
        # tgt, _ = self.transformer1(
        #     tgt, tgt, tgt, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask
        # )
        x, weights = self.transformer2(
            tgt, src, src, key_padding_mask=src_key_padding_mask, attn_mask=src_mask
        )
        x = self.feedforward(x)
        return x, weights


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        x, weights = self.self_attn(
            query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        x = query + self.dropout(x)
        x = self.layernorm(x)
        return x, weights


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        x2 = self.linear2(self.dropout1(F.relu(self.linear1(x))))
        x = F.relu(x + self.dropout2(x2))
        x = self.layernorm(x)
        return x

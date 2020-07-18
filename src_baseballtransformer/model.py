import logging
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import *
from fairseq.modules import SinusoidalPositionalEmbedding, LearnedPositionalEmbedding
from module import *

logger = logging.getLogger(__name__)


class BaseballTransformer(nn.Module):
    def __init__(
        self,
        n_pitcher: List[int],
        n_batter: List[int],
        n_state: List[int],
        n_memory_layer: int,
        n_encoder_layer: int,
        memory_len=50,
        num_y_pitching=3,
        d_model=64,
        n_head=1,
        dropout=0.1,
        attention_type="dot",
    ):
        """
        Args:
            n_pitcher, n_batter, n_state: the number of input features (required).
                every feature of pitcher and batter is discretized
            n_memory_layer: the number of memory layers (required).
            n_encoder_layer: the number of encoder layers (required).
            memory_len: the sequence length of memory input
            num_y_pitching: the number of pitching type (default=3). fast, horizontal, vertical 
            d_model: the number of expected features in the input (default=64).
            n_head: the number of heads in the multiheadattention models (default=1).
            dropout: the dropout value (default=0.1).
            attention_type: the type of attention layer (default=dot)
        """
        super().__init__()
        self.num_y_pitching = num_y_pitching
        self.attention_type = attention_type
        self.dropout = dropout
        # feedforward dimension은 (모델) 차원 x 4) 로 고정
        dim_feedforward = 4 * d_model
        self.pitcher_embedding = DiscreteEmbedding(d_model, *n_pitcher,)
        self.batter_embedding = DiscreteEmbedding(d_model, *n_batter,)
        self.state_embedding = DiscreteEmbedding(d_model, *n_state,)
        self.memory_embedding = MemoryEmbedding(
            d_model, memory_len + 1, dropout
        )  # Add 1 for [cls] token

        self.memory_encoder = TransformerEncoder(
            d_model, n_head, dim_feedforward, dropout, n_memory_layer
        )
        self.pitcher_encoder = TransformerEncoder(
            d_model, n_head, dim_feedforward, dropout, n_encoder_layer
        )
        self.batter_encoder = TransformerEncoder(
            d_model, n_head, dim_feedforward, dropout, n_encoder_layer
        )
        self.state_encoder = TransformerEncoder(
            d_model, n_head, dim_feedforward, dropout, n_encoder_layer
        )
        self.attn_pitcher = nn.Linear(d_model, d_model)
        self.attn_batter = nn.Linear(d_model, d_model)
        self.attn_state = nn.Linear(d_model, d_model)

        self.fc_layer1 = nn.Linear(4 * d_model, 4 * d_model)
        self.fc_layer2 = nn.Linear(4 * d_model, 4 * d_model)
        self.classifier = nn.Linear(4 * d_model, self.num_y_pitching)

    def attention(self, w, src, tgt):
        """
        Do attention
        w : attention linear layer / d_model, d_model
        src : N, d_model
        tgt : N, L, d_model
        """
        # (N, 1, d_model) x (N, d_model, L) = (N, 1, L)
        if self.attention_type == "dot":
            # scaled dot product (일반 dot product는 한 feature에 쏠리는 경향이 있음)
            energy = torch.bmm(src.unsqueeze(1), tgt.transpose(1, 2)) / src.shape[-1] ** (1 / 2)
        elif self.attention_type == "general":
            energy = torch.bmm(w(src.unsqueeze(1)), tgt.transpose(1, 2)) / src.shape[-1] ** (1 / 2)
        elif self.attention_type == "han":
            # from the paper "Hierarchical Attention Network"
            energy = torch.bmm(
                w(src.unsqueeze(1)).tanh(), w(tgt).tanh().transpose(1, 2)
            ) / src.shape[-1] ** (1 / 2)
        attn_weight = torch.softmax(energy, dim=-1)

        # (N, 1, L) x (N, L, d_model) = (N, 1, d_model) -> (N, d_model)
        context = torch.bmm(attn_weight, tgt).squeeze(1)
        # attn_weight : (N, L)
        # context : (N, d_model)
        return attn_weight, context

    def forward(
        self, pitcher, batter, state, memory_pitch, memory_label, memory_mask=None,
    ):
        pitcher_x = self.pitcher_embedding(pitcher)
        batter_x = self.batter_embedding(batter)
        state_x = self.state_embedding(state)
        memory_x = self.memory_embedding(memory_pitch, memory_label)

        # dim of x = (L, N, d_model)
        memory_x, memory_x_weight = self.memory_encoder(memory_x, memory_mask)
        pitcher_x, pitcher_x_weight = self.pitcher_encoder(pitcher_x)
        batter_x, batter_x_weight = self.batter_encoder(batter_x)
        state_x, state_x_weight = self.state_encoder(state_x)

        # Get [cls] token from last timestep
        memory_pool = memory_x[-1]
        pitcher_attn, pitcher_context = self.attention(self.attn_pitcher, memory_pool, pitcher_x)
        batter_attn, batter_context = self.attention(self.attn_batter, memory_pool, batter_x)
        state_attn, state_context = self.attention(self.attn_state, memory_pool, state_x)

        # dim of last_hidden = (N, 4 * d_model)
        last_hidden = torch.cat(
            (memory_pool, pitcher_context, batter_context, state_context), dim=-1
        )
        out = F.relu(self.fc_layer1(last_hidden))
        out = self.fc_layer2(F.dropout(out, p=self.dropout, training=self.training))
        out = out + last_hidden
        out = self.classifier(out)

        # weights
        weights = [
            memory_x_weight,
            pitcher_x_weight,
            batter_x_weight,
            state_x_weight,
            pitcher_attn,
            batter_attn,
            state_attn,
        ]
        return out, weights

import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


# Case 1, 2 모델 (긍/부정 상황에서 투구구질 label 달리함)
class BaseballTransformerOnlyPitcher(nn.Module):
    def __init__(
        self,
        n_pitcher_cont,
        n_batter_cont,
        n_state_cont,
        n_encoder_layer,
        n_decoder_layer,
        n_concat_layer,
        do_grouping=True,
        num_y_pitching=9,
        num_y_batting=2,
        d_model=64,
        nhead=2,
        dim_feedforward=128,
        dropout=0.1,
    ):
        """
        Args:
            n_pitcher_cont, n_batter_cont, n_state_cont: the number of each continuous input features (required).
            n_encoder_layer: the number of encoder layers (required).
            n_decoder_layer: the number of decoder layers (required).
            n_concat_layer: the number of concat layers (required).
            grouping: whether each pitcher & hitter stats embedding is grouping(total, season, short) or not
            num_y_pitching: the number of pitching type (default=7). except 2 balls <KNUC, SINK>
            num_y_batting: the number of battinf type (default=2).
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multiheadattention models (default=2).
            dim_feedforward: the dimension of the feedforward network model (default=128).
            dropout: the dropout value (default=0.1).
        """
        super(BaseballTransformerOnlyPitcher, self).__init__()
        self.num_y_pitching = num_y_pitching

        self.pitcher_embedding = IntegratedEmbedding(
            d_model, n_pitcher_cont, "pitcher" if do_grouping else None, *([2] * 9), 2, 2, 2,
        )
        self.batter_embedding = IntegratedEmbedding(
            d_model, n_batter_cont, "hitter" if do_grouping else None, 2, 2,
        )
        self.state_embedding = IntegratedEmbedding(
            d_model, n_state_cont, "state" if do_grouping else None, 10, 8, 9,
        )

        self.pitcher_encoder = TransformerEncoder(
            d_model, nhead, dim_feedforward, dropout, n_encoder_layer
        )
        self.batter_encoder = TransformerEncoder(
            d_model, nhead, dim_feedforward, dropout, n_encoder_layer
        )
        self.state_encoder = TransformerEncoder(
            d_model, nhead, dim_feedforward, dropout, n_encoder_layer
        )

        self.pitcher_layers_with_batter = CrossmodalTransformerEncoder(
            d_model, nhead, dim_feedforward, dropout, n_decoder_layer
        )
        self.pitcher_layers_with_state = CrossmodalTransformerEncoder(
            d_model, nhead, dim_feedforward, dropout, n_decoder_layer
        )

        self.connection_layer = nn.Linear(2 * d_model, d_model)

        self.dropout_pitcher = nn.Dropout(dropout)
        self.pitching_classifier = nn.Linear(d_model, self.num_y_pitching)

        self.pitch_classifiers = nn.ModuleList([])
        for layer in range(self.num_y_pitching):
            classifier = nn.Linear(d_model, 1)
            self.pitch_classifiers.append(classifier)

    def forward(
        self,
        pitcher_discrete,
        pitcher_continuous,
        batter_discrete,
        batter_continuous,
        state_discrete,
        state_continuous,
        do_concat=False,
    ):
        pitcher_x = self.pitcher_embedding(pitcher_discrete, pitcher_continuous)
        batter_x = self.batter_embedding(batter_discrete, batter_continuous)
        state_x = self.state_embedding(state_discrete, state_continuous)

        # dim = (L, N, d_model)
        pitcher_x = self.pitcher_encoder(pitcher_x)
        batter_x = self.batter_encoder(batter_x)
        state_x = self.state_encoder(state_x)

        pitcher_x_with_batter = self.pitcher_layers_with_batter(batter_x, pitcher_x)
        pitcher_x_with_state = self.pitcher_layers_with_state(state_x, pitcher_x)
        pitcher_x2 = torch.cat([pitcher_x_with_batter, pitcher_x_with_state], dim=-1)
        pitcher_x = self.connection_layer(pitcher_x2)

        # Take only the pitch type variables
        pitcher_x = self.dropout_pitcher(pitcher_x)[:9]

        pitching_score = []
        for i in range(len(self.pitch_classifiers)):
            pitching_score.append(self.pitch_classifiers[i](pitcher_x[i]))
        # dim = (N, L, 1) -> (N, L)
        pitching_score = torch.stack(pitching_score, dim=0).transpose(0, 1).squeeze(-1)
        return pitching_score


# Case 3 모델 (긍/부정 상황에서 투구구질 label 똑같이함, Input으로 긍/부정 상황 추가)
class BaseballTransformerPitcherSentiment(nn.Module):
    def __init__(
        self,
        n_pitcher_cont,
        n_batter_cont,
        n_state_cont,
        n_encoder_layer,
        n_decoder_layer,
        n_concat_layer,
        do_grouping=True,
        num_y_pitching=9,
        num_y_batting=2,
        d_model=64,
        nhead=2,
        dim_feedforward=128,
        dropout=0.1,
    ):
        """
        Args:
            n_pitcher_cont, n_batter_cont, n_state_cont: the number of each continuous input features (required).
            n_encoder_layer: the number of encoder layers (required).
            n_decoder_layer: the number of decoder layers (required).
            n_concat_layer: the number of concat layers (required).
            grouping: whether each pitcher & hitter stats embedding is grouping(total, season, short) or not
            num_y_pitching: the number of pitching type (default=7). except 2 balls <KNUC, SINK>
            num_y_batting: the number of battinf type (default=2).
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multiheadattention models (default=2).
            dim_feedforward: the dimension of the feedforward network model (default=128).
            dropout: the dropout value (default=0.1).
        """
        super(BaseballTransformerPitcherSentiment, self).__init__()
        self.num_y_pitching = num_y_pitching

        self.pitcher_embedding = IntegratedEmbedding(
            d_model, n_pitcher_cont, "pitcher" if do_grouping else None, *([2] * 9), 2, 2, 2,
        )
        self.batter_embedding = IntegratedEmbedding(
            d_model, n_batter_cont, "hitter" if do_grouping else None, 2, 2,
        )
        self.state_embedding = IntegratedEmbedding(
            d_model, n_state_cont, "state" if do_grouping else None, 10, 8, 9,
        )
        ## input_sentiment
        self.sentiment_embedding = Sentiment_DiscreteEmbedding(d_model, 2)

        self.pitcher_encoder = TransformerEncoder(
            d_model, nhead, dim_feedforward, dropout, n_encoder_layer
        )
        self.batter_encoder = TransformerEncoder(
            d_model, nhead, dim_feedforward, dropout, n_encoder_layer
        )
        self.state_encoder = TransformerEncoder(
            d_model, nhead, dim_feedforward, dropout, n_encoder_layer
        )
        self.sentiment_encoder = TransformerEncoder(
            d_model, nhead, dim_feedforward, dropout, n_encoder_layer
        )

        self.pitcher_layers_with_batter = CrossmodalTransformerEncoder(
            d_model, nhead, dim_feedforward, dropout, n_decoder_layer
        )
        self.pitcher_layers_with_state = CrossmodalTransformerEncoder(
            d_model, nhead, dim_feedforward, dropout, n_decoder_layer
        )
        self.pitcher_layers_with_sentiment = CrossmodalTransformerEncoder(
            d_model, nhead, dim_feedforward, dropout, n_decoder_layer
        )

        self.connection_layer = nn.Linear(3 * d_model, d_model)

        self.dropout_pitcher = nn.Dropout(dropout)
        self.pitching_classifier = nn.Linear(d_model, self.num_y_pitching)

        self.pitch_classifiers = nn.ModuleList([])
        for layer in range(self.num_y_pitching):
            classifier = nn.Linear(d_model, 1)
            self.pitch_classifiers.append(classifier)

    def forward(
        self,
        pitcher_discrete,
        pitcher_continuous,
        batter_discrete,
        batter_continuous,
        state_discrete,
        state_continuous,
        sentiment_discrete,
        do_concat=False,
    ):
        pitcher_x = self.pitcher_embedding(pitcher_discrete, pitcher_continuous)
        batter_x = self.batter_embedding(batter_discrete, batter_continuous)
        state_x = self.state_embedding(state_discrete, state_continuous)
        sentiment_x = self.sentiment_embedding(sentiment_discrete).transpose(1, 0)

        # dim = (L, N, d_model)
        pitcher_x = self.pitcher_encoder(pitcher_x)
        batter_x = self.batter_encoder(batter_x)
        state_x = self.state_encoder(state_x)
        sentiment_x = self.sentiment_encoder(sentiment_x)

        pitcher_x_with_batter = self.pitcher_layers_with_batter(batter_x, pitcher_x)
        pitcher_x_with_state = self.pitcher_layers_with_state(state_x, pitcher_x)
        pitcher_x_with_sentiment = self.pitcher_layers_with_sentiment(sentiment_x, pitcher_x)
        pitcher_x2 = torch.cat([pitcher_x_with_batter, pitcher_x_with_state, pitcher_x_with_sentiment], dim=-1)
        pitcher_x = self.connection_layer(pitcher_x2)

        # Take only the pitch type variables
        pitcher_x = self.dropout_pitcher(pitcher_x)[:9]

        pitching_score = []
        for i in range(len(self.pitch_classifiers)):
            pitching_score.append(self.pitch_classifiers[i](pitcher_x[i]))
        # dim = (N, L, 1) -> (N, L)
        pitching_score = torch.stack(pitching_score, dim=0).transpose(0, 1).squeeze(-1)
        return pitching_score


# Self-attention 전의 input embedding (Categorical 변수, Continuous 변수)
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


# input embedding: Continuous 변수
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
            self.total_weights = nn.Linear(20, d_model)
            self.season_weights = nn.Linear(20, d_model)
            self.short_weights = nn.Linear(19, d_model)
            # hitter grouping
        elif self.grouping == "hitter":
            self.total_weights = nn.Linear(18, d_model)
            self.season_weights = nn.Linear(18, d_model)
            self.short_weights = nn.Linear(18, d_model)
        else:
            raise ValueError("Invalid parameter name for grouping(use pitcher, hitter, or state)")

    def forward(self, x):
        if self.grouping is None or self.grouping == "state":
            return x.unsqueeze(-1).mul(self.weights)
        elif self.grouping == "pitcher":
            # pitcher grouping
            total = self.total_weights(x[:, :20]).unsqueeze(1)
            season = self.season_weights(x[:, 20:40]).unsqueeze(1)
            short = self.short_weights(x[:, 40 : self.n_feature]).unsqueeze(1)
            return torch.cat((total, season, short), dim=1)
        # hitter grouping
        elif self.grouping == "hitter":
            total = self.total_weights(x[:, :18]).unsqueeze(1)
            season = self.season_weights(x[:, 18:36]).unsqueeze(1)
            short = self.short_weights(x[:, 36 : self.n_feature]).unsqueeze(1)
            return torch.cat((total, season, short), dim=1)


# input embedding: Categorical 변수
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


# input embedding: Categorical 변수 (긍/부정 Input으로 추가)
class Sentiment_DiscreteEmbedding(nn.Module):
    def __init__(self, d_model, *num_embedding_features):
        """ 
        Project Discrete feature to vector space by Embedding layers
        (N x n_feature) -> (N x n_feature x d_model)    
        
        Args:
            num_embedding_features : The number of unique values of each discrete features
        """
        super(Sentiment_DiscreteEmbedding, self).__init__()
        self.embedding_layers = nn.ModuleList([])
        for i in range(len(num_embedding_features)):
            new_layer = nn.Embedding(num_embedding_features[i], d_model)
            self.embedding_layers.append(new_layer)

    def forward(self, x):
        out = []
        for i in range(len(self.embedding_layers)):
            out_ = self.embedding_layers[i](x)  # N x d
            out.append(out_)  # N x d
        return torch.stack(out, dim=1)  # N x n_feature x d


# Transformer의 Decoder 부분이라고 생각하면 됨
class CrossmodalTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, n_layer):
        super(CrossmodalTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        for layer in range(n_layer):
            new_layer = TransformerDecoderBlock(d_model, nhead, dim_feedforward, dropout)
            self.layers.append(new_layer)

    def forward(self, src, tgt):
        for layer in self.layers:
            tgt = layer(src, tgt)
        return tgt


# Transformer의 Encoder 부분
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, n_layer):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        for layer in range(n_layer):
            new_layer = TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            self.layers.append(new_layer)

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src


# Transformer Encoder 부분의 Block 모듈
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
        x = self.transformer(x, x, x, key_padding_mask=x_key_padding_mask, attn_mask=x_attn_mask)
        x = self.feedforward(x)
        return x


# Transformer Decoder 부분의 Block 모듈
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
        """
        tgt = self.transformer1(
            tgt, tgt, tgt, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask
        )
        """
        x = self.transformer2(
            tgt, src, src, key_padding_mask=src_key_padding_mask, attn_mask=src_mask
        )
        x = self.feedforward(x)
        return x


# Transformer Encoder/Decoder Block 모듈에 포함될 <Multi-headd + Dropout + Layernormalization> 모듈
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        x = self.self_attn(
            query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )[0]
        x = query + self.dropout(x)
        x = self.layernorm(x)
        return x


# Transformer 의 Feedforward 모듈
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

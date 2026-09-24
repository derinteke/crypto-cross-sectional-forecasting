"""LightGBM on the daily snapshot, and GRU / Transformer on the hourly sequence.

EN: The tree model sees one row per (day, coin): the engineered snapshot. The
    sequence models see the last 120 hours of raw hourly channels AND the same
    snapshot, concatenated before the output head. I give them the snapshot on
    purpose: the question I want answered is "does the hourly path add anything
    on top of the features a tree already has?", not "can a GRU rediscover
    30-day momentum from 5 days of data?" (it cannot; the data is not there).
    Every network has an `n_outputs` head: 1 for a point forecast, or one per
    quantile for the probabilistic version.
TR: Ağaç modeli (gün, coin) başına bir satır görüyor: mühendislikle üretilmiş
    anlık görüntü. Dizi modelleri ise son 120 saatin ham saatlik kanallarını VE
    aynı anlık görüntüyü görüyor; ikisi çıkış başlığından önce birleştiriliyor.
    Anlık görüntüyü onlara bilerek veriyorum: cevaplamak istediğim soru "saatlik
    yol, bir ağacın zaten sahip olduğu özelliklerin üzerine bir şey katıyor
    mu?", "bir GRU 5 günlük veriden 30 günlük momentumu yeniden keşfedebilir
    mi?" değil (edemez; veri orada değil).
    Her ağın bir `n_outputs` başlığı var: nokta tahmin için 1, olasılıksal
    versiyon için quantile başına bir çıktı.
"""

from __future__ import annotations

import math

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src import config

# --------------------------------------------------------------------------- #
# LightGBM
# --------------------------------------------------------------------------- #
LGBM_PARAMS = dict(
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=200,   # returns are noisy: leaves need many samples
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    verbose=-1,
    seed=config.SEED,
    deterministic=True,
    num_threads=4,
)
LGBM_MAX_ROUNDS = 2000
LGBM_EARLY_STOP = 100


def fit_lgbm(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    quantile: float | None = None,
) -> lgb.Booster:
    """Point model (L2) or one quantile model, early-stopped on validation."""
    params = dict(LGBM_PARAMS)
    if quantile is None:
        params.update(objective="regression", metric="l2")
    else:
        params.update(objective="quantile", alpha=quantile, metric="quantile")
    dtrain = lgb.Dataset(x_train, y_train, free_raw_data=False)
    dval = lgb.Dataset(x_val, y_val, reference=dtrain, free_raw_data=False)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=LGBM_MAX_ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(LGBM_EARLY_STOP, verbose=False)],
    )


# --------------------------------------------------------------------------- #
# Sequence networks / Dizi ağları
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    """Sinusoidal positions, reused from my Jena Climate project."""

    def __init__(self, d_model: int, max_len: int = 1000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class _Head(nn.Module):
    """Concatenate the sequence summary with the snapshot, then predict."""

    def __init__(self, seq_dim: int, n_static: int, n_outputs: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_dim + n_static, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_outputs),
        )

    def forward(self, h: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, static], dim=-1))


class GRUForecaster(nn.Module):
    def __init__(self, n_channels: int, n_static: int, n_outputs: int = 1,
                 hidden_size: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.rnn = nn.GRU(n_channels, hidden_size, batch_first=True)
        self.head = _Head(hidden_size, n_static, n_outputs, dropout)

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, h_n = self.rnn(seq)
        return self.head(h_n[-1], static)


class TransformerForecaster(nn.Module):
    """Small encoder-only transformer with mean pooling (as in my Jena project).

    EN: 120 hourly steps x 5 channels is not a lot of signal, and financial
        data is mostly noise, so I keep this deliberately small: d_model 32,
        two layers. A bigger model would memorise the training years.
    TR: 120 saatlik adım x 5 kanal çok fazla sinyal değil ve finansal veri
        büyük ölçüde gürültü; bu yüzden bunu bilerek küçük tutuyorum: d_model 32,
        iki katman. Daha büyük bir model eğitim yıllarını ezberlerdi.
    """

    def __init__(self, n_channels: int, n_static: int, n_outputs: int = 1,
                 d_model: int = 32, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = _Head(d_model, n_static, n_outputs, dropout)

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.pos(self.input_proj(seq)))
        return self.head(self.norm(h.mean(dim=1)), static)


SEQ_MODELS = {"GRU": GRUForecaster, "Transformer": TransformerForecaster}


def build_seq_model(name: str, n_channels: int, n_static: int, n_outputs: int = 1) -> nn.Module:
    if name not in SEQ_MODELS:
        raise KeyError(f"Unknown model {name!r}. Known: {list(SEQ_MODELS)}")
    return SEQ_MODELS[name](n_channels=n_channels, n_static=n_static, n_outputs=n_outputs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

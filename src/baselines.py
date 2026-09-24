"""The signals every model has to beat, computed before any model is trained.

EN: In cross-sectional crypto the honest baselines are not "predict zero",
    they are the textbook factors. If a gradient-boosted model only rediscovers
    one-week momentum, it has not earned its complexity. So the bar is:
      Random      scores drawn at random: the null. After costs it loses money,
                  which shows exactly what trading costs alone do to a strategy.
      Momentum    last 7 days' return: winners keep winning.
      Reversal    minus last 24 hours' return: short-term overshoot corrects.
      Ridge       a linear model on the same features the big models see.
TR: Kesitsel kriptoda dürüst baseline'lar "sıfır tahmin et" değil, ders
    kitabı faktörleri. Gradient boosting bir model yalnızca bir haftalık
    momentumu yeniden keşfediyorsa, karmaşıklığını hak etmiyor. Dolayısıyla
    çıta şu:
      Random      rastgele skorlar: sıfır hipotezi. Maliyetlerden sonra para
                  kaybettiriyor ve tek başına işlem maliyetlerinin bir stratejiye
                  ne yaptığını tam olarak gösteriyor.
      Momentum    son 7 günün getirisi: kazananlar kazanmaya devam ediyor.
      Reversal    son 24 saatin getirisinin eksisi: kısa vadeli aşırı hareket
                  düzeliyor.
      Ridge       büyük modellerin gördüğü özelliklerle aynı özelliklere sahip
                  doğrusal bir model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src import config
from src.features import MARKET_FEATURES, RANK_FEATURES
from src.metrics import daily_rank_ic


def random_signal(frame: pd.DataFrame, seed: int = config.SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(len(frame)), index=frame.index, name="Random")


def momentum_signal(frame: pd.DataFrame) -> pd.Series:
    return frame["ret_168h"].rename("Momentum7d")


def reversal_signal(frame: pd.DataFrame) -> pd.Series:
    return (-frame["ret_24h"]).rename("Reversal24h")


RIDGE_FEATURES = RANK_FEATURES + MARKET_FEATURES
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def fit_ridge(train: pd.DataFrame, val: pd.DataFrame, target: str = "y_rankgauss"):
    """Pick alpha on validation, return (model, mean, std, chosen_alpha).

    EN: Rank features are already on a common scale; the market features are
        not, so I standardise everything with train-only statistics.
    TR: Sıra özellikleri zaten ortak bir ölçekte; piyasa özellikleri değil,
        dolayısıyla her şeyi yalnızca train istatistikleriyle standartlaştırıyorum.
    """
    mu = train[RIDGE_FEATURES].mean()
    sd = train[RIDGE_FEATURES].std().replace(0, 1.0)
    xtr = ((train[RIDGE_FEATURES] - mu) / sd).to_numpy()
    xva = ((val[RIDGE_FEATURES] - mu) / sd).to_numpy()

    best = None
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(xtr, train[target].to_numpy())
        pred = pd.Series(model.predict(xva), index=val.index)
        score = daily_rank_ic(pred, val["y_day0"]).mean()
        if best is None or score > best[0]:
            best = (score, model, alpha)
    return best[1], mu, sd, best[2]


def predict_ridge(model, mu, sd, frame: pd.DataFrame) -> pd.Series:
    x = ((frame[RIDGE_FEATURES] - mu) / sd).to_numpy()
    return pd.Series(model.predict(x), index=frame.index, name="Ridge")


def vol_scaled_quantiles(train: pd.DataFrame, frame: pd.DataFrame,
                         quantiles=config.QUANTILES) -> np.ndarray:
    """Probabilistic baseline: empirical quantiles of vol-standardised returns.

    EN: The simplest defensible distribution forecast: a coin's next-day
        spread scales with its recent volatility, and the shape comes from
        history. Any learned quantile model has to beat this on pinball loss.
    TR: Savunulabilir en basit dağılım tahmini: bir coinin ertesi günkü
        yayılımı son dönem volatilitesiyle ölçekleniyor, şekil ise geçmişten
        geliyor. Öğrenilen her quantile modeli pinball kaybında bunu geçmek
        zorunda.
    """
    scale_tr = train["vol_168h"] * np.sqrt(config.HORIZON_HOURS)
    z = (train["y_demeaned"] / scale_tr).replace([np.inf, -np.inf], np.nan).dropna()
    zq = np.quantile(z, quantiles)
    scale = (frame["vol_168h"] * np.sqrt(config.HORIZON_HOURS)).to_numpy()[:, None]
    return scale * zq[None, :]

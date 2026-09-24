"""Turn predictions into daily portfolios and honest, after-cost returns.

EN: The portfolio is deliberately plain: each day, go long the top fifth of
    coins by prediction and short the bottom fifth, equal weight, dollar
    neutral (+1 on the long leg, -1 on the short leg). Plain on purpose: a
    fancy optimiser can hide a weak signal, and I want the signal to be what is
    being measured. Holding periods run t+1h -> t+25h, so consecutive days
    chain without gaps or overlaps.
    Simplification I state rather than hide: weights are reset to target every
    day and intraday drift of weights is ignored when computing turnover.
TR: Portföy bilerek sade: her gün tahmine göre coinlerin en iyi beşte birini
    long, en kötü beşte birini short yap; eşit ağırlıklı, dollar-neutral (long
    bacakta +1, short bacakta -1). Sade olması bilinçli: süslü bir optimizer
    zayıf bir sinyali gizleyebilir ve ölçülen şeyin sinyal olmasını istiyorum.
    Elde tutma süreleri t+1s -> t+25s; dolayısıyla ardışık günler boşluksuz ve
    örtüşmesiz birbirine bağlanıyor.
    Gizlemek yerine açıkça söylediğim bir basitleştirme: ağırlıklar her gün
    hedefe sıfırlanıyor ve turnover hesaplanırken gün içi ağırlık kayması
    yok sayılıyor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src import metrics as M


def target_weights(
    pred: pd.Series,
    mode: str = "long_short",
    buckets: int = config.QUANTILE_BUCKETS,
) -> pd.DataFrame:
    """Daily target weights, wide (time x symbol).

    Modes:
      long_short  +1/k on the top k, -1/k on the bottom k (k = n // buckets)
      long_only   +1/k on the top k
      benchmark   +1/n on every coin (the equal-weight universe)
      scaled      weights proportional to the demeaned score, gross exposure 2
    """
    pred = pred.dropna()
    wide = pred.unstack("symbol")
    n = wide.notna().sum(axis=1)

    if mode == "benchmark":
        return wide.notna().astype(float).div(n, axis=0)
    if mode == "scaled":
        centred = wide.sub(wide.mean(axis=1), axis=0)
        gross = centred.abs().sum(axis=1).replace(0, np.nan)
        return (2.0 * centred.div(gross, axis=0)).fillna(0.0)

    k = (n // buckets).clip(lower=1)
    rank_desc = wide.rank(axis=1, ascending=False, method="first")
    rank_asc = wide.rank(axis=1, ascending=True, method="first")
    long = rank_desc.le(k, axis=0).astype(float).div(k, axis=0)
    if mode == "long_only":
        return long
    if mode == "long_short":
        short = rank_asc.le(k, axis=0).astype(float).div(k, axis=0)
        return long - short
    raise ValueError(f"unknown mode {mode!r}")


def run_backtest(
    pred: pd.Series,
    log_returns: pd.Series,
    mode: str = "long_short",
    cost_bps: float = config.COST_BPS,
) -> pd.DataFrame:
    """Daily gross/net returns, turnover and cost for one prediction series.

    EN: `log_returns` are the realised t+1h -> t+25h log returns; the portfolio
        return uses simple returns because portfolio returns add up in simple,
        not log, space. Cost = turnover x cost per side.
    TR: `log_returns`, gerçekleşmiş t+1s -> t+25s log getirileri; portföy
        getirisi basit getirilerle hesaplanıyor, çünkü portföy getirileri log
        uzayında değil basit uzayda toplanıyor. Maliyet = turnover x taraf
        başına maliyet.
    """
    weights = target_weights(pred, mode).fillna(0.0)
    simple = np.expm1(log_returns).unstack("symbol").reindex_like(weights).fillna(0.0)

    gross = (weights * simple).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    cost = turnover * cost_bps / 1e4
    return pd.DataFrame(
        {
            "gross": gross,
            "cost": cost,
            "net": gross - cost,
            "turnover": turnover,
            "n_long": (weights > 0).sum(axis=1),
            "n_short": (weights < 0).sum(axis=1),
        }
    )


def summarize(bt: pd.DataFrame, column: str = "net") -> dict:
    r = bt[column]
    return {
        "ann_return": M.annual_return(r),
        "sharpe": M.sharpe(r),
        "max_drawdown": M.max_drawdown(r),
        "avg_turnover": float(bt["turnover"].mean()),
    }


def cost_sensitivity(pred: pd.Series, log_returns: pd.Series,
                     grid=config.COST_GRID_BPS) -> pd.DataFrame:
    """Sharpe and annual return of the long-short book at several cost levels."""
    rows = []
    for bps in grid:
        s = summarize(run_backtest(pred, log_returns, "long_short", bps))
        rows.append({"cost_bps": bps, "sharpe": s["sharpe"], "ann_return": s["ann_return"]})
    return pd.DataFrame(rows).set_index("cost_bps")


def quintile_returns(pred: pd.Series, log_returns: pd.Series,
                     buckets: int = config.QUANTILE_BUCKETS) -> pd.Series:
    """Mean next-day simple return per prediction bucket (1 = lowest)."""
    df = pd.DataFrame({"p": pred, "y": np.expm1(log_returns)}).dropna()
    pct = df.groupby(level="time")["p"].rank(pct=True, method="first")
    df["bucket"] = np.ceil(pct * buckets).clip(1, buckets).astype(int)
    daily = df.groupby([df.index.get_level_values("time"), "bucket"])["y"].mean()
    return daily.groupby(level="bucket").mean()

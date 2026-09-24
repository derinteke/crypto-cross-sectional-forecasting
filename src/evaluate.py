"""Figures for the notebook and the README.

EN: Each figure answers one question a reviewer would ask:
    equity curves     -> does it make money after costs, and when?
    rolling IC        -> is the edge stable or one lucky period?
    quintile ladder   -> is the ranking monotonic, or only the extremes work?
    IC decay          -> how fast does the signal go stale?
    cost sensitivity  -> at what fee level does it stop working?
    calibration       -> are the quantile forecasts honest?
TR: Her grafik bir inceleyicinin soracağı tek bir soruya cevap veriyor:
    sermaye eğrileri    -> maliyetlerden sonra para kazandırıyor mu, ne zaman?
    kayan IC            -> avantaj kararlı mı, yoksa tek bir şanslı dönem mi?
    quintile merdiveni  -> sıralama monoton mu, yoksa yalnızca uçlar mı çalışıyor?
    IC sönümü           -> sinyal ne kadar hızlı bayatlıyor?
    maliyet duyarlılığı -> hangi komisyon seviyesinde çalışmayı bırakıyor?
    kalibrasyon         -> quantile tahminleri dürüst mü?
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, filename: str | None):
    if filename:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES_DIR / filename, bbox_inches="tight")
    return fig


def plot_equity_curves(books: dict[str, pd.DataFrame], models: list[str],
                       benchmark: pd.Series | None = None, column: str = "net",
                       title: str = "", filename: str | None = None):
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in models:
        ax.plot((1 + books[name][column]).cumprod(), label=name, lw=1.4)
    if benchmark is not None:
        ax.plot((1 + benchmark).cumprod(), label="Equal-weight universe (long only)",
                color="grey", lw=1, ls="--")
    ax.set_yscale("log")
    ax.axhline(1.0, color="black", lw=0.6)
    ax.set_ylabel("growth of 1 (log scale)")
    ax.set_title(title or f"Long-short book, {column} of {config.COST_BPS:.0f} bps/side")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, filename)


def plot_rolling_ic(ics: dict[str, pd.Series], window: int = 90, filename: str | None = None):
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, ic in ics.items():
        ax.plot(ic.rolling(window, min_periods=window // 2).mean(), label=name, lw=1.3)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel(f"{window}-day mean rank IC")
    ax.set_title("Is the edge stable? Rolling rank IC out of sample")
    ax.legend(fontsize=8, ncol=3)
    return _save(fig, filename)


def plot_quintiles(ladders: dict[str, pd.Series], filename: str | None = None):
    names = list(ladders)
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        vals = ladders[name] * 1e4
        ax.bar(vals.index.astype(str), vals.to_numpy(),
               color=["#c44e52" if v < 0 else "#4c72b0" for v in vals])
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("prediction quintile")
    axes[0].set_ylabel("mean next-day return (bps)")
    fig.suptitle("Quintile ladder: a real signal climbs monotonically", y=1.03)
    return _save(fig, filename)


def plot_ic_decay(table: pd.DataFrame, models: list[str], filename: str | None = None):
    cols = ["ic_mean"] + [f"ic_day{k}" for k in range(1, config.IC_DECAY_DAYS)]
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in models:
        ax.plot(range(len(cols)), table.loc[name, cols].astype(float), marker="o", label=name)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(range(len(cols)), [f"day +{k}" for k in range(len(cols))])
    ax.set_ylabel("mean rank IC")
    ax.set_title("Signal decay: IC against returns k days later")
    ax.legend(fontsize=8)
    return _save(fig, filename)


def plot_cost_sensitivity(curves: dict[str, pd.DataFrame], filename: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, df in curves.items():
        ax.plot(df.index, df["sharpe"], marker="o", label=name)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("cost per side (bps)")
    ax.set_ylabel("annualised Sharpe (net)")
    ax.set_title("How much trading cost can the signal carry?")
    ax.legend(fontsize=8)
    return _save(fig, filename)


def plot_calibration(calib: dict[str, pd.DataFrame], filename: str | None = None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--", label="perfect")
    for name, df in calib.items():
        ax.plot(df["nominal"], df["observed"], marker="o", label=name)
    ax.set_xlabel("nominal quantile")
    ax.set_ylabel("share of outcomes below the forecast")
    ax.set_title("Quantile calibration (out of sample)")
    ax.legend(fontsize=8)
    return _save(fig, filename)


def plot_training_curves(histories: dict, name: str, filename: str | None = None):
    runs = {fold: h for (m, fold), h in histories.items() if m == name}
    fig, ax = plt.subplots(figsize=(8, 4))
    for fold, h in sorted(runs.items()):
        ax.plot(h["val_loss"], label=f"fold {fold}", lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss")
    ax.set_title(f"{name}: validation loss per fold")
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, filename)


def plot_universe(universe_table: pd.DataFrame, top: int = 40, filename: str | None = None):
    """Which coins were in the top 30, month by month: shows the churn."""
    pivot = (universe_table.assign(v=1)
             .pivot_table(index="symbol", columns="month", values="v", fill_value=0))
    order = pivot.sum(axis=1).sort_values(ascending=False).index[:top]
    fig, ax = plt.subplots(figsize=(12, 0.18 * len(order) + 1.5))
    ax.imshow(pivot.loc[order].to_numpy(), aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_yticks(range(len(order)), [s.replace("USDT", "") for s in order], fontsize=7)
    months = pd.to_datetime(pivot.columns)
    ticks = [i for i, m in enumerate(months) if m.month == 1]
    ax.set_xticks(ticks, [months[i].year for i in ticks])
    ax.grid(False)
    ax.set_title(f"Point-in-time universe: the {top} coins that spent longest in the top 30")
    return _save(fig, filename)

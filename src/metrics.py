"""Signal, portfolio and probabilistic metrics.

EN: I report the metrics a quant desk would ask for, not the ones a Kaggle
    leaderboard would. MSE on returns is almost meaningless here: a model that
    predicts zero for everything has a near-perfect MSE and zero value. What
    matters is whether the ranking is right (rank IC), whether that holds up
    day after day (ICIR, Newey-West t-stat), and whether it survives trading
    costs and the number of things I tried (deflated Sharpe).
TR: Bir Kaggle sıralamasının değil, bir quant masasının soracağı metrikleri
    raporluyorum. Getiriler üzerinde MSE burada neredeyse anlamsız: her şeye
    sıfır diyen bir model neredeyse mükemmel MSE alır ve sıfır değer üretir.
    Önemli olan sıralamanın doğru olup olmadığı (rank IC), bunun gün be gün
    sürüp sürmediği (ICIR, Newey-West t-istatistiği) ve işlem maliyetlerine ve
    denediğim şeylerin sayısına dayanıp dayanmadığı (deflated Sharpe).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import config

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Signal quality / Sinyal kalitesi
# --------------------------------------------------------------------------- #
def daily_rank_ic(pred: pd.Series, target: pd.Series) -> pd.Series:
    """Spearman correlation between prediction and outcome, one value per day."""
    df = pd.DataFrame({"p": pred, "y": target}).dropna()
    ranked = df.groupby(level="time").rank()
    ic = ranked.groupby(level="time").apply(
        lambda g: g["p"].corr(g["y"]) if len(g) >= 3 else np.nan
    )
    return ic.dropna()


def newey_west_tstat(x: pd.Series | np.ndarray, lags: int = 5) -> float:
    """t-stat of the mean with a Newey-West (Bartlett) long-run variance.

    EN: Daily ICs are autocorrelated (yesterday's momentum is today's
        momentum), so the plain t-stat overstates significance. Newey-West
        widens the error bar to account for that.
    TR: Günlük IC'ler otokorelasyonlu (dünün momentumu bugünün momentumu),
        dolayısıyla düz t-istatistiği anlamlılığı abartıyor. Newey-West hata
        payını bunu hesaba katacak şekilde genişletiyor.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    lrv = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        lrv += 2.0 * weight * (e[lag:] @ e[:-lag]) / n
    if lrv <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(lrv / n))


def ic_summary(ic: pd.Series) -> dict:
    return {
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std()),
        "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else float("nan"),
        "ic_tstat_nw": newey_west_tstat(ic),
        "ic_hit_rate": float((ic > 0).mean()),
        "n_days": int(len(ic)),
    }


# --------------------------------------------------------------------------- #
# Returns / Getiriler
# --------------------------------------------------------------------------- #
def sharpe(returns: pd.Series, periods: int = config.PERIODS_PER_YEAR) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(periods))


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough fall of the compounded equity curve (negative)."""
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def annual_return(returns: pd.Series, periods: int = config.PERIODS_PER_YEAR) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    growth = float((1.0 + r).prod())
    return growth ** (periods / len(r)) - 1.0 if growth > 0 else -1.0


def deflated_sharpe_ratio(
    returns: pd.Series, trial_sharpes: list[float] | np.ndarray
) -> dict:
    """Bailey & López de Prado (2014) deflated Sharpe ratio.

    EN: If I try N strategies, the best of them will look good by luck alone.
        DSR asks: given how many trials I ran and how much their Sharpes
        varied, what is the probability that this strategy's true Sharpe is
        above zero? It also corrects for fat tails and skew, which crypto has in
        abundance. Sharpes here are per period (daily), not annualised.
    TR: N strateji denersem, en iyisi yalnızca şans eseri iyi görünecektir.
        DSR şunu soruyor: kaç deneme yaptığım ve Sharpe'larının ne kadar
        değiştiği göz önüne alındığında, bu stratejinin gerçek Sharpe'ının
        sıfırın üzerinde olma olasılığı nedir? Ayrıca kriptoda bolca bulunan
        kalın kuyrukları ve çarpıklığı da düzeltiyor. Buradaki Sharpe'lar
        dönem başına (günlük), yıllıklandırılmış değil.
    """
    r = returns.dropna().to_numpy()
    t = len(r)
    sr = r.mean() / r.std(ddof=1)
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)

    trials = np.asarray([s for s in trial_sharpes if np.isfinite(s)], dtype=float)
    n_trials = max(len(trials), 1)
    var_trials = trials.var(ddof=1) if len(trials) > 1 else 0.0
    if n_trials > 1 and var_trials > 0:
        sr0 = np.sqrt(var_trials) * (
            (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1.0 / n_trials)
            + EULER_GAMMA * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
    else:
        sr0 = 0.0
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4.0 * sr**2, 1e-12))
    dsr = stats.norm.cdf((sr - sr0) * np.sqrt(t - 1) / denom)
    return {
        "sharpe_daily": float(sr),
        "sharpe_threshold_daily": float(sr0),
        "n_trials": int(n_trials),
        "deflated_sharpe": float(dsr),
    }


def market_regression(returns: pd.Series, market: pd.Series, lags: int = 5) -> dict:
    """Regress a strategy on the market: how much is beta, how much is alpha?

    EN: A dollar-neutral book is not beta-neutral. If it is long BTC and short
        meme coins, it quietly bets on altcoins falling, and in 2022-26 they
        did. The intercept (with Newey-West errors) is the part of the return
        the market does not explain.
    TR: Dollar-neutral bir portföy beta-neutral değildir. BTC'yi long, meme
        coinleri short ediyorsa, sessizce altcoinlerin düşeceğine bahse giriyor
        demektir ve 2022-26'da düştüler. Kesim terimi (Newey-West hatalarıyla),
        getirinin piyasanın açıklamadığı kısmı.
    """
    import statsmodels.api as sm

    df = pd.DataFrame({"r": returns, "m": market}).dropna()
    fit = sm.OLS(df["r"], sm.add_constant(df["m"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    return {
        "beta_to_market": float(fit.params["m"]),
        "alpha_bps_per_day": float(fit.params["const"] * 1e4),
        "alpha_tstat": float(fit.tvalues["const"]),
    }


# --------------------------------------------------------------------------- #
# Probabilistic / Olasılıksal
# --------------------------------------------------------------------------- #
def pinball_loss(y: np.ndarray, q_pred: np.ndarray, quantiles=config.QUANTILES) -> np.ndarray:
    """Mean pinball loss per quantile. q_pred has shape (n, n_quantiles)."""
    y = np.asarray(y, dtype=float)[:, None]
    q = np.asarray(quantiles, dtype=float)[None, :]
    diff = y - q_pred
    return np.mean(np.maximum(q * diff, (q - 1) * diff), axis=0)


def calibration(y: np.ndarray, q_pred: np.ndarray, quantiles=config.QUANTILES) -> pd.DataFrame:
    """Share of outcomes below each predicted quantile. Perfect = the quantile."""
    y = np.asarray(y, dtype=float)[:, None]
    below = (y <= q_pred).mean(axis=0)
    return pd.DataFrame({"nominal": quantiles, "observed": below})


def interval_coverage(y: np.ndarray, q_pred: np.ndarray, quantiles=config.QUANTILES,
                      lo: float = 0.1, hi: float = 0.9) -> float:
    qs = list(np.round(quantiles, 6))
    i, j = qs.index(round(lo, 6)), qs.index(round(hi, 6))
    y = np.asarray(y, dtype=float)
    return float(((y >= q_pred[:, i]) & (y <= q_pred[:, j])).mean())

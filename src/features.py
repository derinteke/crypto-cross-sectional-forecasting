"""Features and targets, with the time boundary written into the code.

EN: Every decision happens at t = 00:00 UTC. The rule is simple and absolute:
    features may use bars that CLOSED at or before t; targets start one hour
    later. Everything here is a causal rolling window or a backward shift, and
    the only forward shifts in the file are in `targets()`, which is kept
    separate on purpose so a reviewer can check the boundary in one place.
    I enter at the close of the bar ending t+1 (one hour of execution lag: a
    signal computed at 00:00 cannot also be filled at 00:00) and exit 24 hours
    later.
TR: Her karar t = 00:00 UTC'de veriliyor. Kural basit ve kesin: özellikler
    t'de ya da öncesinde KAPANMIŞ barları kullanabilir; hedefler bir saat sonra
    başlıyor. Buradaki her şey nedensel bir kayan pencere ya da geriye doğru bir
    kaydırma; dosyadaki tek ileri kaydırmalar `targets()` içinde ve bu fonksiyon
    bilerek ayrı tutuldu ki bir inceleyici sınırı tek yerde kontrol edebilsin.
    t+1'de biten barın kapanışında giriyorum (bir saatlik icra gecikmesi: 00:00'da
    hesaplanan bir sinyal 00:00'da gerçekleştirilemez) ve 24 saat sonra çıkıyorum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import config
from src.data import Panel
from src.universe import membership

MARKET_SYMBOL = "BTCUSDT"

# EN: per-coin features; each one also gets a cross-sectional rank version.
# TR: coin başına özellikler; her birinin kesitsel sıra versiyonu da üretiliyor.
COIN_FEATURES = [
    "ret_1h", "ret_4h", "ret_24h", "ret_72h", "ret_168h", "ret_720h",
    "vol_24h", "vol_168h", "hl_range_24h", "volume_z", "amihud", "beta_720h",
]
# EN: market context, identical for every coin on a given day.
# TR: piyasa bağlamı; belli bir günde her coin için aynı.
MARKET_FEATURES = ["btc_ret_24h", "btc_ret_168h", "btc_vol_168h"]
RANK_FEATURES = [f"rank_{f}" for f in COIN_FEATURES]
ALL_FEATURES = COIN_FEATURES + MARKET_FEATURES + RANK_FEATURES

SEQ_CHANNELS = ["ret_1h", "hl_range", "log_volume_rel", "btc_ret_1h", "excess_ret_1h"]


# --------------------------------------------------------------------------- #
# Hourly building blocks (all causal) / Saatlik yapı taşları (hepsi nedensel)
# --------------------------------------------------------------------------- #
def _log_prices(panel: Panel) -> pd.DataFrame:
    # EN: forward-fill only carries the LAST KNOWN price forward in time; it
    #     never looks ahead. Missing hours are masked out of the samples later.
    # TR: ileri doldurma yalnızca BİLİNEN SON fiyatı zamanda ileri taşıyor;
    #     asla ileriye bakmıyor. Eksik saatler daha sonra örneklerden eleniyor.
    return np.log(panel.close.ffill())


def hourly_features(panel: Panel) -> dict[str, pd.DataFrame]:
    """Wide (hour x symbol) panels of every coin feature. Strictly causal."""
    logp = _log_prices(panel)
    r1 = logp.diff()
    qv = panel.fields["quote_volume"].fillna(0.0)
    hl = np.log(panel.fields["high"] / panel.fields["low"])

    feats = {f"ret_{k}h": logp - logp.shift(k) for k in (1, 4, 24, 72, 168, 720)}
    feats["vol_24h"] = r1.rolling(24, min_periods=20).std()
    feats["vol_168h"] = r1.rolling(168, min_periods=140).std()
    feats["hl_range_24h"] = hl.rolling(24, min_periods=20).mean()

    daily_avg_720 = qv.rolling(720, min_periods=600).sum() / 30.0
    feats["volume_z"] = np.log1p(qv.rolling(24).sum()) - np.log1p(daily_avg_720)
    feats["amihud"] = np.log(
        (r1.abs() / (qv + 1.0)).rolling(24, min_periods=20).mean() + 1e-12
    )

    btc = r1[MARKET_SYMBOL]
    mean_x = r1.rolling(720, min_periods=600).mean()
    mean_b = btc.rolling(720, min_periods=600).mean()
    cov = r1.mul(btc, axis=0).rolling(720, min_periods=600).mean().sub(
        mean_x.mul(mean_b, axis=0)
    )
    var_b = btc.rolling(720, min_periods=600).var(ddof=0)
    feats["beta_720h"] = cov.div(var_b, axis=0)
    return feats


def market_features(panel: Panel) -> pd.DataFrame:
    logp = _log_prices(panel)[MARKET_SYMBOL]
    r1 = logp.diff()
    return pd.DataFrame(
        {
            "btc_ret_24h": logp - logp.shift(24),
            "btc_ret_168h": logp - logp.shift(168),
            "btc_vol_168h": r1.rolling(168, min_periods=140).std(),
        }
    )


def sequence_channels(panel: Panel) -> np.ndarray:
    """(hour, symbol, channel) float32 array for the sequence models."""
    logp = _log_prices(panel)
    r1 = logp.diff()
    qv = panel.fields["quote_volume"].fillna(0.0)
    hourly_avg_720 = qv.rolling(720, min_periods=600).mean()
    btc = r1[MARKET_SYMBOL]
    channels = [
        r1,
        np.log(panel.fields["high"] / panel.fields["low"]),
        np.log1p(qv) - np.log1p(hourly_avg_720),
        pd.DataFrame({s: btc for s in r1.columns}),
        r1.sub(btc, axis=0),
    ]
    arr = np.stack([c.to_numpy(dtype=np.float32) for c in channels], axis=-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# Targets (the ONLY forward-looking code) / Hedefler (TEK ileriye bakan kod)
# --------------------------------------------------------------------------- #
def targets(
    panel: Panel,
    horizon: int = config.HORIZON_HOURS,
    lag: int = config.EXECUTION_LAG_HOURS,
    decay_days: int = config.IC_DECAY_DAYS,
) -> dict[str, pd.DataFrame]:
    """Forward log returns from t+lag to t+lag+horizon, plus later days for IC decay.

    EN: The exit price is forward-filled: if a coin is delisted mid-holding, I
        am assumed to exit at its last traded price, which is what would happen
        in reality. The entry, in contrast, must be a real traded bar (checked
        in `build_samples`): I cannot buy a coin that is not trading.
    TR: Çıkış fiyatı ileri dolduruluyor: bir coin elde tutarken delist edilirse
        son işlem gördüğü fiyattan çıktığımı varsayıyorum; gerçekte olacak olan
        da bu. Girişin ise gerçek bir işlem barı olması şart (`build_samples`
        içinde kontrol ediliyor): işlem görmeyen bir coini alamam.
    """
    logp = _log_prices(panel)
    out = {}
    for k in range(decay_days):
        start = lag + k * horizon
        out[f"y_day{k}"] = logp.shift(-(start + horizon)) - logp.shift(-start)
    out["entry_traded"] = panel.traded.shift(-lag, fill_value=False)
    return out


# --------------------------------------------------------------------------- #
# Samples / Örnekler
# --------------------------------------------------------------------------- #
@dataclass
class Samples:
    """One row per (decision time, coin) plus the hourly arrays behind them."""

    frame: pd.DataFrame        # index (time, symbol); features, targets, positions
    seq: np.ndarray            # (hour, symbol, channel) for the sequence models
    hours: pd.DatetimeIndex    # hourly grid the positions refer to
    symbols: list[str]

    @property
    def times(self) -> pd.DatetimeIndex:
        return self.frame.index.get_level_values("time").unique()


def _cross_sectional_rank(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Percentile rank within each day, centred on 0 (range about -0.5..0.5)."""
    return df.groupby(level="time")[cols].rank(pct=True) - 0.5


def _rank_gauss(s: pd.Series) -> pd.Series:
    """Map each day's cross-section onto a standard normal via its ranks."""
    ranks = s.groupby(level="time").rank()
    n = s.groupby(level="time").transform("count")
    return pd.Series(norm.ppf((ranks - 0.5) / n), index=s.index)


def build_samples(
    panel: Panel,
    universe: dict[pd.Timestamp, list[str]],
    start: str | None = None,
    end: str | None = None,
    seq_lookback: int = config.SEQ_LOOKBACK_HOURS,
) -> Samples:
    """Assemble the modelling table.

    EN: A (t, coin) pair becomes a sample only if: the coin is in that month's
        universe, it traded in every one of the last `seq_lookback` hours (so no
        window is built on missing data), it had no multi-day halt within the
        30-day feature warm-up (so no feature spans a token swap), and its entry
        bar at t+1 really traded. The same sample set is used by every model, so every comparison
        is on identical rows.
    TR: Bir (t, coin) çifti yalnızca şu koşullarda örnek oluyor: coin o ayın
        evreninde, son `seq_lookback` saatin her birinde işlem görmüş (böylece
        hiçbir pencere eksik veri üzerine kurulmuyor), 30 günlük özellik ısınma
        süresi içinde çok günlü bir durdurma yaşamamış (böylece hiçbir özellik
        bir token swap'ını aşmıyor) ve t+1'deki giriş barı gerçekten işlem görmüş. Aynı örnek kümesi her model tarafından
        kullanılıyor; dolayısıyla her karşılaştırma birebir aynı satırlar
        üzerinde.
    """
    hours = panel.index
    is_decision = hours.hour == config.DECISION_HOUR_UTC
    if start is not None:
        is_decision &= hours >= pd.Timestamp(start)
    if end is not None:
        is_decision &= hours <= pd.Timestamp(end)
    decision_times = hours[is_decision]

    feats = hourly_features(panel)
    mkt = market_features(panel)
    tgt = targets(panel)
    members = membership(universe, hours, panel.symbols)
    full_window = panel.traded.rolling(seq_lookback, min_periods=seq_lookback).sum() == seq_lookback

    # EN: no long halt (a swap / relaunch) anywhere in the feature warm-up.
    # TR: özellik ısınma süresinin hiçbir yerinde uzun bir durdurma (swap /
    #     yeniden çıkış) yok.
    long_gap = panel.traded.rolling(config.LONG_GAP_HOURS).sum() == 0
    recent_gap = long_gap.rolling(config.FEATURE_WARMUP_HOURS + config.LONG_GAP_HOURS,
                                  min_periods=1).max().astype(bool)

    valid = members & full_window & ~recent_gap & panel.traded & tgt["entry_traded"]
    valid = valid.loc[decision_times]

    def stack(df: pd.DataFrame) -> pd.Series:
        return df.loc[decision_times].where(valid).stack(future_stack=False)

    cols = {name: stack(df) for name, df in feats.items()}
    for name in [k for k in tgt if k.startswith("y_day")]:
        cols[name] = tgt[name].loc[decision_times].where(valid).stack(future_stack=False)
    frame = pd.DataFrame({k: cols[k] for k in COIN_FEATURES})
    for name in [k for k in tgt if k.startswith("y_day")]:
        frame[name] = cols[name].reindex(frame.index)
    frame.index.names = ["time", "symbol"]

    for c in MARKET_FEATURES:
        frame[c] = mkt[c].reindex(frame.index.get_level_values("time")).to_numpy()

    # EN: drop rows whose features are undefined (young coins, warm-up period)
    #     or whose day-0 target is unknown (the last day of the data).
    # TR: özellikleri tanımsız (genç coin, ısınma dönemi) ya da 0. gün hedefi
    #     bilinmeyen (verinin son günü) satırları at.
    finite = np.isfinite(frame[COIN_FEATURES + MARKET_FEATURES]).all(axis=1)
    frame = frame[finite & frame["y_day0"].notna()].copy()

    frame[RANK_FEATURES] = _cross_sectional_rank(frame, COIN_FEATURES).to_numpy()
    day_mean = frame.groupby(level="time")["y_day0"].transform("mean")
    frame["y_demeaned"] = frame["y_day0"] - day_mean
    frame["y_rankgauss"] = _rank_gauss(frame["y_day0"])

    # EN: positions let the sequence dataset slice the hourly array directly.
    # TR: konumlar, dizi veri setinin saatlik diziyi doğrudan dilimlemesini sağlıyor.
    sym_pos = {s: i for i, s in enumerate(panel.symbols)}
    frame["hour_pos"] = hours.get_indexer(frame.index.get_level_values("time"))
    frame["sym_pos"] = [sym_pos[s] for s in frame.index.get_level_values("symbol")]

    return Samples(frame=frame.sort_index(), seq=sequence_channels(panel), hours=hours,
                   symbols=panel.symbols)

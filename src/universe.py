"""Point-in-time universe: the 30 most liquid coins, re-picked every month.

EN: The universe is where most crypto backtests leak without anyone noticing.
    Picking "the top 30 coins" using today's market caps means every coin in the
    study is, by construction, one that survived and grew. Here the list for
    month M is chosen on the first day of M, using only the 30 days of volume
    before it. Coins that later collapsed or were delisted (LUNA, FTT, ...) are
    in the universe exactly when a trader at the time would have held them.
TR: Kripto backtest'lerinin çoğu, kimse fark etmeden, evrende sızıntı yapıyor.
    "İlk 30 coin"i bugünün piyasa değerlerine göre seçmek, çalışmadaki her
    coinin tanım gereği hayatta kalmış ve büyümüş olması demek. Burada M ayının
    listesi M'nin ilk gününde, yalnızca ondan önceki 30 günün hacmiyle seçiliyor.
    Sonradan çöken ya da delist edilen coinler (LUNA, FTT, ...) evrende tam da o
    dönemdeki bir trader'ın onları tutacağı zamanlarda yer alıyor.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.data import Panel


def select_universe(
    daily: Panel,
    month_start: pd.Timestamp,
    size: int = config.UNIVERSE_SIZE,
    lookback_days: int = config.UNIVERSE_LOOKBACK_DAYS,
    min_history_days: int = config.MIN_HISTORY_DAYS,
    min_coverage: float = config.MIN_COVERAGE,
) -> list[str]:
    """Top `size` symbols by mean USDT volume over the days before `month_start`.

    EN: Daily bars are indexed by close time, so the bar that closes exactly at
        `month_start` (the previous day) is known at the decision and included;
        nothing that closes later is.
    TR: Günlük barlar kapanış zamanıyla indekslendiği için tam `month_start`
        anında kapanan bar (önceki gün) karar anında biliniyor ve dahil
        ediliyor; daha sonra kapanan hiçbir şey dahil edilmiyor.
    """
    window = slice(month_start - pd.Timedelta(days=lookback_days - 1), month_start)
    traded = daily.traded.loc[window]
    if len(traded) == 0:
        return []
    volume = daily.fields["quote_volume"].loc[window].fillna(0.0)

    history = daily.traded.loc[:month_start]
    first_seen = history.idxmax().where(history.any())
    old_enough = first_seen <= month_start - pd.Timedelta(days=min_history_days)
    covered = traded.mean() >= min_coverage
    alive = traded.iloc[-1]

    eligible = old_enough & covered & alive
    ranked = volume.mean()[eligible].sort_values(ascending=False)
    return list(ranked.index[:size])


def build_universe(daily: Panel, start: str, end: str, **kwargs) -> dict[pd.Timestamp, list[str]]:
    """Universe for every month start in [start, end]."""
    months = pd.date_range(pd.Timestamp(start).normalize(), end, freq="MS")
    return {m: select_universe(daily, m, **kwargs) for m in months}


def universe_frame(universe: dict[pd.Timestamp, list[str]]) -> pd.DataFrame:
    """Long table: month, rank, symbol. Handy for the README and for tests."""
    rows = [
        {"month": m, "rank": r + 1, "symbol": s}
        for m, syms in universe.items()
        for r, s in enumerate(syms)
    ]
    return pd.DataFrame(rows)


def membership(universe: dict[pd.Timestamp, list[str]], index: pd.DatetimeIndex,
               symbols: list[str]) -> pd.DataFrame:
    """Boolean (time x symbol): was the symbol in the universe at time t?

    EN: A timestamp belongs to the month it falls in, and the list for that month
        was fixed at its first instant, so membership never depends on the
        future.
    TR: Bir zaman damgası içine düştüğü aya ait ve o ayın listesi ayın ilk
        anında sabitlendi; dolayısıyla üyelik asla geleceğe bağlı değil.
    """
    months = index.to_period("M").to_timestamp()
    out = pd.DataFrame(False, index=index, columns=symbols)
    for m, syms in universe.items():
        rows = months == m
        cols = [s for s in syms if s in out.columns]
        if rows.any() and cols:
            out.loc[rows, cols] = True
    return out

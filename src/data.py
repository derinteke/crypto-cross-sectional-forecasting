"""Turn the raw monthly zips into clean wide panels (time x symbol).

EN: Every downstream step (universe, features, targets, backtest) works on wide
    panels: one DataFrame per field, rows are timestamps and columns are symbols.
    That shape makes cross-sectional operations one-liners and, more
    importantly, makes it hard to accidentally mix up timestamps between coins.
    Two conventions matter for leakage and I fix them here once:
      1. A bar is indexed by its CLOSE time. `close.loc[t]` is the last price
         known at time t, so "use data <= t" really means what it says.
      2. Missing bars stay NaN. I never interpolate prices: an invented price is
         an invented return. A separate mask records which bars really traded.
TR: Sonraki her adım (evren, özellikler, hedefler, backtest) geniş paneller
    üzerinde çalışıyor: alan başına bir DataFrame, satırlar zaman damgası,
    sütunlar sembol. Bu biçim kesitsel işlemleri tek satıra indiriyor ve daha
    önemlisi coinler arasında zaman damgalarını yanlışlıkla karıştırmayı
    zorlaştırıyor. Sızıntı açısından önemli iki kuralı burada bir kez sabitliyorum:
      1. Bir bar KAPANIŞ zamanıyla indeksleniyor. `close.loc[t]`, t anında
         bilinen son fiyat; dolayısıyla "<= t verisini kullan" tam olarak
         söylediği anlama geliyor.
      2. Eksik barlar NaN kalıyor. Fiyatları asla interpole etmiyorum: uydurulmuş
         bir fiyat, uydurulmuş bir getiridir. Ayrı bir maske hangi barların
         gerçekten işlem gördüğünü kaydediyor.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
PANEL_FIELDS = ("open", "high", "low", "close", "quote_volume", "trades")
_INTERVAL = {"1h": pd.Timedelta(hours=1), "1d": pd.Timedelta(days=1)}


# --------------------------------------------------------------------------- #
# Reading raw files / Ham dosyaları okuma
# --------------------------------------------------------------------------- #
def _to_timestamp(values: pd.Series) -> pd.DatetimeIndex:
    """Binance switched spot files from milliseconds to microseconds in 2025.

    EN: Reading a microsecond stamp as milliseconds lands in the year 50,000+,
        so I detect the unit per value instead of trusting the file date.
    TR: Mikrosaniyelik bir damgayı milisaniye diye okumak 50.000'li yıllara
        düşüyor; bu yüzden birime dosya tarihine güvenmek yerine değer bazında
        karar veriyorum.
    """
    v = values.astype("int64").to_numpy()
    v = np.where(v > 10**14, v // 1000, v)
    return pd.to_datetime(v, unit="ms", utc=True).tz_convert(None)


def read_kline_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        raw = zf.read(zf.namelist()[0])
    first = raw.split(b"\n", 1)[0]
    header = 0 if first[:1].isalpha() else None
    df = pd.read_csv(io.BytesIO(raw), header=header, names=KLINE_COLUMNS)
    df["open_time"] = _to_timestamp(df["open_time"])
    return df


def load_symbol(symbol: str, interval: str) -> pd.DataFrame:
    """All months of one symbol, indexed by bar CLOSE time, duplicates removed."""
    files = sorted((config.RAW_DIR / interval / symbol).glob("*.zip"))
    if not files:
        return pd.DataFrame(columns=list(PANEL_FIELDS))
    df = pd.concat([read_kline_zip(f) for f in files], ignore_index=True)
    df = df.drop_duplicates("open_time").sort_values("open_time")
    df.index = pd.DatetimeIndex(df["open_time"] + _INTERVAL[interval], name="time")
    return df[list(PANEL_FIELDS)].astype("float64")


# --------------------------------------------------------------------------- #
# Panels / Paneller
# --------------------------------------------------------------------------- #
@dataclass
class Panel:
    """Wide panels of one bar interval plus the mask of real bars."""

    fields: dict[str, pd.DataFrame]
    traded: pd.DataFrame  # True where a real bar with volume exists

    @property
    def close(self) -> pd.DataFrame:
        return self.fields["close"]

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    def restrict(self, symbols=None, start=None, end=None) -> "Panel":
        """Sub-panel by symbols and/or time range (inclusive)."""
        cols = list(symbols) if symbols is not None else self.symbols
        sl = slice(start, end)
        return Panel(
            fields={k: v.loc[sl, cols] for k, v in self.fields.items()},
            traded=self.traded.loc[sl, cols],
        )


def build_panel(symbols: list[str], interval: str, verbose: bool = True) -> Panel:
    """Load symbols onto one regular grid. Missing bars stay NaN.

    EN: A bar counts as traded only if it exists AND had non-zero volume. Binance
        sometimes publishes zero-volume bars with a copied price during
        maintenance; treating those as real would create fake flat returns.
    TR: Bir bar yalnızca var olduğunda VE hacmi sıfırdan büyük olduğunda işlem
        görmüş sayılıyor. Binance bakım sırasında bazen fiyatı kopyalanmış sıfır
        hacimli barlar yayınlıyor; bunları gerçek saymak sahte düz getiriler
        yaratırdı.
    """
    per_symbol = {}
    for i, s in enumerate(symbols):
        df = load_symbol(s, interval)
        if len(df):
            per_symbol[s] = df
        if verbose and (i + 1) % 50 == 0:
            print(f"  loaded {i + 1}/{len(symbols)} symbols")
    start = min(df.index.min() for df in per_symbol.values())
    end = max(df.index.max() for df in per_symbol.values())
    grid = pd.date_range(start, end, freq=_INTERVAL[interval], name="time")

    fields = {
        f: pd.DataFrame({s: df[f].reindex(grid) for s, df in per_symbol.items()})
        for f in PANEL_FIELDS
    }
    traded = fields["quote_volume"].fillna(0.0) > 0
    for f in PANEL_FIELDS:
        fields[f] = fields[f].where(traded)
    return Panel(fields=fields, traded=traded)


def save_panel(panel: Panel, name: str) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for f, df in panel.fields.items():
        df.to_parquet(config.PROCESSED_DIR / f"{name}_{f}.parquet")


def load_panel(name: str) -> Panel:
    fields = {
        f: pd.read_parquet(config.PROCESSED_DIR / f"{name}_{f}.parquet")
        for f in PANEL_FIELDS
    }
    return Panel(fields=fields, traded=fields["close"].notna())


def quality_report(panel: Panel) -> pd.DataFrame:
    """Per-symbol coverage: first/last bar and how many bars are missing inside."""
    rows = []
    for s in panel.symbols:
        traded = panel.traded[s]
        if not traded.any():
            continue
        first, last = traded.idxmax(), traded[::-1].idxmax()
        inside = traded.loc[first:last]
        rows.append(
            {
                "symbol": s,
                "first": first,
                "last": last,
                "bars": int(inside.sum()),
                "missing_inside": int((~inside).sum()),
                "missing_pct": 100 * float((~inside).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("symbol").sort_values("first")

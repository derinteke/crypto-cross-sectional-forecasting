"""Build the dataset: daily bars -> point-in-time universe -> hourly bars.

EN: Two stages keep the download small. Daily bars for every USDT pair are
    tiny, and they are all I need to decide who is in the top 30 each month.
    Hourly bars, the expensive part, are then fetched only for coins that were
    ever in the universe, including the ones that were later delisted.
TR: İki aşama indirmeyi küçük tutuyor. Her USDT paritesinin günlük barları
    çok küçük ve her ay ilk 30'da kimin olduğuna karar vermek için tek ihtiyacım
    bunlar. Pahalı kısım olan saatlik barlar ise yalnızca evrene bir kez olsun
    girmiş coinler için indiriliyor; sonradan delist edilenler dahil.

Usage / Kullanım:
    python scripts/download_data.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data import build_panel, quality_report, save_panel  # noqa: E402
from src.download import candidate_symbols, download_symbols  # noqa: E402
from src.universe import build_universe, universe_frame  # noqa: E402


def main() -> None:
    t0 = time.time()
    print("[1/5] Listing USDT spot pairs in the archive ...")
    candidates = candidate_symbols()
    print(f"  {len(candidates)} candidate symbols after excluding stables/fiat/leveraged")

    print("[2/5] Downloading DAILY bars for all candidates ...")
    print(" ", download_symbols(candidates, "1d"))
    daily = build_panel(candidates, "1d")
    save_panel(daily, "daily")

    print("[3/5] Building the point-in-time universe ...")
    universe = build_universe(daily, config.FIRST_SAMPLE_DATE, config.DATA_END_MONTH)
    table = universe_frame(universe)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.REPORTS_DIR / "universe.csv", index=False)
    members = sorted(table["symbol"].unique())
    sizes = table.groupby("month").size()
    print(f"  {len(universe)} months, {len(members)} distinct symbols ever in the top "
          f"{config.UNIVERSE_SIZE}; smallest month has {sizes.min()} coins")

    print("[4/5] Downloading HOURLY bars for universe members (checksummed) ...")
    print(" ", download_symbols(members, "1h", verify_checksum=True))
    hourly = build_panel(members, "1h")
    save_panel(hourly, "hourly")

    print("[5/5] Data quality report ...")
    report = quality_report(hourly)
    report.to_csv(config.REPORTS_DIR / "data_quality.csv")
    end = pd.Timestamp(config.DATA_END_MONTH) + pd.offsets.MonthEnd(0)
    delisted = report[report["last"] < end - pd.Timedelta(days=7)]
    print(f"  hourly grid: {hourly.index.min()} -> {hourly.index.max()} "
          f"({len(hourly.index):,} hours x {len(hourly.symbols)} symbols)")
    print(f"  symbols that stopped trading before the end: {len(delisted)} "
          f"({', '.join(delisted.index[:12])}{' ...' if len(delisted) > 12 else ''})")
    print(f"  median missing bars inside a symbol's life: "
          f"{report['missing_pct'].median():.3f}%")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

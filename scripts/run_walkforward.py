"""Walk-forward runner: baselines first, then the models, then one scorecard.

EN: The order is the point. Baselines are scored and printed before a single
    network is trained, so the bar is on screen before any model tries to
    clear it. `--smoke` runs the whole pipeline on 10 coins and two folds in a
    couple of minutes, to prove the code works before the full run.
TR: Sıra meselenin ta kendisi. Baseline'lar tek bir ağ eğitilmeden önce
    puanlanıp yazdırılıyor; böylece çıta, herhangi bir model onu aşmaya
    çalışmadan önce ekranda duruyor. `--smoke`, bütün hattı 10 coin ve iki fold
    üzerinde birkaç dakikada koşturuyor; tam koşudan önce kodun çalıştığını
    kanıtlamak için.

Usage / Kullanım:
    python scripts/run_walkforward.py --smoke
    python scripts/run_walkforward.py
    python scripts/run_walkforward.py --models LightGBM GRU --tag gru_only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import backtest as B  # noqa: E402
from src import config  # noqa: E402
from src import evaluate as E  # noqa: E402
from src import metrics as M  # noqa: E402
from src import walkforward as W  # noqa: E402
from src.data import load_panel  # noqa: E402
from src.features import MARKET_SYMBOL, build_samples  # noqa: E402
from src.splits import walk_forward_folds  # noqa: E402
from src.train import get_device, set_seed  # noqa: E402

ALL_MODELS = (W.POINT_BASELINES + W.POINT_MODELS + W.QUANTILE_MODELS + W.CONTROL_MODELS)
BASELINE_SET = set(W.POINT_BASELINES + ["VolScaled-Q"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--cost-bps", type=float, default=config.COST_BPS)
    p.add_argument("--tag", default="")
    p.add_argument("--rescore", action="store_true",
                   help="Skip training: re-score the saved predictions of --tag.")
    return p.parse_args()


def load_universe() -> dict[pd.Timestamp, list[str]]:
    table = pd.read_csv(config.REPORTS_DIR / "universe.csv", parse_dates=["month"])
    return {m: list(g.sort_values("rank")["symbol"]) for m, g in table.groupby("month")}


def combine(a: W.WalkForwardResult, b: W.WalkForwardResult) -> W.WalkForwardResult:
    assert a.point.index.equals(b.point.index), "baseline and model rows differ"
    return W.WalkForwardResult(
        point=pd.concat([a.point, b.point], axis=1),
        quantiles={**a.quantiles, **b.quantiles},
        targets=a.targets,
        histories={**a.histories, **b.histories},
        fold_log=a.fold_log + b.fold_log,
    )


def main() -> None:
    args = parse_args()
    tag = args.tag or ("smoke" if args.smoke else "full")
    epochs = config.SMOKE_EPOCHS if args.smoke else args.epochs
    set_seed(config.SEED)
    t_start = time.time()
    if args.rescore:
        report(load_result(tag), args, tag, t_start)
        return

    print("=" * 78 + "\nDEVICE / CİHAZ\n" + "=" * 78)
    device = get_device()

    print("\n" + "=" * 78 + "\nDATA / VERİ\n" + "=" * 78)
    hourly = load_panel("hourly")
    universe = load_universe()
    start, first_test, end = config.FIRST_SAMPLE_DATE, config.FIRST_TEST_DATE, None
    if args.smoke:
        counts = pd.Series([s for syms in universe.values() for s in syms]).value_counts()
        keep = [MARKET_SYMBOL] + [s for s in counts.index if s != MARKET_SYMBOL]
        keep = keep[: config.SMOKE_SYMBOLS]
        hourly = hourly.restrict(symbols=keep,
                                 start=pd.Timestamp(config.SMOKE_START) - pd.Timedelta(days=45),
                                 end=pd.Timestamp(config.SMOKE_END) + pd.Timedelta(days=7))
        start, first_test, end = config.SMOKE_START, config.SMOKE_FIRST_TEST, config.SMOKE_END
    t0 = time.time()
    samples = build_samples(hourly, universe, start=start, end=end)
    frame = samples.frame
    per_day = frame.groupby(level="time").size()
    print(f"hourly panel : {len(hourly.index):,} hours x {len(hourly.symbols)} symbols")
    print(f"samples      : {len(frame):,} (day, coin) rows over {len(per_day):,} days "
          f"| coins per day median {per_day.median():.0f}, min {per_day.min()}")
    print(f"built in {time.time() - t0:.1f}s")

    folds = walk_forward_folds(start, first_test, frame.index.get_level_values("time").max())
    print("\n" + "\n".join(str(f) for f in folds))

    baselines = [m for m in args.models if m in BASELINE_SET]
    models = [m for m in args.models if m not in BASELINE_SET]

    print("\n" + "=" * 78 + "\nBASELINES / SAĞDUYU REFERANSLARI  (before any training)\n" + "=" * 78)
    res = W.run_walk_forward(samples, folds, baselines, device, epochs=epochs, verbose=False)
    table, _ = W.score_point_models(res, args.cost_bps)
    cols = ["ic_mean", "icir", "ic_tstat_nw", "sharpe_gross", "sharpe_net", "turnover"]
    print(table[cols].to_string(float_format=lambda v: f"{v:.3f}"))

    if models:
        print("\n" + "=" * 78 + "\nMODELS / MODELLER\n" + "=" * 78)
        res = combine(res, W.run_walk_forward(samples, folds, models, device, epochs=epochs))

    save_result(res, tag)
    # EN: the trial log only grows when something was actually trained.
    # TR: deneme kaydı yalnızca gerçekten bir şey eğitildiğinde büyüyor.
    table, _ = W.score_point_models(res, args.cost_bps)
    trials = table[["sharpe_net"]].assign(tag=tag, cost_bps=args.cost_bps)
    trials_path = config.REPORTS_DIR / "trials.csv"
    trials.to_csv(trials_path, mode="a", header=not trials_path.exists())
    report(res, args, tag, t_start)


def save_result(res: W.WalkForwardResult, tag: str) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    res.point.join(res.targets).to_parquet(config.PROCESSED_DIR / f"predictions_{tag}.parquet")
    np.savez(config.PROCESSED_DIR / f"quantiles_{tag}.npz", **res.quantiles)
    hist = {f"{m}|{fold}": h for (m, fold), h in res.histories.items()}
    (config.PROCESSED_DIR / f"histories_{tag}.json").write_text(json.dumps(hist))
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res.fold_log).to_csv(config.REPORTS_DIR / f"folds_{tag}.csv", index=False)


def load_result(tag: str) -> W.WalkForwardResult:
    df = pd.read_parquet(config.PROCESSED_DIR / f"predictions_{tag}.parquet")
    target_cols = [c for c in df.columns if c.startswith("y_day")] + ["y_demeaned"]
    with np.load(config.PROCESSED_DIR / f"quantiles_{tag}.npz") as z:
        quantiles = {k: z[k] for k in z.files}
    histories = {}
    hist_path = config.PROCESSED_DIR / f"histories_{tag}.json"
    if hist_path.exists():
        for key, h in json.loads(hist_path.read_text()).items():
            m, fold = key.split("|")
            histories[(m, int(fold))] = h
    return W.WalkForwardResult(point=df.drop(columns=target_cols), quantiles=quantiles,
                               targets=df[target_cols], histories=histories)


def report(res: W.WalkForwardResult, args, tag: str, t_start: float) -> None:
    """Score, print, save tables and draw every figure for one result."""
    print("\n" + "=" * 78 + "\nRESULTS / SONUÇLAR  (out of sample, stitched folds)\n" + "=" * 78)
    table, books = W.score_point_models(res, args.cost_bps)
    cols = ["ic_mean", "icir", "ic_tstat_nw", "sharpe_gross", "sharpe_net", "turnover"]
    show = cols + ["ann_return_net", "max_dd_net", "beta_to_market", "alpha_bps_per_day",
                   "alpha_tstat", "psr", "deflated_sharpe"]
    print(table[show].to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"(deflated Sharpe treats {table.attrs['n_trials']} candidate strategies as trials)")

    benchmark = B.run_backtest(res.point.iloc[:, 0], res.targets["y_day0"], "benchmark", 0.0)["gross"]
    print(f"\nEqual-weight universe (long only, no costs): Sharpe {M.sharpe(benchmark):.2f}, "
          f"max drawdown {M.max_drawdown(benchmark):.1%}")

    point_names = [c for c in res.point.columns]
    years = W.per_year(res, books, point_names)
    print("\nPer year / Yıllara göre:\n" + years.to_string(float_format=lambda v: f"{v:.3f}"))

    qtable = None
    if res.quantiles:
        qtable, calib = W.score_quantile_models(res)
        print("\nQuantile forecasts / Quantile tahminleri:\n"
              + qtable.to_string(float_format=lambda v: f"{v:.5f}"))

    # ---- persist ---------------------------------------------------------- #
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.REPORTS_DIR / f"results_{tag}.csv")
    years.to_csv(config.REPORTS_DIR / f"results_by_year_{tag}.csv")
    if qtable is not None:
        qtable.to_csv(config.REPORTS_DIR / f"results_quantile_{tag}.csv")

    # ---- figures ---------------------------------------------------------- #
    headline = [m for m in ["LightGBM", "GRU", "Transformer", "Ridge", "Momentum7d",
                            "Reversal24h", "Random"] if m in res.point.columns]
    E.plot_equity_curves(books, headline, benchmark, filename=f"equity_{tag}.png")
    E.plot_equity_curves(books, headline, benchmark, column="gross",
                         title="Long-short book before costs", filename=f"equity_gross_{tag}.png")
    ics = {m: M.daily_rank_ic(res.point[m], res.targets["y_day0"]) for m in headline if m != "Random"}
    E.plot_rolling_ic(ics, filename=f"rolling_ic_{tag}.png")
    E.plot_quintiles({m: B.quintile_returns(res.point[m], res.targets["y_day0"]) for m in headline},
                     filename=f"quintiles_{tag}.png")
    E.plot_ic_decay(table, headline, filename=f"ic_decay_{tag}.png")
    smoothed = [f"{m} (smoothed)" for m in W.POINT_MODELS if f"{m} (smoothed)" in books]
    E.plot_equity_curves(books, [m for m in W.POINT_MODELS if m in books] + smoothed, benchmark,
                         title="Raw vs smoothed signal, net of costs",
                         filename=f"equity_smoothed_{tag}.png")
    curves = {m: B.cost_sensitivity(res.point[m], res.targets["y_day0"])
              for m in headline if m != "Random"}
    for m in [m for m in W.POINT_MODELS if m in res.point.columns]:
        curves[f"{m} (smoothed)"] = B.cost_sensitivity(B.smooth_signal(res.point[m]),
                                                       res.targets["y_day0"])
    E.plot_cost_sensitivity(curves, filename=f"cost_sensitivity_{tag}.png")
    if res.quantiles:
        E.plot_calibration(calib, filename=f"calibration_{tag}.png")
    for name in {m for (m, _) in res.histories}:
        E.plot_training_curves(res.histories, name, filename=f"training_{name}_{tag}.png")

    print(f"\nTables  -> {config.REPORTS_DIR}")
    print(f"Figures -> {config.FIGURES_DIR}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()

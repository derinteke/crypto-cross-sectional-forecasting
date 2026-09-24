"""Run every model through every fold, then score the stitched test predictions.

EN: One function trains and predicts, one function scores. Keeping them apart
    means the scoring code never has access to a model, only to predictions,
    so nothing in the evaluation can accidentally refit on test data.
    The test predictions of consecutive folds are stitched into a single
    out-of-sample history (2022 -> 2026) and every metric is computed on that.
TR: Bir fonksiyon eğitiyor ve tahmin ediyor, bir fonksiyon puanlıyor. Onları
    ayrı tutmak, puanlama kodunun asla bir modele değil yalnızca tahminlere
    erişmesi demek; dolayısıyla değerlendirmedeki hiçbir şey kazara test verisi
    üzerinde yeniden fit edemiyor. Ardışık fold'ların test tahminleri tek bir
    örneklem dışı geçmişe (2022 -> 2026) dikiliyor ve her metrik bunun üzerinden
    hesaplanıyor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import backtest as B
from src import baselines as BL
from src import config
from src import metrics as M
from src.features import ALL_FEATURES, SEQ_CHANNELS, Samples
from src.models import build_seq_model, count_parameters, fit_lgbm
from src.splits import Fold
from src.train import FoldScaler, FoldTensors, predict_network, set_seed, train_network

POINT_BASELINES = ["Random", "Momentum7d", "Reversal24h", "Ridge"]
POINT_MODELS = ["LightGBM", "GRU", "Transformer"]
QUANTILE_MODELS = ["VolScaled-Q", "LightGBM-Q", "GRU-Q", "Transformer-Q"]
CONTROL_MODELS = ["LightGBM-shuffled"]
NULL_CONTROLS = {"Random", "LightGBM-shuffled"}


@dataclass
class WalkForwardResult:
    point: pd.DataFrame                          # (time, symbol) x model
    quantiles: dict[str, np.ndarray]             # model -> (n, n_q), aligned to point
    targets: pd.DataFrame                        # y_day0..y_dayK, y_demeaned
    histories: dict = field(default_factory=dict)
    fold_log: list = field(default_factory=list)


def _shuffle_within_day(y: pd.Series, seed: int) -> np.ndarray:
    """Permute targets across coins inside each day: kills any real signal."""
    rng = np.random.default_rng(seed)
    out = y.copy()
    for _, idx in y.groupby(level="time").indices.items():
        out.iloc[idx] = y.iloc[idx].to_numpy()[rng.permutation(len(idx))]
    return out.to_numpy()


def run_walk_forward(
    samples: Samples,
    folds: list[Fold],
    models: list[str],
    device,
    epochs: int = config.EPOCHS,
    lookback: int = config.SEQ_LOOKBACK_HOURS,
    verbose: bool = True,
) -> WalkForwardResult:
    frame = samples.frame
    times = frame.index.get_level_values("time")
    point_parts, quant_parts, histories, log = [], {}, {}, []

    for fold in folds:
        t0 = time.time()
        tr = frame[(times >= fold.train_start) & (times <= fold.train_end)]
        va = frame[(times >= fold.val_start) & (times <= fold.val_end)]
        te = frame[(times >= fold.test_start) & (times <= fold.test_end)]
        if len(te) == 0 or len(tr) == 0 or len(va) == 0:
            continue
        if verbose:
            print(f"\n{fold}  (rows: train {len(tr):,} / val {len(va):,} / test {len(te):,})")

        preds = pd.DataFrame(index=te.index)
        entry = {"fold": fold.number, "test_start": fold.test_start, "n_test": len(te)}

        # ---- baselines ---------------------------------------------------- #
        if "Random" in models:
            preds["Random"] = BL.random_signal(te, seed=config.SEED + fold.number)
        if "Momentum7d" in models:
            preds["Momentum7d"] = BL.momentum_signal(te)
        if "Reversal24h" in models:
            preds["Reversal24h"] = BL.reversal_signal(te)
        if "Ridge" in models:
            ridge, mu, sd, alpha = BL.fit_ridge(tr, va)
            preds["Ridge"] = BL.predict_ridge(ridge, mu, sd, te)
            entry["ridge_alpha"] = alpha
        if "VolScaled-Q" in models:
            quant_parts.setdefault("VolScaled-Q", []).append(BL.vol_scaled_quantiles(tr, te))

        # ---- LightGBM ----------------------------------------------------- #
        if "LightGBM" in models:
            booster = fit_lgbm(tr[ALL_FEATURES], tr["y_rankgauss"].to_numpy(),
                               va[ALL_FEATURES], va["y_rankgauss"].to_numpy())
            preds["LightGBM"] = booster.predict(te[ALL_FEATURES],
                                                num_iteration=booster.best_iteration)
            entry["lgbm_rounds"] = booster.best_iteration
        if "LightGBM-shuffled" in models:
            booster = fit_lgbm(tr[ALL_FEATURES],
                               _shuffle_within_day(tr["y_rankgauss"], config.SEED + fold.number),
                               va[ALL_FEATURES],
                               _shuffle_within_day(va["y_rankgauss"], config.SEED + 100 + fold.number))
            preds["LightGBM-shuffled"] = booster.predict(te[ALL_FEATURES],
                                                         num_iteration=booster.best_iteration)
        if "LightGBM-Q" in models:
            cols = []
            for q in config.QUANTILES:
                booster = fit_lgbm(tr[ALL_FEATURES], tr["y_demeaned"].to_numpy(),
                                   va[ALL_FEATURES], va["y_demeaned"].to_numpy(), quantile=q)
                cols.append(booster.predict(te[ALL_FEATURES], num_iteration=booster.best_iteration))
            quant_parts.setdefault("LightGBM-Q", []).append(np.sort(np.column_stack(cols), axis=1))

        # ---- sequence networks ---------------------------------------------- #
        nets = [m for m in models if m.split("-")[0] in ("GRU", "Transformer")]
        if nets:
            scaler = FoldScaler.fit(samples, tr, lookback)
        for name in nets:
            quantile = name.endswith("-Q")
            target = "y_demeaned" if quantile else "y_rankgauss"
            set_seed(config.SEED)
            model = build_seq_model(name.split("-")[0], n_channels=len(SEQ_CHANNELS),
                                    n_static=len(ALL_FEATURES),
                                    n_outputs=len(config.QUANTILES) if quantile else 1)
            if verbose:
                print(f"  [{name}] {count_parameters(model):,} parameters")
            data = {part: FoldTensors(samples, rows, scaler, target, lookback, device)
                    for part, rows in (("train", tr), ("val", va), ("test", te))}
            model, hist = train_network(model, data["train"], data["val"], device,
                                        quantile=quantile, epochs=epochs, verbose=verbose)
            histories[(name, fold.number)] = hist.as_dict()
            out = predict_network(model, data["test"], quantile=quantile)
            if quantile:
                quant_parts.setdefault(name, []).append(out)
            else:
                preds[name] = out

        entry["seconds"] = round(time.time() - t0, 1)
        log.append(entry)
        point_parts.append(preds)
        if verbose:
            ic = {c: M.daily_rank_ic(preds[c], te["y_day0"]).mean() for c in preds.columns}
            print("  test IC: " + "  ".join(f"{k} {v:+.4f}" for k, v in ic.items()))

    point = pd.concat(point_parts).sort_index()
    order = point.index
    target_cols = [c for c in frame.columns if c.startswith("y_day")] + ["y_demeaned"]
    # EN: quantile blocks were appended fold by fold in the same row order as
    #     `point_parts`, so concatenating them keeps them aligned.
    # TR: quantile blokları fold fold, `point_parts` ile aynı satır sırasında
    #     eklendi; dolayısıyla birleştirmek hizayı koruyor.
    unsorted_index = pd.concat(point_parts).index
    reorder = pd.Series(np.arange(len(unsorted_index)), index=unsorted_index).loc[order].to_numpy()
    quantiles = {k: np.concatenate(v)[reorder] for k, v in quant_parts.items()}
    return WalkForwardResult(point=point, quantiles=quantiles,
                             targets=frame.loc[order, target_cols],
                             histories=histories, fold_log=log)


# --------------------------------------------------------------------------- #
# Scoring / Puanlama
# --------------------------------------------------------------------------- #
def quantile_score(q: np.ndarray) -> np.ndarray:
    """Median over interval width: an expected move per unit of uncertainty."""
    qs = list(np.round(config.QUANTILES, 6))
    lo, mid, hi = qs.index(0.1), qs.index(0.5), qs.index(0.9)
    return q[:, mid] / np.maximum(q[:, hi] - q[:, lo], 1e-8)


def score_point_models(res: WalkForwardResult, cost_bps: float = config.COST_BPS) -> tuple[pd.DataFrame, dict]:
    """One row per model: signal metrics, long-short book gross/net, DSR."""
    y0 = res.targets["y_day0"]
    books, rows = {}, {}
    signals = {c: res.point[c] for c in res.point.columns}
    for name, q in res.quantiles.items():
        if name != "VolScaled-Q":
            signals[f"{name} (q50/width)"] = pd.Series(quantile_score(q), index=res.point.index)
    for name in [m for m in POINT_MODELS if m in res.point.columns]:
        signals[f"{name} (smoothed)"] = B.smooth_signal(res.point[name])

    for name, pred in signals.items():
        ic = M.daily_rank_ic(pred, y0)
        row = M.ic_summary(ic)
        for k in range(1, config.IC_DECAY_DAYS):
            row[f"ic_day{k}"] = M.daily_rank_ic(pred, res.targets[f"y_day{k}"]).mean()
        bt = B.run_backtest(pred, y0, "long_short", cost_bps)
        books[name] = bt
        g, n = B.summarize(bt, "gross"), B.summarize(bt, "net")
        row.update(sharpe_gross=g["sharpe"], sharpe_net=n["sharpe"],
                   ann_return_net=n["ann_return"], max_dd_net=n["max_drawdown"],
                   turnover=n["avg_turnover"])
        rows[name] = row

    # EN: every candidate strategy I evaluated counts as a trial. The two null
    #     controls (random scores, shuffled-target model) are not candidates I
    #     would ever pick; their cost-driven Sharpes of -3 to -4 would inflate
    #     the spread of trial Sharpes and make the test meaninglessly harsh.
    #     PSR is the same test with a single trial, for reference.
    # TR: değerlendirdiğim her aday strateji bir deneme sayılıyor. İki sıfır
    #     kontrolü (rastgele skorlar, karıştırılmış hedefli model) asla
    #     seçeceğim adaylar değil; maliyetten gelen -3 ile -4 arası Sharpe'ları,
    #     deneme Sharpe'larının yayılımını şişirip testi anlamsızca sertleştirirdi.
    #     PSR, referans için aynı testin tek denemeli hâli.
    market = B.run_backtest(res.point.iloc[:, 0], y0, "benchmark", 0.0)["gross"]
    candidates = [n for n in books if n not in NULL_CONTROLS]
    trial_sr = [books[n]["net"].mean() / books[n]["net"].std() for n in candidates]
    for name, bt in books.items():
        rows[name]["psr"] = M.deflated_sharpe_ratio(bt["net"], [])["deflated_sharpe"]
        rows[name]["deflated_sharpe"] = M.deflated_sharpe_ratio(bt["net"], trial_sr)["deflated_sharpe"]
        rows[name].update(M.market_regression(bt["gross"], market))
    table = pd.DataFrame(rows).T
    table.index.name = "model"
    table.attrs["n_trials"] = len(candidates)
    return table.sort_values("ic_mean", ascending=False), books


def score_quantile_models(res: WalkForwardResult) -> tuple[pd.DataFrame, dict]:
    y = res.targets["y_demeaned"].to_numpy()
    rows, calib = {}, {}
    for name, q in res.quantiles.items():
        pin = M.pinball_loss(y, q)
        rows[name] = {"pinball_mean": float(pin.mean()),
                      "coverage_80": M.interval_coverage(y, q),
                      "median_width_80": float(np.median(q[:, -1] - q[:, 0]))}
        calib[name] = M.calibration(y, q)
    return pd.DataFrame(rows).T.sort_values("pinball_mean"), calib


def per_year(res: WalkForwardResult, books: dict, models: list[str]) -> pd.DataFrame:
    """Rank IC and net Sharpe by calendar year: does the edge survive regimes?"""
    y0 = res.targets["y_day0"]
    out = []
    for name in models:
        ic = M.daily_rank_ic(res.point[name], y0)
        net = books[name]["net"]
        for year in sorted(set(ic.index.year)):
            out.append({"model": name, "year": year,
                        "ic_mean": ic[ic.index.year == year].mean(),
                        "sharpe_net": M.sharpe(net[net.index.year == year])})
    return pd.DataFrame(out).pivot(index="model", columns="year")

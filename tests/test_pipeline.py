"""Tests that turn my no-leakage claims into something checkable.

EN: They run on a small synthetic market (random-walk prices for a handful of
    coins, one of which is delisted mid-way, with a hole of missing bars), so
    they need no download and finish in seconds. Synthetic data is also the
    right tool here: I can plant exactly the traps I want to prove I avoid.
TR: Küçük, sentetik bir piyasa üzerinde koşuyorlar (bir avuç coin için rastgele
    yürüyüş fiyatları; biri yolun ortasında delist ediliyor ve bir eksik bar
    deliği var); dolayısıyla indirme gerektirmiyor ve saniyeler içinde
    bitiyorlar. Sentetik veri burada doğru araç: kaçındığımı kanıtlamak
    istediğim tuzakları tam olarak istediğim yere yerleştirebiliyorum.

Run / Koşum:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import backtest as B  # noqa: E402
from src import config  # noqa: E402
from src import metrics as M  # noqa: E402
from src.data import Panel, _to_timestamp  # noqa: E402
from src.download import is_candidate  # noqa: E402
from src.features import (  # noqa: E402
    ALL_FEATURES, build_samples, hourly_features, market_features, sequence_channels,
)
from src.splits import walk_forward_folds  # noqa: E402
from src.train import FoldScaler  # noqa: E402
from src.universe import build_universe, membership, select_universe  # noqa: E402
from src.walkforward import _shuffle_within_day  # noqa: E402

SYMBOLS = ["BTCUSDT", "ETHUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT", "DEADUSDT"]
START = pd.Timestamp("2021-01-01 01:00")
N_HOURS = 24 * 150
DELIST_AT = START + pd.Timedelta(days=100)
HOLE = slice(START + pd.Timedelta(days=80), START + pd.Timedelta(days=80, hours=5))


def _make_panel(freq: str, n: int, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    index = pd.date_range(START if freq == "h" else START.normalize() + pd.Timedelta(days=1),
                          periods=n, freq=freq, name="time")
    scale = 0.01 if freq == "h" else 0.05
    logp = np.cumsum(rng.normal(0, scale, (n, len(SYMBOLS))), axis=0) + 3.0
    close = pd.DataFrame(np.exp(logp), index=index, columns=SYMBOLS)
    spread = np.exp(np.abs(rng.normal(0, scale / 2, close.shape)))
    volume = pd.DataFrame(rng.uniform(1e5, 1e6, close.shape), index=index, columns=SYMBOLS)
    volume["BTCUSDT"] *= 10  # BTC is always the most liquid
    fields = {
        "open": close.shift(1).fillna(close),
        "high": close * spread,
        "low": close / spread,
        "close": close,
        "quote_volume": volume,
        "trades": volume / 100,
    }
    traded = pd.DataFrame(True, index=index, columns=SYMBOLS)
    traded.loc[index >= DELIST_AT, "DEADUSDT"] = False       # delisted coin
    traded.loc[HOLE, "AAAUSDT"] = False                       # missing bars
    for f in fields:
        fields[f] = fields[f].where(traded)
    return Panel(fields=fields, traded=traded)


@pytest.fixture(scope="module")
def hourly() -> Panel:
    return _make_panel("h", N_HOURS)


@pytest.fixture(scope="module")
def daily() -> Panel:
    return _make_panel("D", 150, seed=1)


@pytest.fixture(scope="module")
def universe() -> dict:
    months = pd.date_range("2021-02-01", "2021-06-01", freq="MS")
    return {m: list(SYMBOLS) for m in months}


@pytest.fixture(scope="module")
def samples(hourly, universe):
    return build_samples(hourly, universe, start="2021-02-01")


# --------------------------------------------------------------------------- #
# Look-ahead / Geleceğe bakma
# --------------------------------------------------------------------------- #
def test_features_do_not_change_when_the_future_is_deleted(hourly):
    """The strongest look-ahead test: cut the data at t, recompute, compare.

    EN: If any feature at time t used a bar after t, deleting those bars would
        change its value. Rolling windows, forward fills, betas: all covered.
    TR: t anındaki herhangi bir özellik t sonrasındaki bir barı kullansaydı, o
        barları silmek değerini değiştirirdi. Kayan pencereler, ileri
        doldurmalar, beta'lar: hepsi kapsanıyor.
    """
    full = hourly_features(hourly)
    full_mkt = market_features(hourly)
    full_seq = sequence_channels(hourly)
    for cut in (START + pd.Timedelta(days=45), START + pd.Timedelta(days=81)):
        cut_panel = hourly.restrict(end=cut)
        cut_feats = hourly_features(cut_panel)
        for name, df in cut_feats.items():
            pd.testing.assert_series_equal(df.loc[cut], full[name].loc[cut], check_names=False,
                                           obj=name)
        pd.testing.assert_series_equal(market_features(cut_panel).loc[cut], full_mkt.loc[cut],
                                       check_names=False)
        pos = hourly.index.get_loc(cut)
        np.testing.assert_array_equal(sequence_channels(cut_panel)[-1], full_seq[pos])


def test_target_is_exactly_the_next_24h_after_a_one_hour_lag(hourly, samples):
    close = hourly.close.ffill()
    for (t, sym) in samples.frame.index[:: max(1, len(samples.frame) // 25)]:
        entry = close.loc[t + pd.Timedelta(hours=config.EXECUTION_LAG_HOURS), sym]
        exit_ = close.loc[t + pd.Timedelta(hours=config.EXECUTION_LAG_HOURS
                                           + config.HORIZON_HOURS), sym]
        assert samples.frame.loc[(t, sym), "y_day0"] == pytest.approx(np.log(exit_ / entry))


def test_decisions_are_only_taken_at_the_decision_hour(samples):
    times = samples.frame.index.get_level_values("time")
    assert (times.hour == config.DECISION_HOUR_UTC).all()


def test_no_sample_reads_a_missing_bar(hourly, samples):
    """Windows over the planted hole of missing bars are dropped."""
    lookback = pd.Timedelta(hours=config.SEQ_LOOKBACK_HOURS)
    aaa = samples.frame.xs("AAAUSDT", level="symbol").index
    hole_start, hole_end = HOLE.start, HOLE.stop
    touching = aaa[(aaa - lookback < hole_end) & (aaa >= hole_start)]
    assert len(touching) == 0


def test_a_delisted_coin_is_never_bought_after_it_stops_trading(samples):
    dead = samples.frame.xs("DEADUSDT", level="symbol").index
    assert len(dead) > 0, "the coin must be in the sample before its delisting"
    assert dead.max() + pd.Timedelta(hours=config.EXECUTION_LAG_HOURS) < DELIST_AT


def test_rank_features_are_centred_within_each_day(samples):
    ranks = samples.frame["rank_ret_24h"]
    assert ranks.between(-0.5, 0.5).all()
    assert samples.frame.groupby(level="time")["y_demeaned"].mean().abs().max() < 1e-12


# --------------------------------------------------------------------------- #
# Universe / Evren
# --------------------------------------------------------------------------- #
def test_universe_ignores_everything_after_the_month_start(daily):
    """Pump a coin's volume AFTER month start: the selection must not move."""
    month = pd.Timestamp("2021-04-01")
    before = select_universe(daily, month, size=3, min_history_days=30)

    tampered = {k: v.copy() for k, v in daily.fields.items()}
    after = tampered["quote_volume"].index > month
    tampered["quote_volume"].loc[after, "CCCUSDT"] = 1e15
    again = select_universe(Panel(tampered, daily.traded), month, size=3, min_history_days=30)
    assert before == again


def test_universe_needs_history(daily):
    """A coin younger than MIN_HISTORY_DAYS cannot enter."""
    month = pd.Timestamp("2021-02-01")  # the data only starts in January
    assert select_universe(daily, month, size=5, min_history_days=60) == []


def test_delisted_coin_is_in_the_universe_until_it_dies(daily):
    uni = build_universe(daily, "2021-03-01", "2021-06-01", size=6, min_history_days=30)
    assert "DEADUSDT" in uni[pd.Timestamp("2021-04-01")]
    assert "DEADUSDT" not in uni[pd.Timestamp("2021-05-01")]


def test_membership_uses_the_list_of_the_month_it_is_in():
    uni = {pd.Timestamp("2021-01-01"): ["A"], pd.Timestamp("2021-02-01"): ["B"]}
    idx = pd.DatetimeIndex(["2021-01-31 23:00", "2021-02-01 00:00"])
    mem = membership(uni, idx, ["A", "B"])
    assert mem.loc[idx[0], "A"] and not mem.loc[idx[0], "B"]
    assert mem.loc[idx[1], "B"] and not mem.loc[idx[1], "A"]


# --------------------------------------------------------------------------- #
# Walk-forward / İleriye doğru yürüyen doğrulama
# --------------------------------------------------------------------------- #
def test_folds_are_ordered_and_embargoed():
    folds = walk_forward_folds("2019-03-01", "2022-01-01", "2026-08-30")
    label_span = pd.Timedelta(hours=config.EXECUTION_LAG_HOURS + config.HORIZON_HOURS)
    for f in folds:
        assert f.train_start < f.train_end < f.val_start <= f.val_end < f.test_start <= f.test_end
        # EN: the last label of a slice must end before the next slice starts.
        # TR: bir dilimin son etiketi bir sonraki dilim başlamadan bitmeli.
        assert f.train_end + label_span < f.val_start
        assert f.val_end + label_span < f.test_start
    for a, b in zip(folds, folds[1:]):
        assert a.test_end < b.test_start
        assert b.train_end > a.train_end  # expanding window


def test_scaler_statistics_come_from_train_only(samples):
    frame = samples.frame
    times = frame.index.get_level_values("time")
    split = times.unique()[len(times.unique()) // 2]
    train = frame[times < split]
    scaler = FoldScaler.fit(samples, train, config.SEQ_LOOKBACK_HOURS)
    np.testing.assert_allclose(scaler.static_mean, train[ALL_FEATURES].mean().to_numpy(), rtol=1e-5)
    assert scaler.static_mean[0] != pytest.approx(frame[ALL_FEATURES[0]].mean(), rel=1e-6)


def test_shuffling_keeps_each_day_but_breaks_the_link(samples):
    y = samples.frame["y_rankgauss"]
    shuffled = pd.Series(_shuffle_within_day(y, 0), index=y.index)
    for t in y.index.get_level_values("time").unique()[:5]:
        assert sorted(y.loc[t]) == pytest.approx(sorted(shuffled.loc[t]))
    assert not np.allclose(y.to_numpy(), shuffled.to_numpy())


# --------------------------------------------------------------------------- #
# Backtest / Geriye dönük test
# --------------------------------------------------------------------------- #
def _toy():
    times = pd.to_datetime(["2022-01-01", "2022-01-02"])
    syms = ["A", "B", "C", "D", "E"]
    idx = pd.MultiIndex.from_product([times, syms], names=["time", "symbol"])
    pred = pd.Series([5, 4, 3, 2, 1, 1, 2, 3, 4, 5], index=idx, dtype=float)
    simple = pd.Series([0.10, 0.0, 0.0, 0.0, -0.05, 0.02, 0.0, 0.0, 0.0, 0.03], index=idx)
    return pred, np.log1p(simple)


def test_backtest_matches_a_hand_calculation():
    pred, logret = _toy()
    bt = B.run_backtest(pred, logret, "long_short", cost_bps=10.0)
    # day 1: long A (+10%), short E (-5%) -> +0.15 ; turnover 2 (enter both legs)
    # day 2: long E (+3%), short A (+2%) -> +0.01 ; turnover 4 (flip both legs)
    assert bt["gross"].tolist() == pytest.approx([0.15, 0.01])
    assert bt["turnover"].tolist() == pytest.approx([2.0, 4.0])
    assert bt["cost"].tolist() == pytest.approx([0.002, 0.004])
    assert bt["net"].tolist() == pytest.approx([0.148, 0.006])


def test_long_short_book_is_dollar_neutral():
    pred, _ = _toy()
    w = B.target_weights(pred, "long_short")
    assert w.sum(axis=1).abs().max() < 1e-12
    assert w.abs().sum(axis=1).tolist() == pytest.approx([2.0, 2.0])


def test_benchmark_is_equal_weight():
    pred, _ = _toy()
    w = B.target_weights(pred, "benchmark")
    assert w.to_numpy() == pytest.approx(np.full((2, 5), 0.2))


# --------------------------------------------------------------------------- #
# Metrics / Metrikler
# --------------------------------------------------------------------------- #
def test_rank_ic_is_one_for_a_perfect_ranking():
    pred, logret = _toy()
    ic = M.daily_rank_ic(logret * 3 + 1, logret)
    assert ic.tolist() == pytest.approx([1.0, 1.0])


def test_newey_west_matches_statsmodels():
    import statsmodels.api as sm

    rng = np.random.default_rng(0)
    x = np.convolve(rng.normal(0.05, 1, 600), np.ones(3) / 3, mode="valid")
    ols = sm.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    assert M.newey_west_tstat(x, lags=5) == pytest.approx(ols.tvalues[0], rel=1e-6)


def test_deflated_sharpe_punishes_more_trials():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 1000))
    few = M.deflated_sharpe_ratio(r, [0.1, 0.05, 0.0])["deflated_sharpe"]
    many = M.deflated_sharpe_ratio(r, list(rng.normal(0, 0.05, 200)))["deflated_sharpe"]
    assert 0 <= many < few <= 1


def test_pinball_and_calibration_on_known_quantiles():
    from scipy.stats import norm

    rng = np.random.default_rng(2)
    y = rng.standard_normal(200_000)
    q = np.tile(norm.ppf(config.QUANTILES), (len(y), 1))
    calib = M.calibration(y, q)
    assert np.allclose(calib["observed"], calib["nominal"], atol=0.005)
    assert M.interval_coverage(y, q) == pytest.approx(0.8, abs=0.005)
    worse = M.pinball_loss(y, q * 2).mean()
    assert M.pinball_loss(y, q).mean() < worse


# --------------------------------------------------------------------------- #
# Raw data handling / Ham veri işleme
# --------------------------------------------------------------------------- #
def test_candidate_filter():
    bases = {"BTC", "ETH", "JUP", "USDC"}
    assert is_candidate("BTCUSDT", bases)
    assert is_candidate("JUPUSDT", bases)          # ends in UP, is not leveraged
    assert not is_candidate("BTCUPUSDT", bases)    # leveraged token
    assert not is_candidate("ETHDOWNUSDT", bases)
    assert not is_candidate("USDCUSDT", bases)     # stablecoin
    assert not is_candidate("NVDABUSDT", bases)    # tokenized equity
    assert not is_candidate("BTCEUR", bases)       # wrong quote


def test_millisecond_and_microsecond_stamps_agree():
    ms = pd.Series([1_735_689_600_000])            # 2025-01-01 in ms
    us = pd.Series([1_735_689_600_000_000])        # same instant in us
    assert _to_timestamp(ms)[0] == _to_timestamp(us)[0] == pd.Timestamp("2025-01-01")

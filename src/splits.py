"""Walk-forward folds with purging and an embargo.

EN: A single train/test split answers "how did this model do in one period?".
    Walk-forward answers the question a trader actually faces: "if I had
    retrained every six months with only the data available then, what would
    have happened?". Each fold trains on everything before it (expanding
    window), validates on the three months just before the test, and leaves a
    gap of `EMBARGO_DAYS` between every pair of slices. The gap matters because
    a sample at day t has a label that ends at t + 25h: without it, the last
    training labels would overlap the first validation day.
TR: Tek bir train/test bölmesi "bu model tek bir dönemde nasıl yaptı?"
    sorusuna cevap veriyor. Walk-forward ise bir trader'ın gerçekten karşılaştığı
    soruya cevap veriyor: "o gün elimde olan veriyle her altı ayda bir yeniden
    eğitseydim ne olurdu?". Her fold kendinden önceki her şeyle eğitiliyor
    (genişleyen pencere), testten hemen önceki üç ayla doğrulanıyor ve her iki
    dilim arasında `EMBARGO_DAYS` günlük bir boşluk bırakılıyor. Boşluk önemli,
    çünkü t günündeki bir örneğin etiketi t + 25 saatte bitiyor: boşluk olmasa
    son eğitim etiketleri ilk doğrulama günüyle örtüşürdü.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config


@dataclass(frozen=True)
class Fold:
    number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp   # inclusive
    val_start: pd.Timestamp
    val_end: pd.Timestamp     # inclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp    # inclusive

    def slice_of(self, times: pd.Index, part: str) -> pd.Index:
        start, end = getattr(self, f"{part}_start"), getattr(self, f"{part}_end")
        return times[(times >= start) & (times <= end)]

    def __str__(self) -> str:
        f = lambda t: t.strftime("%Y-%m-%d")  # noqa: E731
        return (
            f"fold {self.number:>2}: train {f(self.train_start)}..{f(self.train_end)} | "
            f"val {f(self.val_start)}..{f(self.val_end)} | "
            f"test {f(self.test_start)}..{f(self.test_end)}"
        )


def walk_forward_folds(
    first_sample: str,
    first_test: str,
    last_date: str | pd.Timestamp,
    retrain_months: int = config.RETRAIN_MONTHS,
    val_months: int = config.VAL_MONTHS,
    embargo_days: int = config.EMBARGO_DAYS,
) -> list[Fold]:
    """Expanding-window folds; the last test slice is cut at `last_date`."""
    embargo = pd.Timedelta(days=embargo_days)
    one_day = pd.Timedelta(days=1)
    start = pd.Timestamp(first_sample)
    last = pd.Timestamp(last_date)

    folds = []
    test_start = pd.Timestamp(first_test)
    while test_start <= last:
        test_end = min(test_start + pd.DateOffset(months=retrain_months) - one_day, last)
        val_end = test_start - embargo - one_day
        val_start = test_start - embargo - pd.DateOffset(months=val_months)
        train_end = val_start - embargo - one_day
        folds.append(
            Fold(len(folds) + 1, start, train_end, val_start, val_end, test_start, test_end)
        )
        test_start = test_start + pd.DateOffset(months=retrain_months)
    return folds

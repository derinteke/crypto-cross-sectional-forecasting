"""Device check, fold-local normalisation and the training loop for the networks.

EN: The loop is the one from my Jena Climate project (early stopping on
    validation, ReduceLROnPlateau, gradient clipping, best-epoch restore), with
    two changes: the loss is pluggable (MSE for a point forecast, pinball for
    quantiles), and batches are gathered with one vectorised index instead of a
    DataLoader, because slicing 512 windows out of an in-memory array is faster
    than 512 Python calls.
TR: Döngü Jena Climate projemdekinin aynısı (doğrulamada erken durdurma,
    ReduceLROnPlateau, gradyan kırpma, en iyi epoch'u geri yükleme); iki
    değişiklikle: kayıp fonksiyonu takılıp çıkarılabiliyor (nokta tahmin için
    MSE, quantile'lar için pinball) ve batch'ler bir DataLoader yerine tek bir
    vektörel indeksle toplanıyor; çünkü bellekteki bir diziden 512 pencere
    kesmek, 512 Python çağrısından daha hızlı.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src import config
from src.features import ALL_FEATURES, Samples


# --------------------------------------------------------------------------- #
# Reproducibility and device / Tekrarlanabilirlik ve cihaz
# --------------------------------------------------------------------------- #
def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(verbose: bool = True) -> torch.device:
    """Pick CUDA when available and print proof of what is actually used.

    EN: GTX 1050 Ti (Pascal, sm_61) needs the cu126 PyTorch build; cu128 wheels
        no longer compile for Pascal. Printing the arch list proves it matches.
    TR: GTX 1050 Ti (Pascal, sm_61) cu126 PyTorch derlemesini istiyor; cu128
        wheel'leri artık Pascal için derlenmiyor. Mimari listesini yazdırmak
        eşleştiğini kanıtlıyor.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"torch             : {torch.__version__}")
        print(f"device            : {device}")
        if device.type == "cuda":
            major, minor = torch.cuda.get_device_capability(0)
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"gpu               : {torch.cuda.get_device_name(0)}")
            print(f"compute capability: sm_{major}{minor}")
            print(f"vram              : {total_gb:.1f} GB")
            print(f"built for archs   : {torch.cuda.get_arch_list()}")
        else:
            print("WARNING: running on CPU - training will be noticeably slower.")
    return device


# --------------------------------------------------------------------------- #
# Fold-local tensors / Fold'a özel tensörler
# --------------------------------------------------------------------------- #
@dataclass
class FoldScaler:
    """Mean/std fitted on one fold's TRAIN rows only, applied to every slice."""

    seq_mean: np.ndarray
    seq_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    y_scale: float  # std of the demeaned return, for the quantile head

    @classmethod
    def fit(cls, samples: Samples, train: pd.DataFrame, lookback: int) -> "FoldScaler":
        # EN: sequence statistics come from the hours that train windows can
        #     read, for the coins that appear in train. Nothing after train_end.
        # TR: dizi istatistikleri, train pencerelerinin okuyabildiği saatlerden
        #     ve train'de görünen coinlerden geliyor. train_end sonrası hiçbir şey.
        h0 = max(int(train["hour_pos"].min()) - lookback + 1, 0)
        h1 = int(train["hour_pos"].max()) + 1
        syms = np.unique(train["sym_pos"].to_numpy())
        block = samples.seq[h0:h1][:, syms, :].reshape(-1, samples.seq.shape[-1])
        block = block[np.abs(block).sum(axis=1) > 0]  # skip untraded padding
        static = train[ALL_FEATURES].to_numpy(dtype=np.float64)
        return cls(
            seq_mean=block.mean(axis=0).astype(np.float32),
            seq_std=np.maximum(block.std(axis=0), 1e-8).astype(np.float32),
            static_mean=static.mean(axis=0).astype(np.float32),
            static_std=np.maximum(static.std(axis=0), 1e-8).astype(np.float32),
            y_scale=float(train["y_demeaned"].std()),
        )


class FoldTensors:
    """Everything a network needs for one slice (train/val/test) of one fold."""

    CLIP = 5.0

    def __init__(self, samples: Samples, rows: pd.DataFrame, scaler: FoldScaler,
                 target: str, lookback: int, device: torch.device) -> None:
        self.samples = samples
        self.index = rows.index
        self.lookback = lookback
        self.scaler = scaler
        self.device = device
        self.hour_pos = rows["hour_pos"].to_numpy()
        self.sym_pos = rows["sym_pos"].to_numpy()
        static = (rows[ALL_FEATURES].to_numpy(np.float32) - scaler.static_mean) / scaler.static_std
        self.static = torch.from_numpy(np.clip(static, -self.CLIP, self.CLIP))
        y = rows[target].to_numpy(np.float32)
        if target == "y_demeaned":
            y = y / scaler.y_scale
        self.y = torch.from_numpy(y)
        self._offsets = np.arange(-lookback + 1, 1)

    def __len__(self) -> int:
        return len(self.hour_pos)

    def batch(self, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = self.hour_pos[idx, None] + self._offsets[None, :]
        seq = self.samples.seq[rows, self.sym_pos[idx, None], :]
        seq = (seq - self.scaler.seq_mean) / self.scaler.seq_std
        seq = torch.from_numpy(np.clip(seq, -self.CLIP, self.CLIP).astype(np.float32))
        return (
            seq.to(self.device, non_blocking=True),
            self.static[idx].to(self.device, non_blocking=True),
            self.y[idx].to(self.device, non_blocking=True),
        )

    def batches(self, batch_size: int, shuffle: bool, rng: np.random.Generator | None = None):
        order = rng.permutation(len(self)) if shuffle else np.arange(len(self))
        for i in range(0, len(order), batch_size):
            yield self.batch(order[i : i + batch_size])


# --------------------------------------------------------------------------- #
# Losses / Kayıplar
# --------------------------------------------------------------------------- #
class PinballLoss(nn.Module):
    def __init__(self, quantiles=config.QUANTILES) -> None:
        super().__init__()
        self.register_buffer("q", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = y.unsqueeze(-1) - pred
        return torch.maximum(self.q * diff, (self.q - 1) * diff).mean()


class PointLoss(nn.Module):
    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(pred.squeeze(-1), y)


# --------------------------------------------------------------------------- #
# Training / Eğitim
# --------------------------------------------------------------------------- #
@dataclass
class History:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    epoch_seconds: list = field(default_factory=list)
    best_epoch: int | None = None
    best_val_loss: float = float("inf")
    stopped_early: bool = False

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@torch.no_grad()
def _evaluate(model, data: FoldTensors, criterion, batch_size: int) -> float:
    model.eval()
    total, n = 0.0, 0
    for seq, static, y in data.batches(batch_size, shuffle=False):
        total += criterion(model(seq, static), y).item() * y.size(0)
        n += y.size(0)
    return total / max(n, 1)


def train_network(
    model: nn.Module,
    train: FoldTensors,
    val: FoldTensors,
    device: torch.device,
    quantile: bool = False,
    epochs: int = config.EPOCHS,
    lr: float = config.LEARNING_RATE,
    batch_size: int = config.BATCH_SIZE,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    verbose: bool = True,
) -> tuple[nn.Module, History]:
    """Train on one fold and restore the best validation epoch."""
    model = model.to(device)
    criterion = (PinballLoss() if quantile else PointLoss()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, patience // 2)
    )
    rng = np.random.default_rng(config.SEED)
    history = History()
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    stale = 0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running, n = 0.0, 0
        for seq, static, y in train.batches(batch_size, shuffle=True, rng=rng):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(seq, static), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * y.size(0)
            n += y.size(0)
        val_loss = _evaluate(model, val, criterion, batch_size)
        scheduler.step(val_loss)

        history.train_loss.append(running / n)
        history.val_loss.append(val_loss)
        history.epoch_seconds.append(time.time() - t0)
        improved = val_loss < history.best_val_loss - 1e-5
        if improved:
            history.best_val_loss, history.best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if verbose:
            print(f"    epoch {epoch + 1:>2}/{epochs}  train {running / n:.4f}  "
                  f"val {val_loss:.4f}  ({history.epoch_seconds[-1]:.1f}s){' *' if improved else ''}")
        if stale >= patience:
            history.stopped_early = True
            break

    model.load_state_dict(best_state)
    return model.to(device), history


@torch.no_grad()
def predict_network(model: nn.Module, data: FoldTensors, quantile: bool = False,
                    batch_size: int = 2048) -> np.ndarray:
    """Point predictions (n,) or sorted quantiles in return units (n, n_q)."""
    model.eval()
    outs = [model(seq, static).cpu().numpy() for seq, static, _ in
            data.batches(batch_size, shuffle=False)]
    out = np.concatenate(outs, axis=0)
    if not quantile:
        return out[:, 0]
    # EN: sorting removes quantile crossing without changing a well-ordered row.
    # TR: sıralamak, düzgün sıralı bir satırı değiştirmeden quantile kesişmesini gideriyor.
    return np.sort(out, axis=1) * data.scaler.y_scale

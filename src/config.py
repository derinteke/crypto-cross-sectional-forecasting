"""Every setting that shapes a result lives here, in one place.

EN: If a number changes the outcome of a backtest, it belongs in this file and
    not buried in a function default. That makes the whole study auditable: a
    reader can see every choice I made before any model saw any data.
TR: Bir sayı backtest'in sonucunu değiştiriyorsa bu dosyada durmalı, bir
    fonksiyonun varsayılan argümanına gömülmemeli. Böylece çalışmanın tamamı
    denetlenebilir oluyor: okuyan kişi, hiçbir model veriyi görmeden önce
    yaptığım her seçimi görebiliyor.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths / Yollar
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --------------------------------------------------------------------------- #
# Data source / Veri kaynağı
# --------------------------------------------------------------------------- #
# EN: Binance's public archive. It keeps the files of delisted pairs, which is
#     exactly what I need to avoid survivorship bias.
# TR: Binance'in herkese açık arşivi. Delist edilmiş paritelerin dosyalarını da
#     saklıyor; survivorship bias'tan kaçınmak için tam ihtiyacım olan şey bu.
BINANCE_DATA_URL = "https://data.binance.vision"
BINANCE_LISTING_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
QUOTE_ASSET = "USDT"

DATA_START_MONTH = "2019-01"
DATA_END_MONTH = "2026-08"  # last complete month / son tamamlanmış ay

# EN: Bases that are not risky assets: stablecoins, fiat and gold tokens. Their
#     returns are ~0 by design, so ranking them against crypto is meaningless.
# TR: Riskli varlık olmayan baz varlıklar: stablecoin'ler, fiat ve altın
#     tokenları. Getirileri tasarım gereği ~0; kriptoyla sıralamak anlamsız.
EXCLUDED_BASES = frozenset(
    {
        "USDC", "BUSD", "TUSD", "USDP", "PAX", "DAI", "FDUSD", "UST", "USTC",
        "USDS", "USDSB", "SUSD", "EUR", "GBP", "AUD", "TRY", "BRL", "RUB",
        "UAH", "NGN", "ZAR", "BIDR", "IDRT", "BKRW", "BVND", "PAXG", "EURI",
        "AEUR", "XUSD", "USD1", "BFUSD", "RLUSD", "USDE", "PYUSD", "XAUT", "U",
        # EN: wrapped / staked copies of BTC, ETH, SOL: same asset, same return.
        # TR: BTC, ETH, SOL'ün sarmalanmış / stake edilmiş kopyaları: aynı varlık.
        "WBTC", "WBETH", "BETH", "BNSOL",
        # EN: 2019-20 3x leveraged BTC tokens. / TR: 3x kaldıraçlı BTC tokenları.
        "BULL", "BEAR",
    }
)

# EN: Tokenized US equities and ETFs (Binance lists them with a trailing "B",
#     e.g. NVDAB = Nvidia) arrived in 2025-26. They are not crypto and trade on
#     stock-market hours, so they do not belong in a crypto cross-section. I list
#     them explicitly rather than guess with a regex: an explicit list is
#     auditable.
# TR: Tokenize ABD hisseleri ve ETF'leri (Binance sonlarına "B" ekliyor, ör.
#     NVDAB = Nvidia) 2025-26'da geldi. Kripto değiller ve borsa saatlerinde
#     işlem görüyorlar; dolayısıyla bir kripto kesitine ait değiller. Regex ile
#     tahmin etmek yerine açıkça listeliyorum: açık bir liste denetlenebilir.
TOKENIZED_EQUITY_BASES = frozenset(
    {
        "AAOIB", "AAPLB", "ALABB", "AMATB", "AMDB", "AMZNB", "ARMB", "ASMLB",
        "ASTSB", "AVGOB", "AXTIB", "BABAB", "BMNRB", "CBRSB", "COHRB", "COINB",
        "CRCLB", "CRDOB", "CRWVB", "DELLB", "DJTB", "DRAMB", "EWYB", "FLNCB",
        "GLWB", "GMEB", "GOOGLB", "HOODB", "IBMB", "INTCB", "INTWB", "IRENB",
        "KORUB", "LITEB", "METAB", "MRVLB", "MSFTB", "MSTRB", "MUUB", "MVLLB",
        "NBISB", "NFLXB", "NOKB", "NVDAB", "ORCLB", "PLTRB", "PYPLB", "QCOMB",
        "QNTB", "QQQB", "RKLBB", "SKHYB", "SMCIB", "SMHB", "SNDKB", "SNXXB",
        "SOXLB", "SOXSB", "SPCXB", "SPYB", "TQQQB", "TSLAB", "TSMB", "USARB",
        "WDCB",
    }
)

# --------------------------------------------------------------------------- #
# Universe / Evren
# --------------------------------------------------------------------------- #
UNIVERSE_SIZE = 30
UNIVERSE_LOOKBACK_DAYS = 30   # volume ranking window / hacim sıralama penceresi
MIN_HISTORY_DAYS = 60         # a coin must be this old to enter / en az bu yaşta
MIN_COVERAGE = 0.95           # share of days with data in the window
# EN: data-driven stablecoin guard: a coin whose daily returns barely move is
#     pegged to something, whatever its name. Catches stables I did not list.
# TR: veriye dayalı stablecoin koruması: günlük getirileri neredeyse hiç
#     oynamayan bir coin, adı ne olursa olsun bir şeye sabitlenmiştir.
#     Listelemediğim stablecoin'leri de yakalıyor.
STABLE_MAX_DAILY_VOL = 0.005

# --------------------------------------------------------------------------- #
# Task / Görev
# --------------------------------------------------------------------------- #
DECISION_HOUR_UTC = 0     # I decide once a day at 00:00 UTC
EXECUTION_LAG_HOURS = 1   # ...and can only trade one hour later
HORIZON_HOURS = 24        # holding period / elde tutma süresi
SEQ_LOOKBACK_HOURS = 120  # 5 days of hourly history for the sequence models
IC_DECAY_DAYS = 5         # how many days ahead I check signal decay
FEATURE_WARMUP_HOURS = 720  # the longest feature window (30 days)

# EN: A trading halt this long means the coin was swapped, redenominated or
#     relaunched under the same ticker (LUNA -> LUNA 2.0, SUN, BNX, STRAX).
#     The price after the gap is not the same asset as before it, so no feature
#     may look across it. Samples resume once a full warm-up has passed.
# TR: Bu kadar uzun bir işlem durdurma, coinin swap edildiği, redenominasyona
#     uğradığı ya da aynı sembolle yeniden çıkarıldığı anlamına geliyor (LUNA ->
#     LUNA 2.0, SUN, BNX, STRAX). Boşluktan sonraki fiyat, öncekiyle aynı varlık
#     değil; dolayısıyla hiçbir özellik onun öbür tarafına bakamaz. Örnekler tam
#     bir ısınma süresi geçtikten sonra yeniden başlıyor.
LONG_GAP_HOURS = 72

# --------------------------------------------------------------------------- #
# Walk-forward / İleriye doğru yürüyen doğrulama
# --------------------------------------------------------------------------- #
FIRST_SAMPLE_DATE = "2019-03-01"
FIRST_TEST_DATE = "2022-01-01"
RETRAIN_MONTHS = 6
VAL_MONTHS = 3
EMBARGO_DAYS = 2  # > horizon + execution lag, so no label straddles a gap

# --------------------------------------------------------------------------- #
# Portfolio and costs / Portföy ve maliyetler
# --------------------------------------------------------------------------- #
QUANTILE_BUCKETS = 5          # long the top fifth, short the bottom fifth
COST_BPS = 15.0               # per side: taker fee + slippage
COST_GRID_BPS = (0.0, 10.0, 20.0, 30.0)
PERIODS_PER_YEAR = 365        # crypto trades every day
# EN: fixed before any test result was seen; see backtest.smooth_signal.
# TR: herhangi bir test sonucu görülmeden önce sabitlendi; bkz. backtest.smooth_signal.
SMOOTHING_HALFLIFE_DAYS = 3.0

# --------------------------------------------------------------------------- #
# Probabilistic forecasts / Olasılıksal tahminler
# --------------------------------------------------------------------------- #
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# --------------------------------------------------------------------------- #
# Training / Eğitim
# --------------------------------------------------------------------------- #
SEED = 42
BATCH_SIZE = 512
EPOCHS = 30
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 0  # Windows + DataLoader workers = pain; data fits in memory

# --------------------------------------------------------------------------- #
# Smoke test / Duman testi
# --------------------------------------------------------------------------- #
SMOKE_SYMBOLS = 10
SMOKE_START = "2023-01-01"
SMOKE_FIRST_TEST = "2024-01-01"
SMOKE_END = "2024-12-31"
SMOKE_EPOCHS = 1

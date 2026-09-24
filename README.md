# Crypto Cross-Sectional Forecasting

*Türkçe açıklama için: [Türkçe](#türkçe)*

Every day at 00:00 UTC I rank the ~30 most liquid coins on Binance by how I expect them to do against each other over the next 24 hours, and trade that ranking as a long-short book. Most of the work went into the second half of the project, which was trying to show the result isn't real. That meant a point-in-time universe, walk-forward validation with an embargo, trading costs, a regression on the market to separate beta from alpha, two null controls, and a deflated Sharpe ratio that accounts for how many strategies I tried.

The short version: the ranking signal is clearly there. Whether it survives costs well enough to trade is something I can't prove.

![Raw vs smoothed signal, net of costs](reports/figures/equity_smoothed_full.png)

## Results

Everything below is out of sample: ten walk-forward folds stitched together, January 2022 to August 2026, 1,703 trading days, 15 bps cost per side.

| Strategy | Rank IC | IC t (NW) | Sharpe gross | Sharpe net | Turnover / day | β to market | α bps/day (t) | PSR | Deflated Sharpe |
|---|---|---|---|---|---|---|---|---|---|
| **LightGBM (smoothed)** | 0.109 | 15.5 | 1.80 | **1.45** | 0.44 | −0.51 | 30.3 (4.3) | 0.999 | **0.12** |
| GRU (smoothed) | 0.108 | 15.4 | 1.78 | 1.46 | 0.41 | −0.54 | 30.0 (4.5) | 0.999 | 0.12 |
| Transformer (smoothed) | 0.107 | 15.2 | 1.68 | 1.35 | 0.40 | −0.52 | 27.8 (4.1) | 0.998 | 0.08 |
| LightGBM | **0.118** | **17.1** | 2.36 | 1.12 | 1.51 | −0.42 | **40.5 (5.6)** | 0.991 | 0.03 |
| Transformer | 0.114 | 16.5 | 2.32 | 1.13 | 1.45 | −0.43 | 39.3 (5.5) | 0.992 | 0.03 |
| GRU | 0.113 | 16.2 | 2.06 | 0.85 | 1.46 | −0.45 | 34.3 (5.1) | 0.966 | 0.01 |
| LightGBM-Q (median / width) | 0.104 | 15.3 | 2.74 | 0.86 | 1.98 | −0.33 | 41.1 (6.0) | 0.967 | 0.01 |
| *Ridge (baseline)* | 0.111 | 15.4 | 1.10 | −0.06 | 1.49 | −0.55 | 17.6 (2.5) | 0.450 | 0.00 |
| *Momentum 7d (baseline)* | −0.015 | −2.4 | 0.60 | −0.44 | 1.23 | −0.13 | 9.8 (1.2) | 0.176 | 0.00 |
| *Reversal 24h (baseline)* | 0.024 | 4.0 | −0.03 | −2.85 | 3.07 | +0.10 | 0.2 (0.0) | 0.000 | 0.00 |
| *Random (null control)* | −0.006 | −1.3 | −0.06 | −4.23 | 3.22 | +0.01 | −0.5 (−0.1) | 0.000 | – |
| *LightGBM, shuffled target (null control)* | −0.010 | −2.0 | −0.65 | −3.41 | 2.54 | +0.04 | −8.6 (−1.4) | 0.000 | – |

For comparison, the market the β is measured against (the equal-weight universe, long only, no costs) had a Sharpe of −0.31 and a max drawdown of −94% over the same period.

The quantile models predict nine quantiles of each coin's next-day relative return:

| Model | Pinball loss | vs. baseline | 80% interval coverage |
|---|---|---|---|
| Transformer-Q | 0.00942 | −2.1% | 80.2% |
| GRU-Q | 0.00942 | −2.1% | 81.2% |
| LightGBM-Q | 0.00943 | −2.0% | 80.1% |
| *Volatility-scaled empirical quantiles (baseline)* | 0.00962 | – | 79.3% |

### What I take from this

1. The ranking signal is real. Rank IC is 0.11–0.12 with a Newey-West t-stat around 17, it's positive in every calendar year, and after regressing out the market about 40 bps/day of alpha is left (t ≈ 5.6). Both null controls sit at zero, so the pipeline isn't making the signal up.

2. A good IC doesn't mean a good PnL. Most of the IC comes from the low-volatility, or "lottery", effect: the most volatile coins have a terrible median day, but every now and then a huge pump. On the data before the test period (2019-2021), going from the calmest fifth of coins to the most volatile fifth, the median next-day relative return drops from −27 to −98 bps while the mean hardly moves (+1 to +4 bps). Spearman IC rewards getting the median right, but a portfolio earns the mean. That's how Ridge can have an IC of 0.11 and a gross Sharpe of only 1.1.

3. Beta isn't alpha. Every book is dollar neutral, but its beta is around −0.4 to −0.5, because it ends up long BTC, ETH and BNB and short the newest, most volatile coins (WIF, PEPE, ENA, WLD...). Between 2022 and 2026 that tilt alone would have made money, since the equal-weight universe lost 94%. So next to the Sharpe I report the intercept of a Newey-West regression on the market.

4. Trees beat the linear model, and sequence models don't beat trees. From the same features LightGBM gets more than twice Ridge's alpha (40 vs 18 bps/day). The GRU and Transformer see the last 120 hours of hourly data on top of the same daily snapshot and still add nothing over LightGBM, which trains in seconds per fold instead of minutes.

5. Costs decide it, and the net edge isn't proven. The raw forecasts turn the book over 1.5 times a day, and at 15 bps a side that eats half the gross Sharpe. Smoothing the ranks with a 3-day exponential average, which I fixed before looking at any test results, cuts turnover by 70% and brings net Sharpe up to about 1.45 (PSR 99.9%). But I evaluated 12 candidate strategies, and once that's accounted for the deflated Sharpe comes out around 0.12, far below 0.95. I can't claim the edge after costs isn't luck.

6. The quantile models are well calibrated but don't add much. They beat a simple "recent volatility × historical shape" baseline by only about 2% in pinball loss, and sizing positions by median / interval width didn't beat the plain point forecast after costs.

| | |
|---|---|
| ![Quintile ladder](reports/figures/quintiles_full.png) | ![Cost sensitivity](reports/figures/cost_sensitivity_full.png) |
| ![Rolling IC](reports/figures/rolling_ic_full.png) | ![Calibration](reports/figures/calibration_full.png) |

## What I was careful about

| Trap | What I did | Checked by |
|---|---|---|
| Survivorship bias | The universe is re-picked every month from the 30 days *before* the month, using Binance's archive, which keeps delisted pairs. 65 of the 257 coins stopped trading during the sample (LUNA, FTT, MATIC...). | `test_universe_ignores_everything_after_the_month_start`, `test_delisted_coin_is_in_the_universe_until_it_dies` |
| Look-ahead in features | Every feature is a causal rolling window, and bars are indexed by close time. | `test_features_do_not_change_when_the_future_is_deleted`: delete everything after t, recompute, and require identical values |
| Trading on the signal bar | One hour of execution lag, so the target runs from t+1h to t+25h. | `test_target_is_exactly_the_next_24h_after_a_one_hour_lag` |
| Overlapping labels across splits | Expanding walk-forward, 3-month validation, and a 2-day embargo between every slice. | `test_folds_are_ordered_and_embargoed` |
| Scaler leakage | Every normaliser is fitted per fold, on train rows only. | `test_scaler_statistics_come_from_train_only` |
| Invented prices | Missing hours are never interpolated; any window that touches one is dropped. | `test_no_sample_reads_a_missing_bar` |
| A pipeline that finds signal in noise | Random scores, and a LightGBM trained on targets shuffled within each day. | Both have IC ≈ 0 out of sample |
| Mistaking beta for alpha | Every book is regressed on the equal-weight market with Newey-West errors. | `beta_to_market`, `alpha_tstat` columns |
| Picking the lucky strategy | Deflated Sharpe ratio (Bailey & López de Prado) over all 12 candidate strategies. | `deflated_sharpe` column |
| Tuning on the test set | The smoothing half-life was fixed in `config.py` before I saw any test result, and there's no hyper-parameter search on test. | `test_smoothing_is_causal` |

To make sure the tests actually catch these mistakes, I also mutation-tested the suite. Planting a centred rolling window, removing the missing-bar rule or removing the token-swap rule each makes the relevant test fail.

### Three data problems I ran into

1. A stablecoin nobody told me about. `U` (2026) is pegged at 1.00 and made it into the top 30 by volume. A list of names can't catch stablecoins that don't exist yet, so I added a check based on the data itself: if daily volatility over the ranking window is below 0.5%, the coin counts as pegged, whatever its ticker.
2. Token swaps that reuse a ticker. After multi-day trading halts, `LUNAUSDT` came back as LUNA 2.0 (a +12 log return in a single "hour"), and SUN, BNX, STRAX and BTCST were redenominated. A 30-day momentum feature that spans the gap reads a swap as a 100,000x move. Now any halt of 72 hours or more breaks the series. There are no samples until a full 30-day warm-up has passed, and the universe requires 60 days of *continuous* trading, so LUNA 2.0 has to earn its history again.
3. Timestamps that change unit. In 2025 Binance switched its spot files from milliseconds to microseconds. Read naively, 2025 lands somewhere around the year 50,000. The loader now detects the unit value by value.

A few groups are also excluded by name: stablecoins and fiat, wrapped or staked copies (WBTC, WBETH, BNSOL...), leveraged tokens (BTCUP, ETHDOWN, BULL, BEAR, but not JUP) and the tokenized US equities Binance listed in 2025-26 (NVDAB, TSLAB...).

## The setup in detail

| | |
|---|---|
| Data | Binance spot klines from data.binance.vision, hourly, 2019-01 to 2026-08. 590 candidate USDT pairs, 257 of which were ever in the universe. |
| Universe | Top 30 by mean 30-day USDT volume, re-picked on the first day of each month. A coin must have traded on at least 95% of the last 60 days and must not be pegged. |
| Decision | Once a day at 00:00 UTC; enter at 01:00, exit 24 hours later. |
| Target | Log return from t+1h to t+25h. Point models train on its within-day rank mapped to a normal distribution; quantile models on the within-day demeaned return. |
| Features | 12 per coin (returns from 1h to 30d, realised vol, high-low range, volume shock, Amihud illiquidity, 30-day beta to BTC), their within-day ranks, and 3 BTC market features. Sequence models also get 5 hourly channels over the last 120 hours. |
| Portfolio | Long the top fifth, short the bottom fifth, equal weight, dollar neutral, rebalanced daily. |
| Validation | Expanding walk-forward with 10 folds: retrain every 6 months, 3-month validation, 2-day embargo. The test period is 2022-01 to 2026-08. |
| Metrics | Rank IC (mean, ICIR, Newey-West t), IC decay, quintile ladder, gross and net Sharpe, max drawdown, turnover, β and α against the market, PSR, deflated Sharpe; pinball loss, interval coverage and calibration for the quantile models. |

## Project layout

```
crypto-cross-sectional-forecasting/
├── src/
│   ├── config.py        every setting that can change a result
│   ├── download.py      archive listing, resumable parallel download, checksums
│   ├── data.py          raw zips -> wide (time x symbol) panels; ms/us timestamps
│   ├── universe.py      point-in-time top-30, stablecoin guard
│   ├── features.py      features, targets (the only forward-looking code), samples
│   ├── splits.py        walk-forward folds with embargo
│   ├── baselines.py     random, momentum, reversal, Ridge, vol-scaled quantiles
│   ├── models.py        LightGBM, GRU, Transformer (point or quantile head)
│   ├── train.py         device check, fold-local scaling, training loop, pinball loss
│   ├── backtest.py      portfolios, costs, turnover, smoothing
│   ├── metrics.py       IC, Newey-West, Sharpe, drawdown, DSR, beta/alpha, pinball
│   ├── walkforward.py   run every model through every fold, then score
│   └── evaluate.py      figures
├── scripts/
│   ├── download_data.py     daily bars -> universe -> hourly bars (~30 min first time)
│   └── run_walkforward.py   baselines first, then models, then the scorecard
├── tests/test_pipeline.py   25 tests on a synthetic market with planted traps
├── notebooks/crypto_cross_sectional_end_to_end.ipynb   the whole story, executed
└── reports/                 result tables, universe, data quality, figures
```

## Setup

I use Python 3.10. PyTorch goes in first, from its own index, because the CUDA build isn't on PyPI. My GPU is a GTX 1050 Ti (Pascal, sm_61), and PyTorch's newer cu128 wheels no longer support Pascal, so I use cu126.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## How to run

```bash
pytest -q                                     # 25 leak / correctness tests, ~30 s, no download needed
python scripts/download_data.py               # ~1.2 GB of zips, ~30 min the first time, resumable
python scripts/run_walkforward.py --smoke     # 10 coins, 2 folds, 1 epoch: proves the pipeline in ~1 min
python scripts/run_walkforward.py             # full run: ~70 min on a GTX 1050 Ti
python scripts/run_walkforward.py --rescore   # re-score saved predictions without retraining
```

The notebook loads the saved predictions from the full run, retrains fold 1 live, and checks that the LightGBM predictions match the saved ones exactly.

## Limitations

- The short side assumes there's a perpetual future for every coin, and funding payments and borrow availability aren't modelled. A lot of the edge comes from the short side, in exactly the coins where that assumption is weakest.
- Costs are a flat 15 bps per side with no market-impact model, and I assume fills at the hourly close.
- The test period (2022-2026) is mostly an altcoin bear market, which flatters any book that's short high-beta coins. That's the reason for the β/α split.
- Weights are reset to target every day, so intraday drift is ignored in turnover.

Things I'd like to try next: beta- and volatility-neutral portfolio construction, funding rates as both a cost and a feature, and longer holding periods.

## References

- Binance public market data: https://data.binance.vision
- Bailey, D. H. & López de Prado, M. (2014). *The Deflated Sharpe Ratio.* Journal of Portfolio Management.
- Newey, W. K. & West, K. D. (1987). *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.* Econometrica.
- Bali, T. G., Cakici, N. & Whitelaw, R. F. (2011). *Maxing Out: Stocks as Lotteries and the Cross-Section of Expected Returns.* Journal of Financial Economics.
- Liu, Y., Tsyvinski, A. & Wu, X. (2022). *Common Risk Factors in Cryptocurrency.* Journal of Finance.

---

## Türkçe

Her gün 00:00 UTC'de Binance'teki en likit 30 kadar coini, önümüzdeki 24 saatte birbirlerine göre nasıl bir performans göstereceklerini tahmin ederek sıralıyorum ve bu sıralamayla long-short bir portföy kuruyorum. Asıl emeği ise işin ikinci yarısına verdim, yani sonucun gerçek olmadığını göstermeye çalışmaya. Bunun için evreni her ay yalnızca o tarihte bilinen bilgiyle seçtim (point-in-time), embargo'lu walk-forward doğrulama kullandım, işlem maliyetlerini hesaba kattım, beta ile alpha'yı ayırmak için piyasaya karşı regresyon kurdum, iki null kontrol çalıştırdım ve kaç strateji denediğimi de hesaba katan deflated Sharpe oranına baktım.

Kısacası sıralama sinyali gerçekten var. Ama maliyetlerden sonra üzerine işlem yapmaya değecek kadar güçlü olduğunu kanıtlayamıyorum.

### Sonuçlar

Tablolar yukarıda. Hepsi örneklem dışı: on walk-forward fold'un birleştirilmesiyle Ocak 2022 ile Ağustos 2026 arası, 1.703 işlem günü, taraf başına 15 bps maliyet. Karşılaştırma için, β'nın ölçüldüğü piyasa (eşit ağırlıklı evren, sadece long, maliyetsiz) aynı dönemde −0,31 Sharpe ve −%94 maksimum düşüş gördü.

Buradan çıkardıklarım:

1. Sıralama sinyali gerçek. Rank IC 0,11 ile 0,12 arasında, Newey-West t-istatistiği 17 civarında ve her takvim yılında pozitif. Piyasanın etkisi regresyonla çıkarıldıktan sonra bile günde yaklaşık 40 bps alpha kalıyor (t ≈ 5,6). İki null kontrol de sıfırda, yani sinyali kurduğum düzenek uydurmuyor.

2. Yüksek IC, kârlı portföy demek değil. IC'nin büyük kısmı düşük volatilite ya da "piyango" etkisinden geliyor: en oynak coinlerin medyan günü çok kötü, ama arada bir çok sert yükseliyorlar. Test öncesi veride (2019-2021) en sakin beşte birlik dilimden en oynak dilime gidildikçe ertesi günün medyan göreli getirisi −27 bps'den −98 bps'ye düşüyor, ortalama ise neredeyse hiç değişmiyor (+1 ile +4 bps). Spearman IC medyanı doğru tahmin etmeyi ödüllendiriyor, portföy ise ortalamayı kazanıyor. Ridge'in IC'si 0,11 olduğu halde gross Sharpe'ının 1,1'de kalması bundan.

3. Beta, alpha değil. Portföylerin hepsi dolar bazında nötr ama betaları −0,4 ile −0,5 civarında, çünkü BTC, ETH ve BNB'de long, en yeni ve en oynak coinlerde (WIF, PEPE, ENA, WLD...) short kalıyorlar. Eşit ağırlıklı evren 2022-2026 arasında %94 değer kaybettiği için bu eğilim tek başına bile para kazandırırdı. Bu yüzden Sharpe'ın yanında, piyasaya karşı kurulan Newey-West regresyonunun sabit terimini de raporluyorum.

4. Ağaç modeli doğrusal modeli geçiyor, dizi modelleri ise ağacı geçemiyor. LightGBM aynı özelliklerden Ridge'in iki katından fazla alpha çıkarıyor (günde 40'a karşı 18 bps). GRU ve Transformer aynı günlük verinin üstüne son 120 saatin saatlik verisini de görüyor, yine de LightGBM'in üzerine bir şey koyamıyorlar. Üstelik LightGBM her fold'da dakikalar değil, saniyeler içinde eğitiliyor.

5. Sonucu maliyetler belirliyor ve maliyet sonrası kazanç kanıtlanmış değil. Ham tahminlerle portföy günde 1,5 kez dönüyor; taraf başına 15 bps ile bu, gross Sharpe'ın yarısını götürüyor. Sıralamaları 3 günlük üstel ortalamayla yumuşatmak (bunu test sonuçlarına bakmadan önce sabitledim) turnover'ı %70 azaltıyor ve net Sharpe'ı 1,45 civarına çıkarıyor (PSR %99,9). Fakat toplam 12 aday strateji denedim ve bu hesaba katıldığında deflated Sharpe 0,12 civarına iniyor, 0,95'in çok altına. Yani maliyet sonrası kazancın şans eseri olmadığını söyleyemem.

6. Quantile modelleri iyi kalibre ama pek bir şey katmıyor. "Son dönem volatilitesi × geçmişteki dağılım şekli" gibi basit bir yöntemi pinball loss'ta sadece %2 kadar geçiyorlar. Pozisyonları medyan / aralık genişliğine göre boyutlandırmak da maliyetlerden sonra düz nokta tahminini geçemedi.

### Nelere dikkat ettim

Her tuzağın karşılığı yukarıdaki tabloda, çoğunu yakalayan ayrı bir test de var. Kısaca:

- Survivorship bias: Evreni her ay, o aydan önceki 30 güne bakarak Binance arşivinden yeniden seçiyorum. Arşiv delist edilen pariteleri de sakladığı için ölen coinler de evrende kalıyor. Örneklemdeki 257 coinden 65'i dönem içinde işlemden kalktı (LUNA, FTT, MATIC...).
- Geleceğe bakan özellikler: Her özellik yalnızca geçmişi gören bir kayan pencere. İlgili test, t'den sonraki her şeyi silip özellikleri yeniden hesaplıyor ve değerlerin birebir aynı çıkmasını bekliyor.
- Sinyal barında işlem yapmak: Sinyalle işlem arasında bir saat gecikme var; hedef t+1 saatten t+25 saate kadar olan getiri.
- Split'ler arasında çakışan etiketler: Genişleyen walk-forward, 3 aylık validation ve her dilim arasında 2 günlük embargo.
- Scaler sızıntısı: Her normalizasyon her fold'da yalnızca train satırlarıyla fit ediliyor.
- Uydurma fiyatlar: Eksik saatler hiçbir zaman interpolasyonla doldurulmuyor, eksik bir saate değen pencere atılıyor.
- Gürültüde sinyal bulan bir düzenek: Rastgele skorlar ve hedefleri her gün kendi içinde karıştırılmış bir LightGBM. İkisinin de IC'si sıfır civarında.
- Betayı alpha sanmak: Her portföy, eşit ağırlıklı piyasaya karşı Newey-West regresyonuyla ayrıştırılıyor.
- Şanslı stratejiyi seçmek: 12 adayın hepsi üzerinden deflated Sharpe hesaplanıyor.
- Test setine göre ayar yapmak: Yumuşatmanın yarı ömrü, hiçbir test sonucu görülmeden `config.py` içinde sabitlendi.

Testlerin bu hataları gerçekten yakaladığından emin olmak için mutasyon testi de yaptım. Koda ortalanmış bir kayan pencere eklemek ya da eksik bar veya token swap kuralını kaldırmak, ilgili testi kırıyor.

### Veride karşılaştığım üç sorun

1. Kimsenin haber vermediği bir stablecoin. `U` (2026) 1,00'a sabitli ve hacimde ilk 30'a girdi. İsim listesi henüz var olmayan stablecoin'leri yakalayamayacağı için veriye bakan bir kontrol ekledim: sıralama penceresinde günlük volatilitesi %0,5'in altında kalan coin, adı ne olursa olsun sabitli sayılıyor.
2. Aynı sembolle geri dönen token swap'ları. Birkaç gün süren işlem durdurmalarından sonra `LUNAUSDT` LUNA 2.0 olarak geri geldi (tek bir "saatte" +12 log getiri); SUN, BNX, STRAX ve BTCST'de de redenominasyon yapıldı. Bu boşluğun üzerinden hesaplanan 30 günlük momentum, swap'ı 100.000 katlık bir hareket gibi okuyor. Artık 72 saat veya daha uzun süren bir durdurma seriyi kesiyor. 30 günlük ısınma süresi dolmadan örnek üretilmiyor, evrene girmek için de 60 gün *kesintisiz* işlem görmek gerekiyor. Yani LUNA 2.0 geçmişini sıfırdan biriktirmek zorunda.
3. Birimi değişen zaman damgaları. Binance 2025'te spot dosyalarında milisaniyeden mikrosaniyeye geçti. Buna dikkat etmeden okuyunca 2025, 50.000'li yıllara düşüyor. Yükleyici artık birimi her değer için ayrı ayrı tespit ediyor.

Bunlara ek olarak bazı gruplar isimden dışarıda bırakılıyor: stablecoin'ler ve fiat, sarmalanmış ya da stake edilmiş kopyalar (WBTC, WBETH, BNSOL...), kaldıraçlı tokenlar (BTCUP, ETHDOWN, BULL, BEAR; JUP hariç) ve Binance'in 2025-26'da listelediği tokenize ABD hisseleri (NVDAB, TSLAB...).

### Kurulum ve çalıştırma

Python 3.10 kullanıyorum. PyTorch'un CUDA sürümü PyPI'da olmadığı için önce onu kendi index'inden kuruyorum. Ekran kartım GTX 1050 Ti (Pascal, sm_61). PyTorch'un yeni cu128 paketleri Pascal'ı artık desteklemediği için cu126 kullanıyorum. Komutlar yukarıdaki [Setup](#setup) ve [How to run](#how-to-run) bölümlerinde. Testler veri indirmeden yaklaşık 30 saniyede çalışıyor. Verinin (~1,2 GB) ilk indirilmesi 30 dakika kadar sürüyor, tam koşu da GTX 1050 Ti'de yaklaşık 70 dakika.

Notebook, tam koşunun kaydedilmiş tahminlerini yüklüyor, fold 1'i baştan eğitiyor ve LightGBM tahminlerinin kaydedilenlerle birebir aynı olduğunu kontrol ediyor.

### Sınırlamalar

- Short tarafında her coin için perpetual kontrat olduğunu varsayıyorum; funding ödemeleri ve ödünç alınabilirlik modelde yok. Kazancın önemli bir kısmı da short tarafından ve tam da bu varsayımın en zayıf olduğu coinlerden geliyor.
- Maliyet taraf başına sabit 15 bps, piyasa etkisi modeli yok. Saatlik kapanış fiyatından işlem yapılabildiğini varsayıyorum.
- Test dönemi (2022-2026) büyük ölçüde bir altcoin ayı piyasası. Bu da yüksek betalı coinleri short eden her portföyü olduğundan iyi gösteriyor; β/α ayrımını bu yüzden ekledim.
- Ağırlıklar her gün hedefe geri çekiliyor, turnover hesabında gün içindeki kaymayı yok sayıyorum.

Sırada denemek istediklerim: beta ve volatilite açısından nötr portföy kurmak, funding oranlarını hem maliyet hem özellik olarak eklemek ve daha uzun elde tutma süreleri.

---

*This is a research project, not investment advice. / Bu bir araştırma projesi, yatırım tavsiyesi değildir.*

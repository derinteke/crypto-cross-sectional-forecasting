"""Download Binance spot klines from the public archive, resumably.

EN: I use data.binance.vision instead of the REST API for one reason: the REST
    API forgets delisted pairs, the archive does not. A universe built only from
    coins that still trade today would quietly select the survivors, and every
    backtest on it would look better than reality.
    Downloads are idempotent: a file that is already on disk and opens as a
    valid zip is never fetched again, so an interrupted run just resumes.
TR: REST API yerine data.binance.vision kullanmamın tek bir nedeni var: REST API
    delist edilmiş pariteleri unutuyor, arşiv unutmuyor. Yalnızca bugün hâlâ
    işlem gören coinlerden kurulan bir evren sessizce hayatta kalanları seçer ve
    üzerindeki her backtest gerçekte olduğundan iyi görünür.
    İndirmeler idempotent: diskte zaten olan ve geçerli bir zip olarak açılan
    dosya bir daha indirilmez, dolayısıyla yarıda kalan bir koşu kaldığı yerden
    devam ediyor.
"""

from __future__ import annotations

import hashlib
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from src import config

_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
_LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


# --------------------------------------------------------------------------- #
# Listing / Listeleme
# --------------------------------------------------------------------------- #
def _list_bucket(prefix: str, delimiter: str | None = "/") -> tuple[list[str], list[str]]:
    """Return (common_prefixes, keys) under a prefix, following pagination."""
    prefixes: list[str] = []
    keys: list[str] = []
    marker = ""
    while True:
        params = {"prefix": prefix, "marker": marker}
        if delimiter:
            params["delimiter"] = delimiter
        resp = _get(config.BINANCE_LISTING_URL, params=params)
        root = ET.fromstring(resp.content)
        prefixes += [e.text for e in root.findall("s3:CommonPrefixes/s3:Prefix", _S3_NS)]
        keys += [e.text for e in root.findall("s3:Contents/s3:Key", _S3_NS)]
        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=_S3_NS)
        if truncated != "true":
            break
        next_marker = root.findtext("s3:NextMarker", default="", namespaces=_S3_NS)
        marker = next_marker or (keys[-1] if keys else prefixes[-1])
    return prefixes, keys


def list_spot_symbols() -> list[str]:
    """Every spot symbol the archive has ever published monthly klines for."""
    prefixes, _ = _list_bucket("data/spot/monthly/klines/")
    return sorted(p.rstrip("/").split("/")[-1] for p in prefixes)


def base_asset(symbol: str, quote: str = config.QUOTE_ASSET) -> str:
    return symbol[: -len(quote)]


def is_candidate(symbol: str, all_bases: set[str], quote: str = config.QUOTE_ASSET) -> bool:
    """Keep USDT pairs of genuine risky assets.

    EN: Leveraged tokens (BTCUP, ETHDOWN...) are dropped by checking that the part
        before the suffix is itself a listed base; a naive `endswith("UP")` would
        also throw away JUP.
    TR: Kaldıraçlı tokenları (BTCUP, ETHDOWN...) son ekten önceki kısmın kendisi
        listelenmiş bir baz varlık mı diye bakarak eliyorum; saf bir
        `endswith("UP")` kontrolü JUP'u da atardı.
    """
    if not symbol.endswith(quote):
        return False
    base = base_asset(symbol, quote)
    if not base.isascii() or not base.isalnum():
        return False
    if base in config.EXCLUDED_BASES or base in config.TOKENIZED_EQUITY_BASES:
        return False
    for suffix in _LEVERAGED_SUFFIXES:
        stem = base[: -len(suffix)]
        if base.endswith(suffix) and len(stem) >= 2 and stem in all_bases:
            return False
    return True


def candidate_symbols() -> list[str]:
    symbols = list_spot_symbols()
    usdt = [s for s in symbols if s.endswith(config.QUOTE_ASSET)]
    bases = {base_asset(s) for s in usdt}
    return [s for s in usdt if is_candidate(s, bases)]


def available_months(symbol: str, interval: str) -> list[str]:
    """Months (YYYY-MM) the archive holds for this symbol and interval."""
    _, keys = _list_bucket(f"data/spot/monthly/klines/{symbol}/{interval}/", delimiter=None)
    pat = re.compile(rf"{symbol}-{interval}-(\d{{4}}-\d{{2}})\.zip$")
    months = [m.group(1) for k in keys if (m := pat.search(k))]
    lo, hi = config.DATA_START_MONTH, config.DATA_END_MONTH
    return sorted(m for m in months if lo <= m <= hi)


# --------------------------------------------------------------------------- #
# Download / İndirme
# --------------------------------------------------------------------------- #
def _get(url: str, params: dict | None = None, retries: int = 5) -> requests.Response:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp
        except requests.HTTPError as exc:
            # EN: a 404 is an answer, not a glitch: never retry it.
            # TR: 404 bir aksaklık değil, bir cevap: asla yeniden deneme.
            if exc.response is not None and exc.response.status_code == 404:
                raise
            if attempt == retries - 1:
                raise
        except requests.RequestException:
            if attempt == retries - 1:
                raise
        time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def raw_path(symbol: str, interval: str, month: str) -> Path:
    return config.RAW_DIR / interval / symbol / f"{symbol}-{interval}-{month}.zip"


def _is_valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def download_month(symbol: str, interval: str, month: str, verify_checksum: bool) -> str:
    """Fetch one monthly zip. Returns 'cached', 'downloaded' or 'missing'."""
    path = raw_path(symbol, interval, month)
    if path.exists() and _is_valid_zip(path):
        return "cached"
    url = (
        f"{config.BINANCE_DATA_URL}/data/spot/monthly/klines/"
        f"{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    )
    try:
        content = _get(url).content
    except requests.HTTPError:
        return "missing"
    if verify_checksum:
        expected = _get(url + ".CHECKSUM").text.split()[0].strip().lower()
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise IOError(f"checksum mismatch for {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(content)
    tmp.replace(path)
    return "downloaded"


def download_symbols(
    symbols: list[str],
    interval: str,
    verify_checksum: bool = False,
    max_workers: int = 16,
    verbose: bool = True,
) -> dict[str, int]:
    """Download every available month for every symbol in parallel."""
    # EN: listing first tells me exactly which months exist, so a 404 later
    #     means a real problem rather than "the coin was not listed yet".
    # TR: önce listelemek hangi ayların var olduğunu kesin söylüyor; böylece
    #     sonradan gelen bir 404 "coin henüz listelenmemişti" değil gerçek bir
    #     sorun anlamına geliyor.
    jobs: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(available_months, s, interval): s for s in symbols}
        for fut in as_completed(futures):
            jobs += [(futures[fut], m) for m in fut.result()]

    counts = {"cached": 0, "downloaded": 0, "missing": 0}
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(download_month, s, interval, m, verify_checksum) for s, m in jobs
        ]
        for fut in as_completed(futures):
            counts[fut.result()] += 1
            done += 1
            if verbose and (done % 500 == 0 or done == len(jobs)):
                print(f"  [{interval}] {done}/{len(jobs)} files  {counts}")
    return counts

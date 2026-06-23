from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

# ----------------------------
# Project paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
# RAW_DATA_DIR = DATA_DIR / "raw"
# PROCESSED_DATA_DIR = DATA_DIR / "processed"
# EXTERNAL_DATA_DIR = DATA_DIR / "external"
CACHE_DIR = DATA_DIR / "cache"

REPORTS_DIR = PROJECT_ROOT / "reports"
# FIGURES_DIR = REPORTS_DIR / "figures"
# TABLES_DIR = REPORTS_DIR / "tables"


NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"

# Ensure core directories exist
for path in [
    DATA_DIR,
    # RAW_DATA_DIR,
    # PROCESSED_DATA_DIR,
    # EXTERNAL_DATA_DIR,
    CACHE_DIR,
    REPORTS_DIR,
    # FIGURES_DIR,
    # TABLES_DIR,
    NOTEBOOKS_DIR,
    SRC_DIR,
    CONFIG_DIR,

]:
    path.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Market / project defaults
# ----------------------------
TRADING_DAYS = 252

# Take Taiwan 10-Year Government Bond Yield at May 31, 2026.
RISK_FREE_RATE = 0.0164
SEED = 42

# Take about 10 years interval.
START = "2016-01-01"
END = "2026-05-31"

N_PORTFOLIOS = 10_000
TARGET_RETURN = 0.12


DEFAULT_TICKERS: Dict[str, str] = {
    "TW_0050": "0050.TW",
    "US_SP500": "SPY",
    "US_NASDAQ": "QQQ",
    "GLOBAL": "VT",
    "EM": "VWO",
    "SMALL_CAP": "IJR",
    "BOND": "BND",
}


DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "TW_0050": (0.05, 0.80),
    "US_SP500": (0.01, 0.80),
    "US_NASDAQ": (0.01, 0.80),
    "GLOBAL": (0.00, 0.30),
    "EM": (0.00, 0.20),
    "SMALL_CAP": (0.00, 0.20),
    "BOND": (0.00, 0.10),
}

LOOKBACK_WINDOWS = [3, 5, 10]

MU_SHRINKAGE = 0.5
COV_SHRINKAGE = 0.2

# ----------------------------
# File naming / cache settings
# ----------------------------
USE_YFINANCE_CACHE = True
CACHE_AS_CSV = True

# %%
"""Project 3 core engine: Robust Portfolio Optimization."""


from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize


from config.settings import (
    CACHE_DIR,
    START,
    END,
    TRADING_DAYS,
    RISK_FREE_RATE,
    SEED,
    N_PORTFOLIOS,
    TARGET_RETURN,
    DEFAULT_TICKERS,
    DEFAULT_BOUNDS,
    LOOKBACK_WINDOWS,

    MU_SHRINKAGE,
    COV_SHRINKAGE,
)


# CACHE_DIR = Path("data/raw/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_ticker_name(ticker: str) -> str:
    return (
        ticker.replace("^", "")
        .replace(".", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def _build_cache_path(
    ticker: str,
    start: str,
    end: str,
    prefix: str = "price_cache",
) -> Path:
    safe_ticker = _safe_ticker_name(ticker)
    return CACHE_DIR / f"{prefix}_{safe_ticker}_{start}_{end}.csv"


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if x])
            for col in df.columns
        ]
    return df


def _read_cached_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_price_data(
    tickers: dict[str, str],
    start: str,
    end: Optional[str] = None,
    refresh: bool = False,
    use_cache: bool = True,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Load adjusted close price data for multiple assets.

    Priority:
    1) read local cache
    2) download from Yahoo Finance
    3) save cache for next time

    Returns
    -------
    pd.DataFrame
        Index: date
        Columns: asset names
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    all_prices = []

    for asset_name, ticker in tickers.items():
        cache_path = _build_cache_path(
            ticker=ticker,
            start=start,
            end=end,
            prefix="price_cache",
        )

        if use_cache and cache_path.exists() and not refresh:
            print(f"Loading {asset_name} from cache: {cache_path.name}")
            df = _read_cached_csv(cache_path)
        else:
            print(f"Downloading {asset_name} ({ticker}) from Yahoo Finance...")

            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=auto_adjust,
                progress=False,
                threads=False,
            )

            if raw is None or raw.empty:
                if use_cache and cache_path.exists():
                    print(
                        f"Download failed. Falling back to cache: {cache_path.name}")
                    df = _read_cached_csv(cache_path)
                else:
                    raise ValueError(
                        f"Download failed for ticker={ticker}, start={start}, end={end}."
                    )
            else:
                raw = _flatten_yfinance_columns(raw).reset_index()

                rename_map = {}
                for col in raw.columns:
                    col_lower = str(col).strip().lower()

                    if col_lower in {"date", "datetime", "index"}:
                        rename_map[col] = "date"
                    elif col_lower.startswith("close"):
                        rename_map[col] = "close"

                raw = raw.rename(columns=rename_map)

                if "date" not in raw.columns or "close" not in raw.columns:
                    raise ValueError(
                        f"Could not extract date/close from {ticker}. "
                        f"Columns: {list(raw.columns)}"
                    )

                df = raw[["date", "close"]].copy()
                df["date"] = pd.to_datetime(
                    df["date"], errors="coerce").dt.normalize()
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df = df.dropna(subset=["date", "close"]).sort_values("date")

                if use_cache:
                    df.to_csv(cache_path, index=False)
                    print(f"Saved cache: {cache_path.name}")

        df = df[["date", "close"]].copy()
        df = df.rename(columns={"close": asset_name})
        all_prices.append(df)

    price = all_prices[0]

    for df in all_prices[1:]:
        price = price.merge(df, on="date", how="outer")

    price = (
        price.sort_values("date")
        .set_index("date")
        .ffill()
        .dropna()
    )

    return price


def compute_returns(price: pd.DataFrame) -> pd.DataFrame:
    return price.pct_change().dropna()


def estimate_mu(
    returns: pd.DataFrame,
    shrink: float = MU_SHRINKAGE,
) -> pd.Series:
    raw_mu = returns.mean() * TRADING_DAYS
    grand_mean = raw_mu.mean()
    return shrink * grand_mean + (1 - shrink) * raw_mu


def estimate_cov(
    returns: pd.DataFrame,
    shrink: float = COV_SHRINKAGE,
) -> pd.DataFrame:
    # Take the diagonal of covariance matrix "raw_cov", and then
    # put it to a diagnal matrix.
    raw_cov = returns.cov() * TRADING_DAYS

    diag_cov = pd.DataFrame(
        np.diag(np.diag(raw_cov)),
        index=raw_cov.index,
        columns=raw_cov.columns,
    )

    return (1 - shrink) * raw_cov + shrink * diag_cov


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    return float(weights @ mu)


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(weights.T @ cov @ weights))


def portfolio_sharpe(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = RISK_FREE_RATE,
) -> float:
    vol = portfolio_volatility(weights, cov)
    if vol <= 0:
        return np.nan
    return (portfolio_return(weights, mu) - rf) / vol


def backtest_portfolio(price: pd.DataFrame, weights: Sequence[float]) -> pd.Series:
    port_ret = compute_returns(price) @ np.asarray(weights)
    nav = (1 + port_ret).cumprod()  # (1 + p) cumulative product
    nav.name = "NAV"
    return nav


def compute_drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def compute_portfolio_metrics(
    price: pd.DataFrame,
    weights: Sequence[float],
    rf: float = RISK_FREE_RATE,
) -> Dict[str, float]:
    """
    「整個回測期間」的風險報酬表現做摘要，不是僅對單年度
    """

    nav = backtest_portfolio(price, weights)
    daily_ret = nav.pct_change().dropna()
    ann_return = daily_ret.mean() * TRADING_DAYS
    ann_vol = daily_ret.std() * np.sqrt(TRADING_DAYS)

    downside_diff = np.minimum(daily_ret - rf / TRADING_DAYS, 0)
    # Compare two arrays and return a new array containing the element-wise minima.
    downside_vol = np.sqrt(np.mean(downside_diff**2)) * np.sqrt(TRADING_DAYS)

    # = min(non-positive) = max(|non-positive|)
    max_drawdown = compute_drawdown(nav).min()

    return {
        "return": ann_return,
        "volatility": ann_vol,
        "downside_volatility": downside_vol,
        "max_drawdown": max_drawdown,
        "sharpe": (ann_return - rf) / ann_vol if ann_vol > 0 else np.nan,
        # Sortino / Calmar 也是對「整個回測期間」的風險報酬表現做摘要，不是單年度
        "sortino": (ann_return - rf) / downside_vol if downside_vol > 0 else np.nan,
        "calmar": ann_return / abs(max_drawdown) if max_drawdown < 0 else np.nan,
    }


def generate_candidate_weights(
    n_assets: int,
    n_portfolios: int = N_PORTFOLIOS,
    seed: int = SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(n_assets), size=n_portfolios)


def filter_weights_by_bounds(
    weights: np.ndarray,
    assets: Sequence[str],
    bounds: Mapping[str, Tuple[float, float]],
) -> np.ndarray:
    mask = np.ones(len(weights), dtype=bool)
    for i, asset in enumerate(assets):
        lo, hi = bounds[asset]
        mask &= weights[:, i] >= lo
        mask &= weights[:, i] <= hi
    return weights[mask]


def compute_candidate_metrics(
    weights: np.ndarray,
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    mu_arr = mu.values
    cov_arr = cov.values
    ann_returns = weights @ mu_arr
    ann_vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_arr, weights))
    return pd.DataFrame(
        {
            "return": ann_returns,
            "volatility": ann_vols,
            "sharpe": (ann_returns - rf) / ann_vols,
        }
    )


def optimize_max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    bounds: Sequence[Tuple[float, float]],
    rf: float = RISK_FREE_RATE,
) -> np.ndarray:
    n_assets = len(mu)
    x0 = np.ones(n_assets) / n_assets

    res = minimize(
        lambda w: -portfolio_sharpe(w, mu.values, cov.values, rf=rf),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        options={"maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def optimize_min_variance(
    cov: pd.DataFrame,
    bounds: Sequence[Tuple[float, float]],
) -> np.ndarray:
    n_assets = len(cov)
    x0 = np.ones(n_assets) / n_assets

    res = minimize(
        lambda w: portfolio_volatility(w, cov.values),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        options={"maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def optimize_min_vol_with_return_floor(
    mu: pd.Series,
    cov: pd.DataFrame,
    bounds: Sequence[Tuple[float, float]],
    min_return: float = TARGET_RETURN,
) -> np.ndarray:
    """
    Find the minimum-volatility portfolio
    subject to a minimum annualized return.
    """
    n_assets = len(mu)
    x0 = np.ones(n_assets) / n_assets

    res = minimize(
        lambda w: portfolio_volatility(w, cov.values),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {
                "type": "ineq",
                "fun": lambda w: portfolio_return(w, mu.values) - min_return,
            },
        ],
        options={"maxiter": 1000},
    )

    if not res.success:
        raise RuntimeError(res.message)

    return res.x


def make_weight_series(weights: Sequence[float], assets: Sequence[str]) -> pd.Series:
    return pd.Series(weights, index=assets, name="weight")


def build_baseline_weights(assets: Sequence[str]) -> Dict[str, pd.Series]:
    assets = list(assets)
    stock_assets = [a for a in assets if a != "BOND"]

    # Equal weight across all assets
    equal = pd.Series(1 / len(assets), index=assets)

    # Equal weight across stocks only, zero for bonds
    equity_only_equal_weight = pd.Series(0.0, index=assets)
    equity_only_equal_weight[stock_assets] = 1 / len(stock_assets)

    # 80% stocks, 20% bonds (if bond asset exists)
    stock_80_bond_20 = pd.Series(0.0, index=assets)
    stock_80_bond_20[stock_assets] = 0.80 / len(stock_assets)
    if "BOND" in assets:
        stock_80_bond_20["BOND"] = 0.20

    # 60% stocks, 40% bonds (if bond asset exists)
    stock_60_bond_40 = pd.Series(0.0, index=assets)
    stock_60_bond_40[stock_assets] = 0.60 / len(stock_assets)
    if "BOND" in assets:
        stock_60_bond_40["BOND"] = 0.40

    # Single asset portfolios (each asset gets 100%)
    tw0050_only = pd.Series(0.0, index=assets)
    us_sp500_only = pd.Series(0.0, index=assets)
    us_nasdaq_only = pd.Series(0.0, index=assets)
    global_only = pd.Series(0.0, index=assets)
    em_only = pd.Series(0.0, index=assets)
    small_cap_only = pd.Series(0.0, index=assets)

    if "TW_0050" in assets:
        tw0050_only["TW_0050"] = 1.0
    if "US_SP500" in assets:
        us_sp500_only["US_SP500"] = 1.0
    if "US_NASDAQ" in assets:
        us_nasdaq_only["US_NASDAQ"] = 1.0
    if "GLOBAL" in assets:
        global_only["GLOBAL"] = 1.0
    if "EM" in assets:
        em_only["EM"] = 1.0
    if "SMALL_CAP" in assets:
        small_cap_only["SMALL_CAP"] = 1.0

    return {

        "TW_0050 Only": tw0050_only,
        "US_SP500 Only": us_sp500_only,
        "US_NASDAQ Only": us_nasdaq_only,
        "GLOBAL Only": global_only,
        "EM Only": em_only,
        "SMALL_CAP Only": small_cap_only,

        "Equal Weight": equal,
        "Equity Only Equal Weight": equity_only_equal_weight,

        "80/20 Stock Bond": stock_80_bond_20,
        "60/40 Stock Bond": stock_60_bond_40,
    }


def summarize_portfolios(
    price: pd.DataFrame,
    portfolios: Mapping[str, pd.Series],
    rf: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    rows = []
    for name, weights in portfolios.items():
        rows.append(
            {"portfolio": name, **compute_portfolio_metrics(price, weights, rf)})
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


def make_weight_table(portfolios: Mapping[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(portfolios).fillna(0)


def save_or_show(path: Optional[Path] = None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def build_efficient_frontier(
    mu,
    cov,
    bounds,
    candidate_metrics,
    n_points=500,
    rf: float = RISK_FREE_RATE,
):
    target_returns = np.linspace(
        candidate_metrics["return"].quantile(0.001),
        candidate_metrics["return"].quantile(0.999),
        n_points,
    )

    frontier = []

    for target_return in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {
                "type": "eq",
                "fun": lambda w, tr=target_return: portfolio_return(w, mu.values) - tr,
            },
        ]

        x0 = np.ones(len(mu)) / len(mu)

        res = minimize(
            lambda w: portfolio_volatility(w, cov.values),
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        if res.success:
            vol = portfolio_volatility(res.x, cov.values)
            ret = portfolio_return(res.x, mu.values)
            sharpe = (ret - rf) / vol

            frontier.append({
                "return": ret,
                "volatility": vol,
                "sharpe": sharpe,
            })

    frontier = pd.DataFrame(frontier)

    if frontier.empty:
        return frontier

    min_vol_idx = frontier["volatility"].idxmin()
    min_vol_return = frontier.loc[min_vol_idx, "return"]

    frontier = (
        frontier[frontier["return"] >= min_vol_return]
        .sort_values("volatility")
        .reset_index(drop=True)
    )

    return frontier


def plot_efficient_frontier(
    candidate_metrics: pd.DataFrame,
    highlighted: Mapping[str, Tuple[float, float]],
    frontier: Optional[pd.DataFrame] = None,
    path: Optional[Path] = None,
    min_return: float = None,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(
        candidate_metrics["volatility"],
        candidate_metrics["return"],
        c=candidate_metrics["sharpe"],
        s=8,
        alpha=0.5,
    )
    plt.colorbar(label="Sharpe Ratio")

    for name, (vol, ret) in highlighted.items():
        # plt.scatter(vol, ret, s=180, marker="*", label=name)
        # halo
        plt.scatter(vol, ret,
                    marker="*",
                    s=700,
                    c="white",
                    zorder=9,
                    )

        # star
        if (min_return is not None) and (name == "Target Return"):
            label = name + f"({min_return:.2%})"
        else:
            label = name

        plt.scatter(vol, ret,
                    marker="*",
                    s=400,
                    edgecolors="black",
                    linewidths=1.5,
                    label=label,
                    zorder=10,
                    alpha=0.6,
                    )

    if frontier is not None and not frontier.empty:
        plt.plot(frontier["volatility"], frontier["return"],
                 linestyle="--", lw=2.5, label="Efficient Frontier", zorder=8)

    plt.title("Efficient Frontier")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.legend()
    plt.grid(True)
    save_or_show(path)


def plot_weight_bar(
    weights: pd.Series,
    title: str = "Portfolio Weights",
    path: Optional[Path] = None,
) -> None:

    weights = weights.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))

    weights.plot(
        kind="bar",
        ax=ax,
        width=0.8,
    )

    # 標示權重
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f%%",
            padding=3,
            fontsize=9,
        )

    # 因為 bar_label 使用原始數值，所以乘 100
    for txt, value in zip(ax.texts, weights):
        txt.set_text(f"{value*100:.2f}%")

    ax.set_title(title)
    ax.set_ylabel("Weight")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    # 留一些上方空間避免文字被切掉
    ax.set_ylim(0, weights.max() * 1.15)

    plt.tight_layout()

    save_or_show(path)


def plot_nav_comparison(
    price: pd.DataFrame,
    portfolios: Mapping[str, pd.Series],
    path: Optional[Path] = None,
) -> None:
    plt.figure(figsize=(10, 5))
    for name, weights in portfolios.items():
        nav = backtest_portfolio(price, weights.values)
        plt.plot(nav.index, nav, label=name)
    plt.title("NAV Comparison")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(True)
    save_or_show(path)


def plot_drawdown_comparison(
    price: pd.DataFrame,
    portfolios: Mapping[str, pd.Series],
    path: Optional[Path] = None,
) -> None:
    """
    Plot portfolio drawdown comparison with:
    1. Drawdown time series
    2. Maximum drawdown summary bar chart
    """
    drawdowns = {}
    max_drawdowns = {}

    for name, weights in portfolios.items():
        nav = backtest_portfolio(price, weights.values)
        dd = compute_drawdown(nav)

        drawdowns[name] = dd
        max_drawdowns[name] = dd.min()

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=False,
        constrained_layout=True,
    )

    cmap = plt.get_cmap("tab10")
    colors = {
        name: cmap(i % 10)
        for i, name in enumerate(drawdowns.keys())
    }

    # 1) Drawdown time series
    for name, dd in drawdowns.items():
        ax1.plot(
            dd.index,
            dd.values,
            label=name,
            color=colors[name],
            linewidth=1.8,
            alpha=0.85,
        )

    min_dd = min(dd.min() for dd in drawdowns.values())
    ymin = min_dd * 1.3

    max_dd = max(dd.max() for dd in drawdowns.values())
    # ymax = compute_returns(price=price).cummax().max() * 1.5
    ymax = 0.01

    ax1.set_ylim([ymin, ymax])
    ax1.axhline(0, color="black", linewidth=1)
    ax1.set_title("Drawdown Comparison")
    ax1.set_ylabel("Drawdown")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.grid(True, alpha=0.35)
    ax1.legend(
        loc="lower left",
        ncol=2,
        frameon=True,
        fontsize=9,
    )

    # 2) Max drawdown bar chart
    max_dd_series = (
        pd.Series(max_drawdowns)
        .sort_values()
    )

    ax2.bar(
        max_dd_series.index,
        max_dd_series.values,
        color=[colors[name] for name in max_dd_series.index],
        alpha=0.85,
    )

    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_ylabel("Max DD")
    ax2.set_xlabel("Portfolio")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.grid(True, axis="y", alpha=0.35)

    for i, value in enumerate(max_dd_series.values):
        ax2.text(
            i,
            value,
            f"{value:.2%}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    ax2.tick_params(axis="x", rotation=20)

    save_or_show(path)


def run_lookback_optimization(
    price: pd.DataFrame,
    bounds_config: Mapping[str, Tuple[float, float]],
    lookback_windows: Sequence[int] = LOOKBACK_WINDOWS,
    rf: float = RISK_FREE_RATE,
    min_return: float = TARGET_RETURN,
) -> Dict[str, pd.DataFrame]:

    assets = price.columns.tolist()
    bounds = [bounds_config[a] for a in assets]

    max_sharpe_results = {}
    min_variance_results = {}
    target_return_results = {}

    end_date = price.index.max()

    for years in lookback_windows:
        start_date = end_date - pd.DateOffset(years=years)

        price_lb = price.loc[start_date:end_date].copy()
        returns_lb = compute_returns(price_lb)

        mu_lb = estimate_mu(returns_lb)
        cov_lb = estimate_cov(returns_lb)

        max_sharpe_w = make_weight_series(
            optimize_max_sharpe(mu_lb, cov_lb, bounds, rf),
            assets,
        )

        min_variance_w = make_weight_series(
            optimize_min_variance(cov_lb, bounds),
            assets,
        )

        target_return_w = make_weight_series(
            optimize_min_vol_with_return_floor(
                mu=mu_lb,
                cov=cov_lb,
                bounds=bounds,
                min_return=min_return,
            ), assets
        )

        max_sharpe_results[f"{years}Y"] = max_sharpe_w
        min_variance_results[f"{years}Y"] = min_variance_w
        target_return_results[f"{years}Y"] = target_return_w

    return {
        "max_sharpe_weights": pd.DataFrame(max_sharpe_results),
        "min_variance_weights": pd.DataFrame(min_variance_results),
        "target_return_weights": pd.DataFrame(target_return_results),
    }


def plot_weight_heatmap(
    weight_df: pd.DataFrame,
    title: str,
    save_path: Optional[Path] = None,
) -> None:
    plt.figure(figsize=(8, 5))

    plt.imshow(weight_df.values, aspect="auto", cmap="YlGnBu")
    plt.colorbar(label="Weight")

    plt.xticks(
        ticks=np.arange(len(weight_df.columns)),
        labels=weight_df.columns,
    )
    plt.yticks(
        ticks=np.arange(len(weight_df.index)),
        labels=weight_df.index,
    )

    for i in range(weight_df.shape[0]):
        for j in range(weight_df.shape[1]):
            plt.text(
                j,
                i,
                f"{weight_df.iloc[i, j]:.2%}",
                ha="center",
                va="center",
            )

    plt.title(title)
    plt.xlabel("Lookback Window")
    plt.ylabel("Asset")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# def run_pipeline(
#     tickers: Mapping[str, str] = DEFAULT_TICKERS,
#     bounds_config: Mapping[str, Tuple[float, float]] = DEFAULT_BOUNDS,
#     start: str = START,
#     end: str = END,
#     n_portfolios: int = N_PORTFOLIOS,
#     rf: float = RISK_FREE_RATE,
#     min_return: float = 0.14,
#     # output_dir: Optional[str | Path] = None,
# ) -> Dict[str, object]:
#     price = load_price_data(tickers=tickers, start=start,
#                             end=end, refresh=False, use_cache=True, auto_adjust=True)
#     returns = compute_returns(price)
#     assets = price.columns.tolist()

#     mu = estimate_mu(returns)
#     cov = estimate_cov(returns)
#     bounds = [bounds_config[a] for a in assets]

#     candidate_weights = generate_candidate_weights(len(assets), n_portfolios)
#     valid_weights = filter_weights_by_bounds(
#         candidate_weights, assets, bounds_config)
#     candidate_metrics = compute_candidate_metrics(valid_weights, mu, cov)

#     frontier = build_efficient_frontier(
#         mu, cov, bounds, candidate_metrics, n_points=500, rf)

#     portfolios = build_baseline_weights(assets)
#     portfolios["Max Sharpe"] = make_weight_series(
#         optimize_max_sharpe(mu, cov, bounds, rf), assets
#     )
#     portfolios["Min Variance"] = make_weight_series(
#         optimize_min_variance(cov, bounds), assets
#     )

#     portfolios["Target Return"] = make_weight_series(
#         optimize_min_vol_with_return_floor(mu, cov, bounds, min_return), assets
#     )

#     robustness_results = run_lookback_optimization(
#         price=price,
#         bounds_config=bounds_config,
#         lookback_windows=LOOKBACK_WINDOWS,
#         rf=RISK_FREE_RATE,
#         min_return=TARGET_RETURN,
#     )

#     metrics_table = summarize_portfolios(price, portfolios)
#     weight_table = make_weight_table(portfolios)

#     # fig_dir = Path(output_dir) if output_dir is not None else None

#     # if fig_dir is not None:
#     #     fig_dir.mkdir(parents=True, exist_ok=True)

#     highlighted = {
#         name: (
#             portfolio_volatility(w.values, cov.values),
#             portfolio_return(w.values, mu.values),
#         )
#         for name, w in portfolios.items()
#         if name in {
#             "Max Sharpe",
#             "Min Variance",
#             "Equal Weight",
#             "Target Return",
#             # "Equity Only Equal Weight",
#             "80/20 Stock Bond",
#             "60/40 Stock Bond"}
#     }
#     # print("highlighted:", highlighted)
#     # print("candidate_metrics head:\n", candidate_metrics.head())

#     print("Results have already computed!")

#     return {
#         "price": price,
#         "returns": returns,
#         "mu": mu,
#         "cov": cov,
#         "frontier": frontier,
#         "candidate_metrics": candidate_metrics,
#         "candidate_weights": valid_weights,
#         "portfolios": portfolios,
#         "metrics_table": metrics_table,
#         "weight_table": weight_table,
#         "robustness_results": robustness_results,
#         "highlighted": highlighted
#     }


def run_pipeline(
    tickers: Mapping[str, str] = DEFAULT_TICKERS,
    bounds_config: Mapping[str, Tuple[float, float]] = DEFAULT_BOUNDS,
    start: str = None,
    end: str = None,
    n_portfolios: int = N_PORTFOLIOS,
    rf: float = RISK_FREE_RATE,
    min_return: float = TARGET_RETURN,
) -> Dict[str, object]:

    from config.settings import START, END

    if start is None:
        start = START

    if end is None:
        end = END

    price = load_price_data(
        tickers=tickers,
        start=start,
        end=end,
        refresh=False,
        use_cache=True,
        auto_adjust=True,
    )

    returns = compute_returns(price)
    assets = price.columns.tolist()

    mu = estimate_mu(returns)
    cov = estimate_cov(returns)
    bounds = [bounds_config[a] for a in assets]

    candidate_weights = generate_candidate_weights(
        len(assets),
        n_portfolios=n_portfolios,
    )

    valid_weights = filter_weights_by_bounds(
        candidate_weights,
        assets,
        bounds_config,
    )

    candidate_metrics = compute_candidate_metrics(
        valid_weights,
        mu,
        cov,
        rf=rf,
    )

    frontier = build_efficient_frontier(
        mu=mu,
        cov=cov,
        bounds=bounds,
        candidate_metrics=candidate_metrics,
        n_points=500,
        rf=rf,
    )

    portfolios = build_baseline_weights(assets)

    portfolios["Max Sharpe"] = make_weight_series(
        optimize_max_sharpe(mu, cov, bounds, rf),
        assets,
    )

    portfolios["Min Variance"] = make_weight_series(
        optimize_min_variance(cov, bounds),
        assets,
    )

    portfolios["Target Return"] = make_weight_series(
        optimize_min_vol_with_return_floor(
            mu=mu,
            cov=cov,
            bounds=bounds,
            min_return=min_return,
        ),
        assets,
    )

    robustness_results = run_lookback_optimization(
        price=price,
        bounds_config=bounds_config,
        lookback_windows=LOOKBACK_WINDOWS,
        rf=rf,
        min_return=min_return,
    )

    metrics_table = summarize_portfolios(price, portfolios, rf=rf)
    weight_table = make_weight_table(portfolios)

    highlighted = {
        name: (
            portfolio_volatility(w.values, cov.values),
            portfolio_return(w.values, mu.values),
        )
        for name, w in portfolios.items()
        if name in {
            "Max Sharpe",
            "Min Variance",
            "Equal Weight",
            "Target Return",
            "80/20 Stock Bond",
            "60/40 Stock Bond",
        }
    }

    return {
        "price": price,
        "returns": returns,
        "mu": mu,
        "cov": cov,
        "frontier": frontier,
        "candidate_metrics": candidate_metrics,
        "candidate_weights": valid_weights,
        "portfolios": portfolios,
        "metrics_table": metrics_table,
        "weight_table": weight_table,
        "robustness_results": robustness_results,
        "highlighted": highlighted,
    }

# # %%
# if __name__ == "__main__":
#     results = run_pipeline(
#         # tickers = DEFAULT_TICKERS,
#         # bounds_config = DEFAULT_BOUNDS,
#         # start = START,
#         # end = END,
#     )

#     # print("\nPerformance Metrics")
#     # print(results["metrics_table"].to_string(index=False))
#     # print("\nPortfolio Weights")
#     # print(results["weight_table"].round(4).to_string())

#     # print("\nRobustness Weights")
#     # print(results["robustness_results"]["max_sharpe_weights"])
#     # print(results["robustness_results"]["min_variance_weights"])

#     fig_dir = None
#     # output_dir="reports/figures"
#     # fig_dir = Path(output_dir) if output_dir is not None else None

#     # if fig_dir is not None:
#     #     fig_dir.mkdir(parents=True, exist_ok=True)

#     plot_efficient_frontier(
#         results["candidate_metrics"],
#         results["highlighted"],
#         results["frontier"],
#         None if fig_dir is None else fig_dir / "efficient_frontier.png",
#         TARGET_RETURN,

#     )
#     plot_weight_bar(
#         results["portfolios"]["Max Sharpe"],
#         "Max Sharpe Portfolio Weights",
#         None if fig_dir is None else fig_dir / "max_sharpe_weights.png",
#     )

#     plot_weight_bar(
#         results["portfolios"]["Min Variance"],
#         "Min Variance Portfolio Weights",
#         None if fig_dir is None else fig_dir / "min_variance_weights.png",
#     )

#     plot_weight_bar(
#         results["portfolios"]["Target Return"],
#         "Target Return Portfolio Weights",
#         None if fig_dir is None else fig_dir / "target_return_weights.png",
#     )

#     nav_and_drawdown_comparison_portfolios = {
#         name: w for name, w in results["portfolios"].items()
#         if name in {
#             "Max Sharpe",
#             "Min Variance",
#             "Equal Weight",
#             "Target Return"
#             # "Equity Only Equal Weight",
#             "80/20 Stock Bond",
#             "60/40 Stock Bond"}
#     }

#     plot_nav_comparison(
#         results["price"],
#         nav_and_drawdown_comparison_portfolios,
#         None if fig_dir is None else fig_dir / "nav_comparison.png",
#     )
#     plot_drawdown_comparison(
#         results["price"],
#         nav_and_drawdown_comparison_portfolios,
#         None if fig_dir is None else fig_dir / "drawdown_comparison.png",
#     )

#     plot_weight_heatmap(
#         results["robustness_results"]["max_sharpe_weights"],
#         "Robustness Analysis: Maximum Sharpe Portfolio",
#         None if fig_dir is None else fig_dir / "robustness_max_sharpe_heatmap.png",
#     )

#     plot_weight_heatmap(
#         results["robustness_results"]["min_variance_weights"],
#         "Robustness Analysis: Minimum Variance Portfolio",
#         None if fig_dir is None else fig_dir / "robustness_min_variance_heatmap.png",
#     )

#     plot_weight_heatmap(
#         results["robustness_results"]["target_return_weights"],
#         "Robustness Analysis: Target Return Portfolio",
#         None if fig_dir is None else fig_dir / "robustness_target_return_heatmap.png",
#     )

# %%

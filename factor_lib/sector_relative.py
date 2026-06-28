"""
Sector-Relative Factor Computation & Pre-Flight Safety Checks.

V1.5 Core Infrastructure — addresses the root cause identified in the V1→V2 audit:
  - ProfitGrowth cross-panel rank r = 0.001 (factor meaning destroyed by universe expansion)
  - BP silently deleted by GS (no error, just missing signal)
  - Small-sample industries producing NaN factors that vanish silently

Design principles:
  1. Sector-relative z-score eliminates universe-dependence of factor ranks
  2. Small-sample fallback prevents NaN propagation in thin industries
  3. Pre-flight hard assertion catches zero-variance factors BEFORE training
  4. All critical failures raise ValueError — no silent data loss

Usage:
  from factor_lib.sector_relative import (
      compute_sector_relative_factor_safe,
      preflight_factor_sanity_check,
  )
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Patch 1: Safe Sector-Relative Z-Score with Small-Sample Fallback
# ═══════════════════════════════════════════════════════════════

def compute_sector_relative_factor_safe(
    df: pd.DataFrame,
    factor_col: str,
    industry_col: str = "sw_l1",
    date_col: str = "date",
    min_stocks: int = 3,
) -> pd.Series:
    """
    Safe sector-relative z-score with automatic small-sample fallback.

    For each (date, industry) group, computes z-score of factor_col within that
    group.  When a group has fewer than `min_stocks` valid observations, falls
    back to the full cross-sectional (all-industry) z-score for that date.

    Additional safety:
      - std=0 or std=NaN within a group → treated as std=1.0 (centering only,
        no scaling), preventing division-by-zero NaN propagation.
      - NaN values in factor_col are preserved through the computation.

    Parameters
    ----------
    df : pd.DataFrame
        Input panel. Must contain date_col, industry_col, and factor_col.
    factor_col : str
        Column name of the raw factor to sector-relative-ize.
    industry_col : str
        Column name of the industry classification (e.g. Shenwan L1).
    date_col : str
        Column name of the date.
    min_stocks : int
        Minimum number of valid (non-NaN) stocks in a group to use
        sector-relative z-score.  Below this, fall back to market-wide z-score.

    Returns
    -------
    pd.Series
        Sector-relative z-score with the same index as df.
        Named "SR_{factor_col}".

    Raises
    ------
    ValueError
        If factor_col or industry_col not in df.columns.
    """
    if factor_col not in df.columns:
        raise ValueError(
            f"factor_col '{factor_col}' not found in DataFrame columns: {df.columns.tolist()}"
        )
    if industry_col not in df.columns:
        raise ValueError(
            f"industry_col '{industry_col}' not found in DataFrame columns: {df.columns.tolist()}"
        )
    if date_col not in df.columns:
        raise ValueError(
            f"date_col '{date_col}' not found in DataFrame columns: {df.columns.tolist()}"
        )

    # Work on a copy to avoid fragmentation
    work = df[[date_col, industry_col, factor_col]].copy()
    raw_values = work[factor_col].values.astype(np.float64)

    # ── Group-level mean, std, and count ──
    grp = work.groupby([date_col, industry_col])[factor_col]
    grp_mean = grp.transform("mean")
    grp_std = grp.transform("std")
    grp_count = grp.transform("count")  # count of non-NaN within group

    # Safety micro-perturbation: replace NaN or 0 std with 1.0
    # This preserves the centering (mean removal) while skipping scaling.
    # Using 1.0 (not 0 or epsilon) preserves the original scale.
    grp_std_safe = grp_std.fillna(1.0).replace(0.0, 1.0)

    # ── Market-wide mean and std as fallback ──
    mkt = work.groupby(date_col)[factor_col]
    mkt_mean = mkt.transform("mean")
    mkt_std = mkt.transform("std").fillna(1.0).replace(0.0, 1.0)

    # ── Compute both z-scores ──
    sr_zscore = (raw_values - grp_mean.values) / grp_std_safe.values
    mkt_zscore = (raw_values - mkt_mean.values) / mkt_std.values

    # ── Conditional merge: sector-relative where group has enough stocks ──
    result_values = np.where(grp_count.values >= min_stocks, sr_zscore, mkt_zscore)

    result = pd.Series(result_values, index=df.index, name=f"SR_{factor_col}")

    n_fallback = int((grp_count.values < min_stocks).sum())
    if n_fallback > 0:
        logger.debug(
            "  SR_%s: %d/%d rows fell back to market-wide z-score (group count < %d)",
            factor_col, n_fallback, len(df), min_stocks,
        )

    return result


def compute_all_sector_relative_factors(
    df: pd.DataFrame,
    factor_cols: list[str],
    industry_col: str = "sw_l1",
    date_col: str = "date",
    min_stocks: int = 3,
) -> pd.DataFrame:
    """
    Convenience: apply compute_sector_relative_factor_safe to multiple columns.

    Parameters
    ----------
    df : pd.DataFrame
    factor_cols : list[str]
        Raw factor column names to sector-relative-ize.
    industry_col, date_col, min_stocks : see compute_sector_relative_factor_safe

    Returns
    -------
    pd.DataFrame
        Original df with additional SR_{col} columns appended.
    """
    result = df.copy()
    for col in factor_cols:
        try:
            result[f"SR_{col}"] = compute_sector_relative_factor_safe(
                result, col, industry_col, date_col, min_stocks,
            )
        except Exception as e:
            logger.error("Failed to compute SR_%s: %s", col, e)
            raise
    return result


# ═══════════════════════════════════════════════════════════
# Patch 2: Pre-Flight Factor Sanity Check (Hard Block)
# ═══════════════════════════════════════════════════════════

def preflight_factor_sanity_check(
    df: pd.DataFrame,
    required_factors: list[str],
    date_col: str = "date",
    threshold: float = 1e-5,
    min_valid_dates_ratio: float = 0.50,
) -> dict[str, dict]:
    """
    Pre-flight factor sanity check — HARD BLOCK before training.

    For every factor in `required_factors`:
      1. Compute cross-sectional standard deviation per date
      2. Flag dates where std < threshold (factor has been "silently killed")
      3. If ANY factor fails, raise ValueError with diagnostic information

    This prevents the "BP was silently deleted by GS" class of bugs from
    ever reaching model training.  Call this:
      - After panel construction (Phase 0.3)
      - At the top of every model training script (Phase 1)

    Parameters
    ----------
    df : pd.DataFrame
        Factor panel.
    required_factors : list[str]
        Factor column names that MUST have cross-sectional variance.
    date_col : str
        Date column name.
    threshold : float
        Minimum acceptable cross-sectional standard deviation.
        For z-scored factors, σ < 1e-5 means the factor is effectively constant.
    min_valid_dates_ratio : float
        Minimum fraction of dates where the factor must have σ > threshold.
        Default 0.50 = at least 50% of dates must pass.

    Returns
    -------
    dict[str, dict]
        Per-factor diagnostics: {factor: {"n_zero_var_dates": int, "zero_var_dates": [...],
                                           "min_std": float, "max_std": float, "passed": bool}}

    Raises
    ------
    ValueError
        If ANY required factor fails the variance check, with full diagnostic
        information about which factors failed, on which dates, and likely causes.
    """
    diagnostics: dict[str, dict] = {}
    failed_factors: list[tuple[str, int, str]] = []  # (factor, n_bad, first_date)

    for factor in required_factors:
        if factor not in df.columns:
            raise ValueError(
                f"Required factor '{factor}' not found in DataFrame. "
                f"Available columns: {df.columns.tolist()}"
            )

        # Cross-sectional std per date
        daily_stds = df.groupby(date_col)[factor].std()

        # Dates where std ≈ 0
        zero_var_mask = daily_stds < threshold
        zero_var_dates = daily_stds[zero_var_mask].index.tolist()
        n_zero = len(zero_var_dates)
        n_total = len(daily_stds)
        passed_ratio = 1.0 - n_zero / max(n_total, 1)
        passed = passed_ratio >= min_valid_dates_ratio

        diagnostics[factor] = {
            "n_zero_var_dates": n_zero,
            "zero_var_dates": [str(d) for d in zero_var_dates[:5]],  # first 5
            "total_dates": n_total,
            "min_std": float(daily_stds.min()) if len(daily_stds) > 0 else np.nan,
            "max_std": float(daily_stds.max()) if len(daily_stds) > 0 else np.nan,
            "mean_std": float(daily_stds.mean()) if len(daily_stds) > 0 else np.nan,
            "passed_ratio": float(passed_ratio),
            "passed": passed,
        }

        if not passed:
            first_date_str = str(zero_var_dates[0]) if zero_var_dates else "N/A"
            failed_factors.append((factor, n_zero, first_date_str))

    # ── Hard block if any factor failed ──
    if failed_factors:
        msg_lines = [
            "=" * 72,
            "PRE-FLIGHT FACTOR SANITY CHECK — FAILED",
            "=" * 72,
            "",
            f"The following {len(failed_factors)} factor(s) have near-zero cross-sectional",
            f"standard deviation on too many dates (threshold: σ < {threshold}):",
            "",
        ]
        for factor, n_zero, first_date in failed_factors:
            diag = diagnostics[factor]
            msg_lines.extend([
                f"  Factor: '{factor}'",
                f"    Zero-variance dates: {n_zero}/{diag['total_dates']} "
                f"({100*n_zero/max(diag['total_dates'],1):.1f}%)",
                f"    First occurrence: {first_date}",
                f"    Min σ: {diag['min_std']:.2e}  |  Max σ: {diag['max_std']:.4f}",
                f"    Mean σ: {diag['mean_std']:.4f}",
                "",
            ])
        msg_lines.extend([
            "Likely causes:",
            "  1. Gram-Schmidt orthogonalization eliminated this factor due to",
            "     collinearity with a higher-priority factor (e.g. EP consumed BP).",
            "  2. Data cleaning code contains an in-place overwrite bug.",
            "  3. The factor is genuinely a constant — it should be removed from",
            "     the feature list, not silently passed through.",
            "",
            "Action required:",
            "  - If GS is ON: check which factor orthogonalized this one away.",
            "  - If GS is OFF: audit the data pipeline for column overwrites.",
            "  - Remove the factor from required_factors only if confirmed useless.",
            "=" * 72,
        ])
        raise ValueError("\n".join(msg_lines))

    # ── All clear ──
    n_factors = len(required_factors)
    logger.info(
        "Pre-flight sanity check PASSED: %d/%d factors have σ > %s on >%.0f%% of dates",
        n_factors, n_factors, threshold, 100 * min_valid_dates_ratio,
    )
    for factor, diag in diagnostics.items():
        logger.debug(
            "  %s: σ ∈ [%.4f, %.4f], mean=%.4f, zero-var: %d/%d dates",
            factor, diag["min_std"], diag["max_std"], diag["mean_std"],
            diag["n_zero_var_dates"], diag["total_dates"],
        )

    return diagnostics


# ═══════════════════════════════════════════════════════════
# Utility: Shenwan Industry Classification Loader
# ═══════════════════════════════════════════════════════════

_SW_INDUSTRY_CACHE: Optional[pd.DataFrame] = None


def load_shenwan_industry(
    symbols: Optional[list[str]] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load Shenwan Level-1 industry classification for CSI 800 universe.

    Tries multiple data sources in order:
      1. Cached parquet: data/sw_industry.parquet
      2. Baostock industry classification API
      3. Fallback: empty DataFrame (market-wide z-score will be used everywhere)

    Parameters
    ----------
    symbols : list[str], optional
        Specific stock symbols to fetch. If None, fetches all.
    force_refresh : bool
        If True, skip cache and re-fetch.

    Returns
    -------
    pd.DataFrame with columns [symbol, sw_l1]
    """
    global _SW_INDUSTRY_CACHE

    from pathlib import Path

    CACHE_PATH = Path("data/sw_industry.parquet")

    if not force_refresh and _SW_INDUSTRY_CACHE is not None:
        return _SW_INDUSTRY_CACHE.copy()

    if not force_refresh and CACHE_PATH.exists():
        logger.info("Loading Shenwan industry from cache: %s", CACHE_PATH)
        _SW_INDUSTRY_CACHE = pd.read_parquet(CACHE_PATH)
        return _SW_INDUSTRY_CACHE.copy()

    # ── Try Baostock ──
    try:
        import baostock as bs
        logger.info("Fetching Shenwan industry from Baostock...")

        bs.login()
        industry_rows = []

        # Helper: convert 6-digit code to baostock format
        def _to_bs_code(code: str) -> str:
            code = str(code).zfill(6)
            if code.startswith(('6', '9')):
                return f"sh.{code}"
            else:
                return f"sz.{code}"

        # Helper: extract 6-digit symbol from baostock code
        def _from_bs_code(bs_code: str) -> str:
            return bs_code.replace("sh.", "").replace("sz.", "").zfill(6)

        # Baostock industry classification codes
        target_symbols = symbols or []
        if not target_symbols:
            # If no symbols specified, try to get all A-share stocks
            rs = bs.query_stock_basic(code_name="A股")
            if rs.error_code == "0":
                while rs.next():
                    row = rs.get_row_data()
                    # row[0] = baostock code (sh.600000), row[1] = name
                    target_symbols.append(_from_bs_code(row[0]))

        logger.info("Fetching industry for %d symbols...", len(target_symbols))
        n_fetched = 0
        for sym in target_symbols:
            try:
                bs_code = _to_bs_code(sym)
                rs = bs.query_stock_industry(bs_code)
                if rs.error_code == "0":
                    while rs.next():
                        row = rs.get_row_data()
                        # Row: [update_date, code, code_name, industry, industry_code]
                        ind_name = row[3] if len(row) > 3 else (row[2] if len(row) > 2 else "未知")
                        # Map to Shenwan L1 categories
                        industry_rows.append({
                            "symbol": sym,
                            "sw_l1": _map_to_sw_l1(ind_name),
                        })
                        n_fetched += 1
                        break  # Take first classification
            except Exception:
                continue

        bs.logout()
        logger.info("Fetched industry for %d/%d symbols", n_fetched, len(target_symbols))

        if industry_rows:
            result = pd.DataFrame(industry_rows)
            result.to_parquet(CACHE_PATH, index=False)
            logger.info("Shenwan industry saved: %d symbols to %s", len(result), CACHE_PATH)
            _SW_INDUSTRY_CACHE = result
            return result.copy()

    except ImportError:
        logger.warning("Baostock not available for industry classification fetch.")
    except Exception as e:
        logger.warning("Failed to fetch Shenwan industry from Baostock: %s", e)

    # ── Fallback: empty ──
    logger.warning(
        "No Shenwan industry data available. "
        "Sector-relative z-score will fall back to market-wide for all stocks. "
        "This is safe but loses industry-specific normalization."
    )
    empty = pd.DataFrame(columns=["symbol", "sw_l1"])
    _SW_INDUSTRY_CACHE = empty
    return empty.copy()


def _map_to_sw_l1(industry_name: str) -> str:
    """
    Map Baostock/CSRC industry name to Shenwan Level-1 category.

    This is an approximate mapping — for production use, obtain
    the official Shenwan classification from a data vendor.
    """
    name = str(industry_name).strip()

    # Direct Shenwan L1 names (pass-through)
    SW_L1_NAMES = {
        "银行", "非银金融", "房地产", "建筑装饰", "建筑材料",
        "有色金属", "钢铁", "基础化工", "石油石化", "煤炭",
        "电力设备", "机械设备", "国防军工", "汽车",
        "电子", "计算机", "通信", "传媒", "互联网",
        "食品饮料", "家用电器", "纺织服装", "轻工制造",
        "医药生物", "美容护理", "社会服务",
        "农林牧渔", "公用事业", "交通运输", "商贸零售",
        "环保", "综合",
    }

    if name in SW_L1_NAMES:
        return name

    # Common CSRC → SW L1 mappings
    MAPPING = {
        "银行": "银行", "银行业": "银行",
        "证券": "非银金融", "保险": "非银金融", "多元金融": "非银金融",
        "房地产开发": "房地产", "房地产": "房地产",
        "建筑施工": "建筑装饰", "建筑": "建筑装饰",
        "有色金属": "有色金属", "钢铁": "钢铁",
        "基础化工": "基础化工", "化学制品": "基础化工", "化工": "基础化工",
        "石油": "石油石化", "石化": "石油石化",
        "煤炭": "煤炭", "采掘": "煤炭",
        "电力设备": "电力设备", "新能源": "电力设备",
        "机械设备": "机械设备", "机械": "机械设备",
        "国防军工": "国防军工", "军工": "国防军工",
        "汽车": "汽车",
        "电子": "电子", "半导体": "电子", "元器件": "电子",
        "计算机": "计算机", "软件": "计算机", "IT": "计算机",
        "通信": "通信", "5G": "通信",
        "传媒": "传媒", "文化传媒": "传媒",
        "食品饮料": "食品饮料", "白酒": "食品饮料", "食品": "食品饮料",
        "家用电器": "家用电器", "家电": "家用电器",
        "纺织服装": "纺织服装", "服装": "纺织服装",
        "轻工制造": "轻工制造", "造纸": "轻工制造",
        "医药": "医药生物", "医药生物": "医药生物", "生物医药": "医药生物",
        "美容": "美容护理", "化妆品": "美容护理", "医美": "美容护理",
        "旅游": "社会服务", "酒店": "社会服务", "餐饮": "社会服务",
        "农业": "农林牧渔", "农林牧渔": "农林牧渔",
        "公用事业": "公用事业", "电力": "公用事业", "水务": "公用事业",
        "交通运输": "交通运输", "物流": "交通运输", "航空": "交通运输",
        "商贸零售": "商贸零售", "零售": "商贸零售", "商业": "商贸零售",
        "环保": "环保",
    }

    for key, value in MAPPING.items():
        if key in name:
            return value

    # Unknown → "综合" (catch-all category)
    return "综合"

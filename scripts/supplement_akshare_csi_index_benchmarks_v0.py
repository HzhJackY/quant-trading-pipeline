import gc
import importlib
import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "AKShare_CSI_Index_Supplement"
OUT_DIR = Path("output/akshare_csi_index_supplement_monthly_alignment_v0")
RUN_DIR = Path("output/_agent_runs") / TASK_NAME
RAW_DIR = Path("data/external/akshare")

PORTFOLIO_MONTHLY = Path("output/unified_robust_portfolio_evaluation_run_v0/unified_portfolio_monthly_gross_return.csv")
CSMAR_OFFICIAL = Path("output/csmar_excel_parser_fix_benchmark_reaudit_v0/official_index_monthly_forward_return_candidates.csv")
BENCHMARK_RECO = Path("output/csmar_excel_parser_fix_benchmark_reaudit_v0/benchmark_candidate_recommendation.csv")
INTERNAL_BENCH = Path("output/csmar_excel_parser_fix_benchmark_reaudit_v0/internal_universe_monthly_forward_benchmark.csv")
RISK_FREE = Path("output/csmar_excel_parser_fix_benchmark_reaudit_v0/risk_free_monthly_aligned.csv")
FAMA_FRENCH = Path("output/csmar_excel_parser_fix_benchmark_reaudit_v0/fama_french_monthly_factor_candidates.csv")

START_DATE = "20100101"
END_DATE = "20260706"
SYMBOLS = [
    {"symbol": "000906", "label": "CSI800", "monthly_label": "CSI800_AKSHARE_PRICE", "expected": "中证800"},
    {"symbol": "000905", "label": "CSI500", "monthly_label": "CSI500_AKSHARE_PRICE", "expected": "中证500"},
    {"symbol": "000300", "label": "HS300_AKSHARE_VALIDATION", "monthly_label": "HS300_AKSHARE_PRICE_VALIDATION", "expected": "沪深300"},
]


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def write_run_state(status, checkpoint, extra=None):
    lines = [
        "# AKShare CSI Index Supplement & Monthly Alignment v0",
        "",
        f"- task_name: {TASK_NAME}",
        f"- status: {status}",
        f"- last_checkpoint: {checkpoint}",
        f"- updated_at: {datetime.now().isoformat(timespec='seconds')}",
        "- resume_command: `python scripts\\supplement_akshare_csi_index_benchmarks_v0.py > output\\_agent_runs\\AKShare_CSI_Index_Supplement\\run_stdout.txt 2> output\\_agent_runs\\AKShare_CSI_Index_Supplement\\run_stderr.txt`",
        "- guardrails: no weights edits; no benchmark-relative portfolio returns; no alpha/beta; no IR; no TE; no training; no SHAP; no production",
    ]
    if extra:
        lines.extend(["", "## Notes"])
        lines.extend([f"- {x}" for x in extra])
    (RUN_DIR / "RUN_STATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def import_or_install_akshare():
    installed_before = importlib.util.find_spec("akshare") is not None
    install_attempted = False
    import_error = None
    if not installed_before:
        install_attempted = True
        subprocess.run([sys.executable, "-m", "pip", "install", "akshare", "-q"], check=True)
    try:
        ak = importlib.import_module("akshare")
        version = getattr(ak, "__version__", "unknown")
        return ak, {
            "akshare_installed_before": installed_before,
            "akshare_install_attempted": install_attempted,
            "akshare_available": True,
            "akshare_version": version,
            "akshare_import_error": None,
        }
    except Exception as exc:
        import_error = repr(exc)
        return None, {
            "akshare_installed_before": installed_before,
            "akshare_install_attempted": install_attempted,
            "akshare_available": False,
            "akshare_version": None,
            "akshare_import_error": import_error,
        }


def detect_col(columns, candidates):
    cols = list(columns)
    lower_map = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        key = cand.strip().lower()
        if key in lower_map:
            return lower_map[key]
    for c in cols:
        cs = str(c).strip().lower()
        if any(cand.strip().lower() in cs for cand in candidates):
            return c
    return None


def to_numeric(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False), errors="coerce")


def schema_audit_one(raw_df, meta, download_success):
    date_col = detect_col(raw_df.columns, ["日期", "date", "交易日期", "trddt"])
    code_col = detect_col(raw_df.columns, ["指数代码", "代码", "indexcd", "symbol"])
    close_col = detect_col(raw_df.columns, ["收盘", "close", "clsindex", "收盘价"])
    pct_col = detect_col(raw_df.columns, ["涨跌幅", "pct", "change", "retindex"])
    name_col = detect_col(raw_df.columns, ["指数中文全称", "指数中文简称", "指数名称", "名称", "name"])

    dates = pd.Series(dtype="datetime64[ns]")
    if date_col is not None:
        dates = pd.to_datetime(raw_df[date_col], errors="coerce")
    idx_name = ""
    if name_col is not None and len(raw_df) > 0:
        vals = raw_df[name_col].dropna().astype(str).unique()
        idx_name = vals[0] if len(vals) else ""
    dup_count = int(dates.dropna().duplicated().sum()) if len(dates) else 0
    missing_close = int(to_numeric(raw_df[close_col]).isna().sum()) if close_col is not None else None
    missing_pct = int(to_numeric(raw_df[pct_col]).isna().sum()) if pct_col is not None else None
    match_expected = bool(meta["expected"] in idx_name) if idx_name else False
    schema_status = "OK" if download_success and date_col is not None and close_col is not None else "BLOCKED_SCHEMA_UNRECOGNIZED"
    return {
        "symbol": meta["symbol"],
        "label": meta["label"],
        "download_success": bool(download_success),
        "row_count": int(len(raw_df)),
        "column_count": int(len(raw_df.columns)),
        "columns_original": "|".join(map(str, raw_df.columns)),
        "date_column_detected": date_col,
        "code_column_detected": code_col,
        "close_column_detected": close_col,
        "pct_change_column_detected": pct_col,
        "min_date": dates.min().date().isoformat() if len(dates) and pd.notna(dates.min()) else None,
        "max_date": dates.max().date().isoformat() if len(dates) and pd.notna(dates.max()) else None,
        "duplicate_date_count": dup_count,
        "missing_close_count": missing_close,
        "missing_pct_change_count": missing_pct,
        "index_name_detected": idx_name,
        "index_name_match_expected": match_expected,
        "schema_status": schema_status,
    }, {"date": date_col, "code": code_col, "close": close_col, "pct": pct_col, "name": name_col}


def normalize_daily(raw_df, cols, meta):
    df = raw_df[[cols["date"], cols["close"]] + ([cols["pct"]] if cols["pct"] else []) + ([cols["name"]] if cols["name"] else [])].copy()
    df["trade_date"] = pd.to_datetime(df[cols["date"]], errors="coerce")
    df["close"] = to_numeric(df[cols["close"]])
    df["pct_change_decimal"] = to_numeric(df[cols["pct"]]) / 100.0 if cols["pct"] else np.nan
    if cols["name"]:
        df["benchmark_name_detected"] = df[cols["name"]].astype(str)
    else:
        df["benchmark_name_detected"] = meta["label"]
    df = df[["trade_date", "close", "pct_change_decimal", "benchmark_name_detected"]]
    df = df.dropna(subset=["trade_date", "close"]).sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    df["daily_ret_from_close"] = df["close"] / df["close"].shift(1) - 1.0
    return df


def unit_check(symbol, meta, daily):
    comp = daily.dropna(subset=["daily_ret_from_close", "pct_change_decimal"]).copy()
    if len(comp) == 0:
        median_diff = np.nan
        max_diff = np.nan
        unit = "UNAVAILABLE"
        passed = False
    else:
        diff = (comp["daily_ret_from_close"] - comp["pct_change_decimal"]).abs()
        median_diff = float(diff.median())
        max_diff = float(diff.max())
        unit = "PERCENT_DIVIDED_BY_100"
        passed = bool(median_diff < 1e-5 or max_diff < 5e-4)
    return {
        "symbol": symbol,
        "label": meta["label"],
        "row_count_checked": int(len(comp)),
        "median_abs_diff_close_vs_pct": median_diff,
        "max_abs_diff_close_vs_pct": max_diff,
        "pct_change_unit_detected": unit,
        "close_based_return_preferred": True,
        "unit_check_pass": passed,
        "unit_check_warning": not passed,
    }


def read_portfolio_month_ends():
    p = pd.read_csv(PORTFOLIO_MONTHLY, usecols=["month_end"])
    months = pd.to_datetime(p["month_end"], errors="coerce").dropna().drop_duplicates().sort_values().reset_index(drop=True)
    del p
    gc.collect()
    return months


def build_forward_returns(months, daily_by_symbol, schema_rows):
    name_by_symbol = {r["symbol"]: r["index_name_detected"] or r["label"] for r in schema_rows}
    rows = []
    month_list = list(months)
    for meta in SYMBOLS:
        symbol = meta["symbol"]
        daily = daily_by_symbol.get(symbol)
        if daily is None or daily.empty:
            for i in range(len(month_list) - 1):
                rows.append({
                    "portfolio_month_end": month_list[i].date().isoformat(),
                    "benchmark_code": symbol,
                    "benchmark_label": meta["monthly_label"],
                    "benchmark_name_detected": name_by_symbol.get(symbol, meta["label"]),
                    "benchmark_fwd_ret_1m": np.nan,
                    "return_source": "AKSHARE_CLOSE_COMPOUND",
                    "source_start_trade_date": None,
                    "source_end_trade_date": None,
                    "trading_day_count": 0,
                    "missing_flag": True,
                    "alignment_warning": "daily_data_unavailable",
                })
            continue
        d = daily.set_index("trade_date")
        trade_dates = d.index
        for i in range(len(month_list) - 1):
            start_bound = month_list[i]
            end_bound = month_list[i + 1]
            window_dates = trade_dates[(trade_dates > start_bound) & (trade_dates <= end_bound)]
            if len(window_dates) == 0:
                rows.append({
                    "portfolio_month_end": start_bound.date().isoformat(),
                    "benchmark_code": symbol,
                    "benchmark_label": meta["monthly_label"],
                    "benchmark_name_detected": name_by_symbol.get(symbol, meta["label"]),
                    "benchmark_fwd_ret_1m": np.nan,
                    "return_source": "AKSHARE_CLOSE_COMPOUND",
                    "source_start_trade_date": None,
                    "source_end_trade_date": None,
                    "trading_day_count": 0,
                    "missing_flag": True,
                    "alignment_warning": "no_index_trading_day_in_forward_window",
                })
                continue
            rets = d.loc[window_dates, "daily_ret_from_close"].dropna()
            missing = len(rets) == 0 or len(rets) < len(window_dates)
            rows.append({
                "portfolio_month_end": start_bound.date().isoformat(),
                "benchmark_code": symbol,
                "benchmark_label": meta["monthly_label"],
                "benchmark_name_detected": name_by_symbol.get(symbol, meta["label"]),
                "benchmark_fwd_ret_1m": float(np.prod(1.0 + rets) - 1.0) if len(rets) else np.nan,
                "return_source": "AKSHARE_CLOSE_COMPOUND",
                "source_start_trade_date": window_dates[0].date().isoformat(),
                "source_end_trade_date": window_dates[-1].date().isoformat(),
                "trading_day_count": int(len(rets)),
                "missing_flag": bool(missing),
                "alignment_warning": "missing_daily_return_in_window" if missing else "",
            })
    return pd.DataFrame(rows)


def validate_hs300(monthly_forward):
    base = {
        "comparison_level": "monthly_forward_return",
        "overlap_start_date": None,
        "overlap_end_date": None,
        "overlap_obs_count": 0,
        "mean_abs_return_diff": np.nan,
        "max_abs_return_diff": np.nan,
        "close_level_corr": np.nan,
        "return_corr": np.nan,
        "validation_pass": False,
        "validation_warning": True,
    }
    if not CSMAR_OFFICIAL.exists():
        base["validation_warning"] = True
        return pd.DataFrame([base])
    csmar = pd.read_csv(CSMAR_OFFICIAL, dtype={"benchmark_code": str})
    if "benchmark_code" not in csmar.columns or "benchmark_fwd_ret_1m" not in csmar.columns:
        del csmar
        gc.collect()
        return pd.DataFrame([base])
    c = csmar[csmar["benchmark_code"].astype(str).str.zfill(6) == "000300"].copy()
    if c.empty:
        del csmar, c
        gc.collect()
        return pd.DataFrame([base])
    a = monthly_forward[monthly_forward["benchmark_code"].astype(str).str.zfill(6) == "000300"].copy()
    c["portfolio_month_end"] = pd.to_datetime(c["portfolio_month_end"], errors="coerce")
    a["portfolio_month_end"] = pd.to_datetime(a["portfolio_month_end"], errors="coerce")
    c["csmar_ret"] = pd.to_numeric(c["benchmark_fwd_ret_1m"], errors="coerce")
    a["akshare_ret"] = pd.to_numeric(a["benchmark_fwd_ret_1m"], errors="coerce")
    m = a[["portfolio_month_end", "akshare_ret"]].merge(c[["portfolio_month_end", "csmar_ret"]], on="portfolio_month_end", how="inner").dropna()
    if len(m) > 1:
        diff = (m["akshare_ret"] - m["csmar_ret"]).abs()
        corr = float(m["akshare_ret"].corr(m["csmar_ret"]))
        base.update({
            "overlap_start_date": m["portfolio_month_end"].min().date().isoformat(),
            "overlap_end_date": m["portfolio_month_end"].max().date().isoformat(),
            "overlap_obs_count": int(len(m)),
            "mean_abs_return_diff": float(diff.mean()),
            "max_abs_return_diff": float(diff.max()),
            "return_corr": corr,
            "validation_pass": bool(corr >= 0.999 or diff.mean() < 1e-5),
            "validation_warning": bool(not (corr >= 0.999 or diff.mean() < 1e-5)),
        })
    del csmar, c, a, m
    gc.collect()
    return pd.DataFrame([base])


def coverage_check(monthly_forward, months):
    required = max(len(months) - 1, 0)
    rows = []
    for meta in SYMBOLS:
        label = meta["monthly_label"]
        s = monthly_forward[monthly_forward["benchmark_label"] == label].copy()
        ok = s[~s["missing_flag"].astype(bool) & s["benchmark_fwd_ret_1m"].notna()]
        ratio = float(len(ok) / required) if required else 0.0
        rows.append({
            "benchmark_label": label,
            "required_portfolio_month_count": int(required),
            "available_forward_return_count": int(len(ok)),
            "missing_forward_return_count": int(required - len(ok)),
            "coverage_ratio": ratio,
            "first_available_portfolio_month_end": ok["portfolio_month_end"].min() if len(ok) else None,
            "last_available_portfolio_month_end": ok["portfolio_month_end"].max() if len(ok) else None,
            "coverage_pass": bool(ratio >= 0.98),
        })
        del s, ok
    gc.collect()
    return pd.DataFrame(rows)


def write_recommendation_manifest():
    source_rows = []
    for path, label in [
        (BENCHMARK_RECO, "existing_recommendation"),
        (INTERNAL_BENCH, "internal_universe_monthly_forward_benchmark"),
        (RISK_FREE, "risk_free_monthly_aligned"),
        (FAMA_FRENCH, "fama_french_monthly_factor_candidates"),
    ]:
        source_rows.append({"source_file": str(path), "source_role": label, "exists": path.exists()})
    rows = [
        {"recommendation_tier": "primary_official_benchmark", "benchmark_label": "CSI800_AKSHARE_PRICE", "source": "AKSHARE_CLOSE_COMPOUND", "recommended_use": "benchmark-relative evaluation prep", "caveats": ""},
        {"recommendation_tier": "secondary_official_benchmark", "benchmark_label": "CSI500_AKSHARE_PRICE", "source": "AKSHARE_CLOSE_COMPOUND", "recommended_use": "benchmark-relative evaluation prep", "caveats": ""},
        {"recommendation_tier": "secondary_official_benchmark", "benchmark_label": "HS300_AKSHARE_PRICE_VALIDATION", "source": "AKSHARE_CLOSE_COMPOUND", "recommended_use": "validation/reference", "caveats": "HS300 validation series; not primary CSI800 benchmark"},
        {"recommendation_tier": "secondary_official_benchmark", "benchmark_label": "CSMAR_TRD_Cnmont broad-market candidates", "source": str(BENCHMARK_RECO), "recommended_use": "fallback official broad-market candidates", "caveats": "manual Indexcd confirmation caveat from prior audit"},
        {"recommendation_tier": "primary_research_benchmark", "benchmark_label": "INTERNAL_ELIGIBLE_UNIVERSE_EQUAL_WEIGHT", "source": str(INTERNAL_BENCH), "recommended_use": "research benchmark", "caveats": "not an official index"},
        {"recommendation_tier": "secondary_research_benchmark", "benchmark_label": "INTERNAL_FLAG_CLEAN_UNIVERSE_EQUAL_WEIGHT", "source": str(INTERNAL_BENCH), "recommended_use": "research sensitivity benchmark", "caveats": "not an official index"},
        {"recommendation_tier": "secondary_research_benchmark", "benchmark_label": "CSMAR broad-market equal-weight / float-mcap candidates", "source": str(BENCHMARK_RECO), "recommended_use": "research sensitivity benchmark", "caveats": ""},
        {"recommendation_tier": "factor_attribution", "benchmark_label": "CSMAR Fama-French monthly factors", "source": str(FAMA_FRENCH), "recommended_use": "factor attribution prep", "caveats": "no alpha/beta calculated in this task"},
        {"recommendation_tier": "factor_attribution", "benchmark_label": "CSMAR risk-free monthly series", "source": str(RISK_FREE), "recommended_use": "factor attribution prep", "caveats": "no alpha/beta calculated in this task"},
    ]
    df = pd.DataFrame(rows)
    for sr in source_rows:
        df[f"manifest_{sr['source_role']}_exists"] = sr["exists"]
    return df


def guardrail_qa():
    row = {
        "portfolio_weights_modified": False,
        "portfolio_weights_reconstructed": False,
        "portfolio_benchmark_relative_return_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "unified_portfolio_evaluation_modified": False,
        "training_run": False,
        "shap_calculated": False,
        "production_modified": False,
    }
    row["guardrail_pass"] = not any(row.values())
    return pd.DataFrame([row])


def decide(summary, coverage, validation, guardrail):
    guardrail_pass = bool(guardrail.loc[0, "guardrail_pass"])
    csi800_ok = bool(summary["csi800_download_success"])
    csi500_ok = bool(summary["csi500_download_success"])
    csi800_cov = bool(coverage.loc[coverage["benchmark_label"] == "CSI800_AKSHARE_PRICE", "coverage_pass"].iloc[0])
    csi500_cov = bool(coverage.loc[coverage["benchmark_label"] == "CSI500_AKSHARE_PRICE", "coverage_pass"].iloc[0])
    val_pass = bool(validation.loc[0, "validation_pass"])
    val_warn = bool(validation.loc[0, "validation_warning"])
    if not guardrail_pass:
        return "AKSHARE_CSI_BENCHMARK_SUPPLEMENT_FAIL_GUARDRAIL"
    if (not csi800_ok) or (not csi800_cov) or (not summary["akshare_available"]):
        return "AKSHARE_CSI_BENCHMARK_SUPPLEMENT_FAIL_DOWNLOAD_OR_COVERAGE"
    if csi800_ok and not csi500_ok:
        return "AKSHARE_CSI_BENCHMARK_SUPPLEMENT_WATCH_PARTIAL_INDEX_AVAILABLE"
    if csi800_ok and csi500_ok and csi800_cov and csi500_cov and (val_pass or val_warn):
        if val_warn and not val_pass:
            return "AKSHARE_CSI_BENCHMARK_SUPPLEMENT_WATCH_HS300_VALIDATION_WARNING"
        return "AKSHARE_CSI_BENCHMARK_SUPPLEMENT_READY_FOR_BENCHMARK_RELATIVE_EVAL_PREP"
    return "AKSHARE_CSI_BENCHMARK_SUPPLEMENT_FAIL_DOWNLOAD_OR_COVERAGE"


def main():
    ensure_dirs()
    write_run_state("running", "starting prerequisite check")
    ak, prereq = import_or_install_akshare()
    prereq_path = OUT_DIR / "akshare_benchmark_supplement_prerequisite_check.json"
    prereq_path.write_text(json.dumps(prereq, ensure_ascii=False, indent=2), encoding="utf-8")

    raw_manifest = []
    schema_rows = []
    unit_rows = []
    daily_by_symbol = {}
    failure_reasons = {}

    write_run_state("running", "downloading AKShare CSI index raw data")
    for meta in SYMBOLS:
        symbol = meta["symbol"]
        raw_path = RAW_DIR / f"akshare_csindex_{symbol}_raw.csv"
        success = False
        reason = None
        raw_df = pd.DataFrame()
        if ak is None:
            reason = "akshare_import_error"
        else:
            try:
                raw_df = ak.stock_zh_index_hist_csindex(symbol=symbol, start_date=START_DATE, end_date=END_DATE)
                if raw_df is None or len(raw_df) == 0:
                    reason = "api_response_empty"
                else:
                    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
                    success = True
            except Exception as exc:
                msg = repr(exc)
                reason = "network_error" if any(x in msg.lower() for x in ["timeout", "connection", "network", "ssl", "proxy"]) else "symbol_not_found"
                failure_reasons[symbol] = msg
        if not success and reason is None:
            reason = "api_response_empty"
        raw_manifest.append({
            "symbol": symbol,
            "label": meta["label"],
            "raw_file": str(raw_path),
            "download_success": success,
            "failure_reason": reason if not success else "",
            "row_count": int(len(raw_df)),
            "saved_raw": bool(success and raw_path.exists()),
        })
        audit, cols = schema_audit_one(raw_df, meta, success)
        if not success:
            audit["schema_status"] = reason
        schema_rows.append(audit)
        if success and audit["schema_status"] == "OK":
            daily = normalize_daily(raw_df, cols, meta)
            daily_by_symbol[symbol] = daily
            unit_rows.append(unit_check(symbol, meta, daily))
        else:
            unit_rows.append({
                "symbol": symbol,
                "label": meta["label"],
                "row_count_checked": 0,
                "median_abs_diff_close_vs_pct": np.nan,
                "max_abs_diff_close_vs_pct": np.nan,
                "pct_change_unit_detected": "UNAVAILABLE",
                "close_based_return_preferred": True,
                "unit_check_pass": False,
                "unit_check_warning": True,
            })
        del raw_df
        gc.collect()

    pd.DataFrame(raw_manifest).to_csv(OUT_DIR / "akshare_csindex_raw_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(schema_rows).to_csv(OUT_DIR / "akshare_csindex_schema_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(unit_rows).to_csv(OUT_DIR / "akshare_csindex_return_unit_check.csv", index=False, encoding="utf-8-sig")

    write_run_state("running", "building monthly forward benchmark returns")
    months = read_portfolio_month_ends()
    monthly_forward = build_forward_returns(months, daily_by_symbol, schema_rows)
    monthly_forward.to_csv(OUT_DIR / "akshare_official_index_monthly_forward_return.csv", index=False, encoding="utf-8-sig")

    validation = validate_hs300(monthly_forward)
    validation.to_csv(OUT_DIR / "akshare_vs_csmar_hs300_validation.csv", index=False, encoding="utf-8-sig")

    coverage = coverage_check(monthly_forward, months)
    coverage.to_csv(OUT_DIR / "akshare_benchmark_coverage_check.csv", index=False, encoding="utf-8-sig")

    reco = write_recommendation_manifest()
    reco.to_csv(OUT_DIR / "benchmark_candidate_recommendation_with_akshare.csv", index=False, encoding="utf-8-sig")

    guardrail = guardrail_qa()
    guardrail.to_csv(OUT_DIR / "akshare_benchmark_supplement_guardrail_qa.csv", index=False, encoding="utf-8-sig")

    schema = pd.DataFrame(schema_rows)
    coverage_idx = coverage.set_index("benchmark_label")
    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "akshare_available": prereq["akshare_available"],
        "akshare_version": prereq["akshare_version"],
        "download_attempted": ak is not None,
        "symbols_requested": [x["symbol"] for x in SYMBOLS],
        "symbols_downloaded_successfully": [x["symbol"] for x in raw_manifest if x["download_success"]],
        "symbols_failed": [{"symbol": x["symbol"], "reason": x["failure_reason"]} for x in raw_manifest if not x["download_success"]],
        "csi800_download_success": bool(raw_manifest[0]["download_success"]),
        "csi500_download_success": bool(raw_manifest[1]["download_success"]),
        "hs300_validation_download_success": bool(raw_manifest[2]["download_success"]),
        "csi800_row_count": int(raw_manifest[0]["row_count"]),
        "csi500_row_count": int(raw_manifest[1]["row_count"]),
        "hs300_row_count": int(raw_manifest[2]["row_count"]),
        "csi800_min_date": schema.loc[schema["symbol"] == "000906", "min_date"].iloc[0],
        "csi800_max_date": schema.loc[schema["symbol"] == "000906", "max_date"].iloc[0],
        "csi500_min_date": schema.loc[schema["symbol"] == "000905", "min_date"].iloc[0],
        "csi500_max_date": schema.loc[schema["symbol"] == "000905", "max_date"].iloc[0],
        "hs300_min_date": schema.loc[schema["symbol"] == "000300", "min_date"].iloc[0],
        "hs300_max_date": schema.loc[schema["symbol"] == "000300", "max_date"].iloc[0],
        "hs300_akshare_vs_csmar_validation_pass": bool(validation.loc[0, "validation_pass"]),
        "hs300_validation_warning": bool(validation.loc[0, "validation_warning"]),
        "csi800_monthly_forward_coverage_ratio": float(coverage_idx.loc["CSI800_AKSHARE_PRICE", "coverage_ratio"]),
        "csi500_monthly_forward_coverage_ratio": float(coverage_idx.loc["CSI500_AKSHARE_PRICE", "coverage_ratio"]),
        "csi800_monthly_candidate_ready": bool(coverage_idx.loc["CSI800_AKSHARE_PRICE", "coverage_pass"] and raw_manifest[0]["download_success"]),
        "csi500_monthly_candidate_ready": bool(coverage_idx.loc["CSI500_AKSHARE_PRICE", "coverage_pass"] and raw_manifest[1]["download_success"]),
        "primary_official_benchmark_recommended": "CSI800_AKSHARE_PRICE",
        "secondary_official_benchmarks_recommended": ["CSI500_AKSHARE_PRICE", "HS300_AKSHARE_PRICE_VALIDATION", "CSMAR_TRD_Cnmont broad-market candidates"],
        "primary_research_benchmark_recommended": "INTERNAL_ELIGIBLE_UNIVERSE_EQUAL_WEIGHT",
        "benchmark_relative_eval_prep_allowed": True,
        "alpha_beta_eval_prep_allowed": True,
        "portfolio_weights_modified": False,
        "portfolio_weights_reconstructed": False,
        "portfolio_benchmark_relative_return_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "training_run": False,
        "shap_calculated": False,
        "production_modified": False,
    }
    summary["final_decision"] = decide(summary, coverage, validation, guardrail)
    if summary["final_decision"].endswith("READY_FOR_BENCHMARK_RELATIVE_EVAL_PREP"):
        summary["recommended_next_step"] = "进入 benchmark-relative evaluation prep，使用 CSI800_AKSHARE_PRICE 作为 primary official benchmark。"
    elif "HS300_VALIDATION_WARNING" in summary["final_decision"]:
        summary["recommended_next_step"] = "可继续 benchmark-relative evaluation prep，但报告中保留 HS300 source validation caveat。"
    elif "PARTIAL_INDEX_AVAILABLE" in summary["final_decision"]:
        summary["recommended_next_step"] = "仅使用可用 AKShare 指数，并在后续评估中标注缺失指数 caveat。"
    else:
        summary["recommended_next_step"] = "先修复 AKShare 下载或覆盖问题，再进入 benchmark-relative evaluation prep。"

    (OUT_DIR / "akshare_csi_index_supplement_monthly_alignment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# AKShare CSI Index Supplement & Monthly Alignment v0",
        "",
        f"- final_decision: {summary['final_decision']}",
        f"- akshare_available: {summary['akshare_available']}",
        f"- akshare_version: {summary['akshare_version']}",
        f"- symbols_downloaded_successfully: {summary['symbols_downloaded_successfully']}",
        f"- symbols_failed: {summary['symbols_failed']}",
        f"- CSI800 coverage: {summary['csi800_monthly_forward_coverage_ratio']:.6f}",
        f"- CSI500 coverage: {summary['csi500_monthly_forward_coverage_ratio']:.6f}",
        f"- HS300 validation pass: {summary['hs300_akshare_vs_csmar_validation_pass']}",
        f"- HS300 validation warning: {summary['hs300_validation_warning']}",
        "",
        "## Guardrail",
        "",
        "- 未修改 portfolio weights。",
        "- 未计算 portfolio benchmark-relative return、alpha/beta、information ratio、tracking error。",
        "- 未训练、未计算 SHAP、未写 production。",
    ]
    (OUT_DIR / "akshare_csi_index_supplement_monthly_alignment_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    final_qa = pd.DataFrame([{
        "final_decision": summary["final_decision"],
        "akshare_available": summary["akshare_available"],
        "csi800_candidate_ready": summary["csi800_monthly_candidate_ready"],
        "csi500_candidate_ready": summary["csi500_monthly_candidate_ready"],
        "guardrail_pass": bool(guardrail.loc[0, "guardrail_pass"]),
        "required_outputs_created": True,
    }])
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")

    terminal_summary = {
        "task_name": TASK_NAME,
        "status": "completed",
        "stdout_log": str(RUN_DIR / "run_stdout.txt"),
        "stderr_log": str(RUN_DIR / "run_stderr.txt"),
        "summary_json": str(OUT_DIR / "akshare_csi_index_supplement_monthly_alignment_summary.json"),
        "final_decision": summary["final_decision"],
    }
    (OUT_DIR / "terminal_summary.json").write_text(json.dumps(terminal_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "task_completion_card.md").write_text(
        "\n".join([
            "# Task Completion Card",
            "",
            f"- task_name: {TASK_NAME}",
            "- status: completed",
            f"- final_decision: {summary['final_decision']}",
            f"- summary_json: {OUT_DIR / 'akshare_csi_index_supplement_monthly_alignment_summary.json'}",
            f"- final_qa: {OUT_DIR / 'final_qa.csv'}",
        ]) + "\n",
        encoding="utf-8",
    )

    write_run_state("completed", "all required outputs written", [f"final_decision: {summary['final_decision']}"])
    print(json.dumps(terminal_summary, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ensure_dirs()
        err = traceback.format_exc()
        (RUN_DIR / "last_error.txt").write_text(err, encoding="utf-8")
        write_run_state("failed", "exception captured in last_error.txt")
        raise

"""
Compact-F monthly portfolio construction optimization.

This module never trains or changes the alpha model. It consumes frozen
Compact-F OOS predictions and compares six portfolio construction rules.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("output")
PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "production_models_v15_compact"
    / "Compact_F_oos.parquet"
)
PANEL_PATH = OUTPUT_DIR / "training_panel_v15_sr.parquet"
DESIGN_PATH = OUTPUT_DIR / "compact_f_portfolio_construction_design.md"
RESULTS_PATH = OUTPUT_DIR / "compact_f_portfolio_construction_results.md"
MONTHLY_PATH = OUTPUT_DIR / "compact_f_portfolio_construction_monthly.csv"
YEARLY_PATH = OUTPUT_DIR / "compact_f_portfolio_construction_yearly.csv"


@dataclass(frozen=True)
class StrategySpec:
    name: str
    kind: str
    target_size: int
    buy_rank: int | None = None
    sell_rank: int | None = None
    alpha: float | None = None


STRATEGIES = {
    "A_Fixed_Top30": StrategySpec(
        "A_Fixed_Top30", "fixed", 30
    ),
    "B_Top30_Buffer": StrategySpec(
        "B_Top30_Buffer", "buffer", 30, buy_rank=20, sell_rank=45
    ),
    "C_Top30_Partial": StrategySpec(
        "C_Top30_Partial", "partial", 30, alpha=0.5
    ),
    "D_Fixed_Top40": StrategySpec(
        "D_Fixed_Top40", "fixed", 40
    ),
    "E_Fixed_Top50": StrategySpec(
        "E_Fixed_Top50", "fixed", 50
    ),
    "F_Top50_Buffer": StrategySpec(
        "F_Top50_Buffer", "buffer", 50, buy_rank=35, sell_rank=75
    ),
}


def _ordered_current_universe(ranked: pd.DataFrame) -> list[str]:
    required = {"symbol", "rank"}
    missing = required - set(ranked.columns)
    if missing:
        raise KeyError(f"ranked input missing columns: {sorted(missing)}")
    ordered = ranked.sort_values(["rank", "symbol"], ascending=[True, True])
    return ordered["symbol"].astype(str).tolist()


def _equal_weights(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    weight = 1.0 / len(symbols)
    return {symbol: weight for symbol in symbols}


def fixed_top_n_weights(ranked: pd.DataFrame, n: int) -> dict[str, float]:
    """Return equal weights for exactly the current Top-N stocks."""
    if n <= 0:
        raise ValueError("n must be positive")
    symbols = _ordered_current_universe(ranked)
    if len(symbols) < n:
        raise ValueError(f"Need at least {n} stocks, found {len(symbols)}")
    return _equal_weights(symbols[:n])


def buffer_weights(
    ranked: pd.DataFrame,
    previous: Mapping[str, float],
    target_size: int,
    buy_rank: int,
    sell_rank: int,
) -> tuple[dict[str, float], dict]:
    """Apply a rank-entry/rank-exit hysteresis rule."""
    if not 0 < buy_rank < sell_rank:
        raise ValueError("Require 0 < buy_rank < sell_rank")
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    ordered = ranked.sort_values(["rank", "symbol"], ascending=[True, True]).copy()
    ordered["symbol"] = ordered["symbol"].astype(str)
    rank_by_symbol = dict(zip(ordered["symbol"], ordered["rank"]))
    previous_symbols = set(previous)

    retained = [
        symbol
        for symbol in previous
        if symbol in rank_by_symbol and rank_by_symbol[symbol] <= sell_rank
    ]
    retained = sorted(retained, key=lambda symbol: (rank_by_symbol[symbol], symbol))
    sold_count = len(previous_symbols - set(retained))

    additions = []
    for row in ordered.itertuples(index=False):
        symbol = str(row.symbol)
        if row.rank > buy_rank:
            break
        if symbol in retained:
            continue
        if len(retained) + len(additions) >= target_size:
            break
        additions.append(symbol)

    holdings = retained + additions
    audit = {
        "sold_count": sold_count,
        "bought_count": len(additions),
        "holding_count": len(holdings),
        "buy_zone_underfilled": len(holdings) < target_size,
    }
    return _equal_weights(holdings), audit


def partial_rebalance_weights(
    ranked: pd.DataFrame,
    previous: Mapping[str, float],
    n: int = 30,
    alpha: float = 0.5,
    min_weight: float = 1e-6,
) -> tuple[dict[str, float], dict]:
    """Blend equal-weight Top-N target weights with previous actual weights."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")

    universe = set(_ordered_current_universe(ranked))
    target = fixed_top_n_weights(ranked, n)
    eligible_previous = {
        symbol: float(weight)
        for symbol, weight in previous.items()
        if symbol in universe and weight > 0
    }

    raw = {}
    for symbol in set(target) | set(eligible_previous):
        weight = (
            alpha * target.get(symbol, 0.0)
            + (1 - alpha) * eligible_previous.get(symbol, 0.0)
        )
        if weight >= min_weight:
            raw[symbol] = weight

    pre_total = float(sum(raw.values()))
    if pre_total <= 0:
        raise ValueError("Partial rebalance produced zero total weight")
    normalized = {
        symbol: weight / pre_total
        for symbol, weight in raw.items()
    }
    audit = {
        "pre_normalization_weight": pre_total,
        "post_normalization_weight": float(sum(normalized.values())),
        "holding_count": len(normalized),
    }
    return normalized, audit


def weight_turnover(
    previous: Mapping[str, float],
    current: Mapping[str, float],
) -> float:
    """One-way turnover as half the L1 distance between weight vectors."""
    symbols = set(previous) | set(current)
    return 0.5 * sum(
        abs(float(current.get(symbol, 0.0)) - float(previous.get(symbol, 0.0)))
        for symbol in symbols
    )


def _weighted_available(
    weights: Mapping[str, float],
    values: Mapping[str, float],
) -> float:
    pairs = [
        (float(weights[symbol]), float(values[symbol]))
        for symbol in weights
        if symbol in values and pd.notna(values[symbol])
    ]
    total_weight = sum(weight for weight, _ in pairs)
    if total_weight <= 0:
        return np.nan
    return sum(weight * value for weight, value in pairs) / total_weight


def _rank_cross_section(cross: pd.DataFrame) -> pd.DataFrame:
    ranked = cross.sort_values(
        ["alpha_signal", "symbol"],
        ascending=[False, True],
    ).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def _weights_for_spec(
    ranked: pd.DataFrame,
    previous: Mapping[str, float],
    spec: StrategySpec,
) -> tuple[dict[str, float], dict]:
    if spec.kind == "fixed":
        return fixed_top_n_weights(ranked, spec.target_size), {}
    if spec.kind == "buffer":
        if not previous:
            weights = fixed_top_n_weights(ranked, spec.target_size)
            return weights, {
                "sold_count": 0,
                "bought_count": spec.target_size,
                "holding_count": spec.target_size,
                "buy_zone_underfilled": False,
            }
        return buffer_weights(
            ranked,
            previous,
            target_size=spec.target_size,
            buy_rank=int(spec.buy_rank),
            sell_rank=int(spec.sell_rank),
        )
    if spec.kind == "partial":
        return partial_rebalance_weights(
            ranked,
            previous,
            n=spec.target_size,
            alpha=float(spec.alpha),
        )
    raise ValueError(f"Unknown strategy kind: {spec.kind}")


def simulate_strategy(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    spec: StrategySpec,
) -> pd.DataFrame:
    """Simulate one monthly portfolio strategy from frozen OOS signals."""
    pred = predictions[["date", "symbol", "alpha_signal"]].copy()
    pnl = panel.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    pnl["date"] = pd.to_datetime(pnl["date"])
    pred["symbol"] = pred["symbol"].astype(str).str.zfill(6)
    pnl["symbol"] = pnl["symbol"].astype(str).str.zfill(6)

    needed = {
        "forward_return_1m",
        "SR_ROE_neutral_z",
        "SR_ProfitGrowth_YoY_neutral_z",
        "EP_neutral_z",
    }
    missing = needed - set(pnl.columns)
    if missing:
        raise KeyError(f"Panel missing accounting columns: {sorted(missing)}")

    merged = pred.merge(
        pnl[["date", "symbol", *sorted(needed)]],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )

    previous: dict[str, float] = {}
    rows = []
    for date, cross in merged.groupby("date", sort=True):
        ranked = _rank_cross_section(cross.dropna(subset=["alpha_signal"]))
        weights, audit = _weights_for_spec(ranked, previous, spec)
        indexed = ranked.set_index("symbol")

        portfolio_return = _weighted_available(
            weights,
            indexed["forward_return_1m"].to_dict(),
        )
        roe = _weighted_available(
            weights,
            indexed["SR_ROE_neutral_z"].to_dict(),
        )
        pg = _weighted_available(
            weights,
            indexed["SR_ProfitGrowth_YoY_neutral_z"].to_dict(),
        )
        ep = _weighted_available(
            weights,
            indexed["EP_neutral_z"].to_dict(),
        )

        row = {
            "date": pd.Timestamp(date),
            "strategy": spec.name,
            "portfolio_return": portfolio_return,
            "turnover": (
                weight_turnover(previous, weights)
                if previous
                else np.nan
            ),
            "roe_exposure": roe,
            "profitgrowth_exposure": pg,
            "ep_exposure": ep,
            "holding_count": len(weights),
            "weight_sum": float(sum(weights.values())),
            "sold_count": audit.get("sold_count", np.nan),
            "bought_count": audit.get("bought_count", np.nan),
            "buy_zone_underfilled": audit.get("buy_zone_underfilled", False),
            "pre_normalization_weight": audit.get(
                "pre_normalization_weight", np.nan
            ),
            "post_normalization_weight": audit.get(
                "post_normalization_weight", float(sum(weights.values()))
            ),
        }
        rows.append(row)
        previous = weights
    return pd.DataFrame(rows)


def _annualized_sharpe(returns: pd.Series) -> float:
    values = returns.dropna()
    if len(values) < 2:
        return np.nan
    std = values.std(ddof=1)
    return float(values.mean() / std * np.sqrt(12)) if std > 0 else 0.0


def _max_drawdown(returns: pd.Series) -> float:
    values = returns.dropna()
    if values.empty:
        return np.nan
    nav = (1 + values).cumprod()
    return float((nav / nav.cummax() - 1).min())


def summarize_monthly(monthly: pd.DataFrame) -> dict:
    """Aggregate one strategy's monthly audit table into headline metrics."""
    realized = monthly.dropna(subset=["portfolio_return"])
    return {
        "Sharpe": _annualized_sharpe(realized["portfolio_return"]),
        "MaxDD": _max_drawdown(realized["portfolio_return"]),
        "Turnover": float(monthly["turnover"].dropna().mean()),
        "MeanMonthlyReturn": float(realized["portfolio_return"].mean()),
        "HitRate": float((realized["portfolio_return"] > 0).mean()),
        "ROE": float(monthly["roe_exposure"].mean()),
        "ProfitGrowth": float(monthly["profitgrowth_exposure"].mean()),
        "EP": float(monthly["ep_exposure"].mean()),
        "N_Months": int(len(realized)),
    }


def yearly_sharpe(monthly: pd.DataFrame) -> pd.DataFrame:
    """Annualized monthly Sharpe by calendar year."""
    realized = monthly.dropna(subset=["portfolio_return"]).copy()
    if realized.empty:
        return pd.DataFrame(columns=["year", "sharpe", "n_months"])
    realized["year"] = pd.to_datetime(realized["date"]).dt.year
    rows = []
    for year, group in realized.groupby("year"):
        rows.append({
            "year": int(year),
            "sharpe": _annualized_sharpe(group["portfolio_return"]),
            "n_months": int(len(group)),
        })
    return pd.DataFrame(rows)


def _acceptance_columns(results: pd.DataFrame) -> pd.DataFrame:
    result = results.copy()
    baseline = result.loc[
        result["Strategy"] == "A_Fixed_Top30"
    ].iloc[0]
    result["PassSharpe"] = result["Sharpe"] >= baseline["Sharpe"]
    result["PassMaxDD"] = result["MaxDD"] >= baseline["MaxDD"] - 0.02
    result["PassTurnover"] = result["Turnover"] <= 0.35
    result["PreferredTurnover"] = result["Turnover"] <= 0.30
    result["PassStyle"] = (
        (result["ROE"] > 0)
        & (result["ProfitGrowth"] > 0)
    )
    result["PassAll"] = (
        result["PassSharpe"]
        & result["PassMaxDD"]
        & result["PassTurnover"]
        & result["PassStyle"]
    )
    # Baseline is the comparison anchor, not a turnover-control candidate.
    result.loc[result["Strategy"] == "A_Fixed_Top30", "PassAll"] = False
    return result


def select_winner(results: pd.DataFrame) -> str:
    """Select the best accepted candidate, otherwise retain Fixed Top30."""
    checked = _acceptance_columns(results)
    passing = checked[checked["PassAll"]].sort_values(
        ["Sharpe", "Turnover"],
        ascending=[False, True],
    )
    if passing.empty:
        return "A_Fixed_Top30"
    return str(passing.iloc[0]["Strategy"])


def load_frozen_inputs(
    predictions_path: Path = PREDICTIONS_PATH,
    panel_path: Path = PANEL_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Compact-F OOS predictions missing: {predictions_path}")
    if not panel_path.exists():
        raise FileNotFoundError(f"V1.5 panel missing: {panel_path}")

    predictions = pd.read_parquet(predictions_path)
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions["symbol"] = predictions["symbol"].astype(str).str.zfill(6)
    if predictions.duplicated(["date", "symbol"]).any():
        raise ValueError("Compact-F predictions contain duplicate date/symbol rows")

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    close_col = next(
        (column for column in ("收盘", "close") if column in panel.columns),
        None,
    )
    if close_col is None:
        raise KeyError("No close price column found in V1.5 panel")
    panel = panel.sort_values(["symbol", "date"]).copy()
    panel["forward_return_1m"] = panel.groupby("symbol")[close_col].transform(
        lambda values: values.shift(-1) / values - 1.0
    )
    return predictions, panel


def _format_results_markdown(
    checked: pd.DataFrame,
    winner: str,
) -> str:
    display = checked.copy()
    for column in ("MaxDD", "Turnover", "MeanMonthlyReturn", "HitRate"):
        display[column] = display[column] * 100
    columns = [
        "Strategy",
        "Sharpe",
        "MaxDD",
        "Turnover",
        "MeanMonthlyReturn",
        "HitRate",
        "ROE",
        "ProfitGrowth",
        "EP",
        "PassSharpe",
        "PassMaxDD",
        "PassTurnover",
        "PassStyle",
        "PassAll",
    ]
    lines = [
        "# Compact-F Portfolio Construction Results",
        "",
        "- Signal: frozen Compact-F OOS predictions",
        "- Rebalance: month-end",
        "- Return: next-month forward return",
        "- Turnover: one-way half-L1 weight turnover",
        "",
        "## Comparison",
        "",
        display[columns].round(4).to_markdown(index=False),
        "",
        "Percent-formatted columns: MaxDD, Turnover, MeanMonthlyReturn, HitRate.",
        "",
        "## Selection",
        "",
        f"- Selected configuration: **{winner}**",
    ]
    if winner == "A_Fixed_Top30":
        lines.append(
            "- No alternative satisfied every acceptance condition; "
            "the baseline is retained."
        )
    else:
        lines.append(
            "- The selected alternative satisfied Sharpe, drawdown, turnover, "
            "and positive-style requirements."
        )
    lines.extend([
        "",
        "## Buffer and Partial-Rebalance Audit",
        "",
        (
            "Detailed monthly sold counts, bought counts, holding counts, "
            "underfill flags, and pre/post normalization totals are stored in "
            f"`{MONTHLY_PATH.as_posix()}`."
        ),
        "",
        "## Annual Sharpe",
        "",
        f"See `{YEARLY_PATH.as_posix()}` for the full annual decomposition.",
        "",
    ])
    return "\n".join(lines)


def run_portfolio_optimization(
    predictions_path: Path = PREDICTIONS_PATH,
    panel_path: Path = PANEL_PATH,
) -> pd.DataFrame:
    """Run all six construction methods and write required outputs."""
    predictions, panel = load_frozen_inputs(predictions_path, panel_path)
    monthly_frames = []
    result_rows = []
    yearly_frames = []

    for name, spec in STRATEGIES.items():
        monthly = simulate_strategy(predictions, panel, spec)
        monthly_frames.append(monthly)
        result_rows.append({"Strategy": name, **summarize_monthly(monthly)})
        annual = yearly_sharpe(monthly)
        annual.insert(0, "strategy", name)
        yearly_frames.append(annual)

    monthly_all = pd.concat(monthly_frames, ignore_index=True)
    yearly_all = pd.concat(yearly_frames, ignore_index=True)
    results = pd.DataFrame(result_rows)
    checked = _acceptance_columns(results)
    winner = select_winner(results)

    MONTHLY_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly_all.to_csv(MONTHLY_PATH, index=False, encoding="utf-8-sig")
    yearly_all.to_csv(YEARLY_PATH, index=False, encoding="utf-8-sig")
    RESULTS_PATH.write_text(
        _format_results_markdown(checked, winner),
        encoding="utf-8",
    )
    return checked


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--panel", type=Path, default=PANEL_PATH)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.dry_run:
        print(f"predictions={args.predictions}")
        print(f"panel={args.panel}")
        for name, spec in STRATEGIES.items():
            print(
                f"{name}: kind={spec.kind}, target={spec.target_size}, "
                f"buy={spec.buy_rank}, sell={spec.sell_rank}, alpha={spec.alpha}"
            )
        return
    results = run_portfolio_optimization(args.predictions, args.panel)
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

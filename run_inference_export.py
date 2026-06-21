"""
模型批量推理與導出 (Inference & Export)
用法:
  python run_inference_export.py --model-dir output/production_models --panel output/training_panel_v3_full.parquet --output output/predictions_v1.parquet
  python run_inference_export.py --model-dir output/production_models_v2_full --panel output/training_panel_v3_full.parquet --output output/predictions_v2_full.parquet
"""

import argparse, logging, sys, time
from pathlib import Path
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("inference_export")

from factor_research.production_engine import ProductionAlphaEngine

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--panel", default="output/training_panel_v3_full.parquet")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    # Load model
    engine = ProductionAlphaEngine.load_models(args.model_dir)
    logger.info("Model: %d features, %d folds", len(engine._feature_cols), engine._n_folds)

    # Load panel
    panel = pd.read_parquet(args.panel)
    panel["date"] = pd.to_datetime(panel["date"])
    logger.info("Panel: %d rows, %d dates", len(panel), panel["date"].nunique())

    # Determine if panel has _neutral_z or _neutral_z_rank
    nz_cols = [c for c in panel.columns if c.endswith("_neutral_z") and not c.endswith("_rank")]
    rank_cols = [c for c in panel.columns if c.endswith("_neutral_z_rank")]
    has_rank = len(rank_cols) > 0
    use_cols = rank_cols if has_rank else nz_cols
    logger.info("Feature cols: %d (has_rank=%s)", len(use_cols), has_rank)

    # Map panel columns to model features
    col_map = {}
    for fc in engine._feature_cols:
        # Try exact match first
        if fc in panel.columns:
            col_map[fc] = fc
        # Try without _rank suffix
        elif fc.replace("_rank", "") in panel.columns:
            col_map[fc.replace("_rank", "")] = fc
        # Try adding _rank suffix
        elif f"{fc}_rank" in panel.columns:
            col_map[f"{fc}_rank"] = fc
    logger.info("Matched %d/%d features", len(col_map), len(engine._feature_cols))

    # Inference per date
    dates = sorted(panel["date"].unique())
    all_preds = []
    prev_signals = {}

    for i, dt in enumerate(dates):
        mask = panel["date"] == dt
        df = panel[mask]
        syms = df["symbol"].values

        # Build features
        X = pd.DataFrame(index=df.index)
        for panel_col, model_col in col_map.items():
            X[model_col] = df[panel_col].values

        # Compute rank if needed
        if not has_rank:
            for col in X.columns:
                X[col] = X[col].rank(pct=True, na_option="bottom").fillna(0.5)

        # Prev signal
        prev = np.array([prev_signals.get(s, 0.5) for s in syms])

        try:
            signals = engine.predict_cross_section(X, prev_signal=prev)
        except Exception as e:
            logger.warning("Skip %s: %s", str(dt)[:10], e)
            continue

        for j, s in enumerate(syms):
            prev_signals[s] = signals[j]

        pred = df[["date","symbol"]].copy()
        pred["prediction"] = signals
        all_preds.append(pred)

        if (i+1) % 24 == 0 or i == 0:
            logger.info("[%3d/%3d] %s  stocks=%d  signal[%.3f,%.3f]",
                        i+1, len(dates), str(dt)[:10], len(df), signals.min(), signals.max())

    result = pd.concat(all_preds, ignore_index=True)
    result.to_parquet(args.output, index=False)
    logger.info("Saved: %s (%d rows)", args.output, len(result))

if __name__ == "__main__":
    main()

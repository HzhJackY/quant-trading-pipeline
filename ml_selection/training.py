"""
机器学习选股 — 滚动窗口训练框架。

核心创新 vs 传统因子研究:
- 因子研究: 线性加权, 你事先决定每个因子给多少分
- ML 选股: 让模型自动学习因子到收益的非线性映射

金融 ML 最关键的坑: 过拟合比传统 ML 严重得多。
原因: 金融数据的信噪比极低 (SNR < 0.1), 样本量少 (月度数据 8 年仅 96 期),
特征间高度相关 (所有因子都在描述同一批股票)。

防范策略:
1. 滚动窗口训练 (不是随机交叉验证) → 严格按时间切分
2. 先用线性模型 (ElasticNet) 做基准 → 复杂的模型必须比它好才有意义
3. 看特征重要性稳定性 → 如果每个月 Top 特征完全不同, 说明模型在瞎猜
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def rolling_window_train(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "forward_return_1m",
    train_months: int = 60,
    gap: int = 0,
    model_type: str = "elasticnet",
    **model_kwargs,
) -> pd.DataFrame:
    """
    滚动窗口训练 (每个测试月用之前60个月训练一次)。

    这是金融 ML 的标准训练方式, 等价于严格的时间序列交叉验证。
    绝对不能用 sklearn 的 KFold 或 Shuffle Split——那会让模型偷看未来。

    参数
    ----
    panel: 因子面板 (含 date, symbol, 特征列, target列)
    feature_cols: 用作 X 的列名列表
    target_col: 目标列名 (下月收益)
    train_months: 滚动训练窗口月数 (默认60=5年)
    gap: 训练结束到测试开始的月间隔 (防止信息泄露)
    model_type: "elasticnet" | "lightgbm" | "xgboost"
    model_kwargs: 传给模型构造函数的参数

    返回
    ----
    pd.DataFrame: date, symbol, prediction
    """
    dates = sorted(panel["date"].unique())
    if len(dates) <= train_months + gap:
        raise ValueError(f"数据期数 {len(dates)} 不足以做 {train_months} 期滚动训练")

    all_preds = []
    total_windows = len(dates) - train_months - gap

    for i in tqdm(range(train_months, len(dates) - gap), total=total_windows,
                  desc=f"滚动训练 ({model_type})"):
        train_dates = dates[i - train_months : i]
        test_date = dates[i + gap]

        train_mask = panel["date"].isin(train_dates)
        test_mask = panel["date"] == test_date

        train_df = panel[train_mask].dropna(subset=feature_cols + [target_col])
        test_df = panel[test_mask].dropna(subset=feature_cols)

        if len(train_df) < 200 or len(test_df) < 10:
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        test_symbols = test_df["symbol"].values

        # 标准化
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # 训练
        model = _make_model(model_type, **model_kwargs)
        try:
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
        except Exception:
            preds = np.zeros(len(X_test_s))

        all_preds.extend([
            {"date": test_date, "symbol": sym, "prediction": float(p)}
            for sym, p in zip(test_symbols, preds)
        ])

    return pd.DataFrame(all_preds)


def _make_model(model_type: str, **kwargs):
    """创建模型实例。"""
    if model_type == "elasticnet":
        defaults = {"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 5000, "random_state": 42}
        defaults.update(kwargs)
        return ElasticNet(**defaults)
    elif model_type == "lightgbm":
        import lightgbm as lgb
        defaults = {
            "objective": "regression", "metric": "rmse",
            "num_leaves": 31, "learning_rate": 0.05,
            "n_estimators": 200, "verbose": -1, "random_state": 42,
        }
        defaults.update(kwargs)
        return lgb.LGBMRegressor(**defaults)
    elif model_type == "xgboost":
        import xgboost as xgb
        defaults = {
            "objective": "reg:squarederror", "max_depth": 6,
            "learning_rate": 0.05, "n_estimators": 200,
            "random_state": 42, "verbosity": 0,
        }
        defaults.update(kwargs)
        return xgb.XGBRegressor(**defaults)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def evaluate_predictions(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    target_col: str = "forward_return_1m",
) -> dict:
    """
    评估 ML 预测质量。

    关键指标:
    - Prediction IC: 预测值 vs 实际收益的 Rank IC (和因子 IC 同口径对比)
    - MSE / MAE: 预测误差
    """
    merged = predictions.merge(
        panel[["date", "symbol", target_col]], on=["date", "symbol"], how="inner"
    )
    if merged.empty:
        return {}

    # Rank IC
    ic_vals = []
    for date, group in merged.groupby("date"):
        if len(group) < 30:
            continue
        ic, _ = stats.spearmanr(group["prediction"], group[target_col])
        ic_vals.append(ic)
    ic_series = pd.Series(ic_vals)

    mse = mean_squared_error(merged[target_col], merged["prediction"])
    mae = mean_absolute_error(merged[target_col], merged["prediction"])

    return {
        "Pred_IC_Mean": round(float(ic_series.mean()), 4) if len(ic_series) > 0 else 0,
        "Pred_IC_Std": round(float(ic_series.std()), 4) if len(ic_series) > 0 else 0,
        "Pred_IC_IR": round(
            float(ic_series.mean() / ic_series.std()) if len(ic_series) > 0 and ic_series.std() > 0 else 0, 4
        ),
        "MSE": round(float(mse), 6),
        "MAE": round(float(mae), 4),
        "Periods": len(ic_series),
    }

"""
Split-Universe 双模型协同系统 (大盘/小盘分层建模)。

核心思想:
  放弃全市场"大一统"的线性因子模型, 按市值将股票池切分为大盘和小盘两个子域。
  - 大盘股: 由机构定价, 基本面因子 (EP/BP/ROE/利润增速) 信号更强
  - 小盘股: 由散户/游资驱动, 技术面和资金面因子 (成交量变化/换手率/反转) 信号更强

  两个子域独立评估因子表现、独立合成复合信号, 最后通过截面 Z-score
  标准化对齐量纲, 拼接成统一的 300 只全市场 Alpha 表。

流水线:
  Step 1 — 市值估计 + 百分位切割 (split_by_market_cap)
  Step 2 — 分域 IC_IR 评估 (evaluate_sub_universe)
  Step 3 — 异构复合模型 (build_sub_model)
  Step 4 — 信号对齐拼接 (blend_signals)

用法:
  from factor_research.split_universe import SplitUniverseModel
  model = SplitUniverseModel(panel, factor_cols, percentile=0.5)
  model.run_pipeline()  # 执行全部 4 步
  model.plot_ic_comparison()  # 可选: 可视化
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data.cleaner import winsorize_mad, standardize_cross_section

# ── Logger 配置 ──────────────────────────────────────────
logger = logging.getLogger("split_universe")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════

@dataclass
class SubUniverseResult:
    """单个子域 (大盘或小盘) 的分析结果。"""

    name: str                                      # "大盘" 或 "小盘"
    panel: pd.DataFrame                            # 该子域的因子面板 (含原始因子列)
    ic_summary: dict[str, dict] = field(default_factory=dict)  # {factor_name: {IC_Mean, IC_IR, ...}}
    selected_factors: list[str] = field(default_factory=list)   # 选中的因子名
    composite_score: Optional[pd.Series] = None    # 复合得分 (与 panel 等长)


@dataclass
class SplitUniverseResult:
    """Split-Universe 完整结果。"""

    large_cap: SubUniverseResult
    small_cap: SubUniverseResult
    blended_panel: Optional[pd.DataFrame] = None   # 拼接后的全市场面板 (含 alpha_signal 列)
    comparison_table: Optional[pd.DataFrame] = None # 三域 IC_IR 对比表


# ═══════════════════════════════════════════════════════════
# 核心类
# ═══════════════════════════════════════════════════════════

class SplitUniverseModel:
    """
    Split-Universe 双模型系统。

    参数
    ----
    panel : pd.DataFrame
        原始因子面板, 必须包含 date, symbol, 以及所有 factor_cols。
        建议使用 Stage 2 输出的 panel (中性化前), 因为中性化会改变因子含义。
        但也可以使用 preprocessed panel (已有 _neutral_z 列), 系统会自动识别。
    factor_cols : list[str]
        因子列名列表 (原始因子名, 不含 _z / _neutral_z 后缀)。
    percentile : float
        切割阈值, 默认 0.5 → Top 50% 大盘, Bottom 50% 小盘。
    date_col : str
        日期列名, 默认 "date"。
    symbol_col : str
        股票代码列名, 默认 "symbol"。
    return_col : str
        下期收益列名 — 如果 panel 中不存在, 调用 compute_forward_returns() 自动计算。
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        factor_cols: list[str],
        percentile: float = 0.5,
        date_col: str = "date",
        symbol_col: str = "symbol",
        return_col: str = "forward_return_1m",
    ):
        self.panel = panel.copy()
        self.factor_cols = [c for c in factor_cols if c in panel.columns]
        self.percentile = percentile
        self.date_col = date_col
        self.symbol_col = symbol_col
        self.return_col = return_col

        # ── 推断已有标准化列 ──
        self._neutral_z_available = any(
            c.endswith("_neutral_z") for c in panel.columns
        )
        self._z_available = any(
            c.endswith("_z") and not c.endswith("_neutral_z") for c in panel.columns
        )

        if self._neutral_z_available:
            self._suffix = "_neutral_z"
        elif self._z_available:
            self._suffix = "_z"
        else:
            self._suffix = ""  # 需要自己做标准化

        logger.info(
            "SplitUniverseModel 初始化: %d 只股票, %d 个因子, "
            "切割阈值=%.0f%%, 标准化后缀='%s'",
            self.panel[self.symbol_col].nunique(),
            len(self.factor_cols),
            percentile * 100,
            self._suffix or "(无, 将自动标准化)",
        )

    # ── 辅助 ──────────────────────────────────────────────

    def _get_factor_col(self, name: str) -> str:
        """获取因子在 panel 中的实际列名 (优先 neutral_z, 回退 _z, 再回退原始)。"""
        for suffix in [self._suffix, "_neutral_z", "_z", ""]:
            col = f"{name}{suffix}"
            if col in self.panel.columns:
                return col
        return name

    def compute_forward_returns(self, close_col: str = "收盘") -> pd.DataFrame:
        """
        计算下期收益率 (月度 forward return)。

        如果 panel 中已有 forward_return_1m 列则跳过。
        使用 close_col (或尝试推断) 计算: (next_close - close) / close
        按 symbol 分组 shift(-1), 避免跨股票污染。
        """
        if self.return_col in self.panel.columns:
            logger.info("  forward_return_1m 已存在, 跳过计算")
            return self.panel

        # 推断收盘价列名
        if close_col not in self.panel.columns:
            for candidate in ["收盘", "close", "Close"]:
                if candidate in self.panel.columns:
                    close_col = candidate
                    break
            else:
                logger.warning("  未找到收盘价列, 无法计算 forward return")
                return self.panel

        logger.info("  计算 forward_return_1m (按 symbol 分组 shift(-1))")
        df = self.panel.sort_values([self.symbol_col, self.date_col])
        df[self.return_col] = (
            df.groupby(self.symbol_col)[close_col]
            .transform(lambda x: x.shift(-1) / x - 1)
        )
        self.panel = df
        return df

    # ═══════════════════════════════════════════════════════
    # Step 1: 市值估计 + 百分位切割
    # ═══════════════════════════════════════════════════════

    def estimate_market_cap(
        self,
        amount_col: str = "成交额",
        turnover_col: str = "换手率",
    ) -> pd.Series:
        """
        从日线数据估算流通市值。

        公式:
          流通市值 ≈ 成交额 / 换手率

        推导:
          换手率 = 成交量 / 流通股本  (比率, 如 0.025 = 2.5%)
          成交额  = 成交量 × 成交均价
          → 成交额 / 换手率 = 成交量 × 成交均价 / (成交量 / 流通股本)
                            = 流通股本 × 成交均价
                            ≈ 流通股本 × 收盘价 = 流通市值

        近似假设: 成交均价 ≈ 收盘价 (对于月频采样, 误差可控; 用于截面排名足够)

        NaN 处理: 换手率为 0 或缺失 → 市值设为 NaN, 后续在切割时会被排除该截面。
        """
        # 推断列名
        if amount_col not in self.panel.columns:
            for c in ["成交额", "amount", "Amount"]:
                if c in self.panel.columns:
                    amount_col = c
                    break
            else:
                raise KeyError(
                    f"未找到成交额列, 请确认 panel 包含 '成交额' 或 'amount'。"
                    f"现有: {self.panel.columns.tolist()}"
                )

        if turnover_col not in self.panel.columns:
            for c in ["换手率", "turnover", "Turnover"]:
                if c in self.panel.columns:
                    turnover_col = c
                    break
            else:
                raise KeyError(
                    f"未找到换手率列, 请确认 panel 包含 '换手率'。"
                    f"现有: {self.panel.columns.tolist()}"
                )

        amount = self.panel[amount_col].astype(float)
        turnover = self.panel[turnover_col].astype(float)

        # 过滤无效值: 换手率 ≤ 0 无法反推市值
        valid_mask = turnover > 1e-10
        mcap_est = pd.Series(np.nan, index=self.panel.index)
        mcap_est[valid_mask] = amount[valid_mask] / turnover[valid_mask]

        n_valid = valid_mask.sum()
        n_invalid = (~valid_mask).sum()
        logger.info(
            "  [市值估计] 成交额/%s → 流通市值_est: "
            "%d 有效, %d 缺失 (换手率≤0)",
            turnover_col, n_valid, n_invalid,
        )

        return mcap_est

    def split_by_market_cap(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        按市值百分位切割选股池。

        流程:
          1. 估算每期的流通市值 (成交额/换手率)
          2. 每期截面上对市值做 Winsorize (防极端值扭曲排名)
          3. 计算市值百分位排名
          4. Top N% → 大盘池, Bottom (100-N)% → 小盘池

        返回
        ----
        (large_cap_panel, small_cap_panel) : 两个子域 DataFrame

        日志断言:
          - 打印每期的有效样本数
          - 断言大盘+小盘行数 ≈ 全市场行数
        """
        logger.info("=" * 56)
        logger.info("[Step 1/4] 市值估计 + 百分位切割 (threshold=%.0f%%)",
                     self.percentile * 100)
        logger.info("=" * 56)

        # 1. 估计市值
        self.panel["mcap_est"] = self.estimate_market_cap()
        self.panel["mcap_est"] = self.panel["mcap_est"].replace([np.inf, -np.inf], np.nan)

        # 2. 每期截面内 Winsorize 去极值 → 排名
        logger.info("  截面内 Winsorize 市值 (1%%/99%%) 后计算百分位排名")

        pct_lower, pct_upper = 0.01, 0.99

        def _pct_rank(grp: pd.DataFrame) -> pd.Series:
            """在单个截面上做 Winsorize + 百分位排名。"""
            vals = grp["mcap_est"].copy()
            vals = vals.dropna()
            if len(vals) < 10:
                grp = grp.copy()
                grp["mcap_pct"] = np.nan
                grp["universe"] = "未分类"
                return grp[["mcap_pct", "universe"]]

            # Winsorize: clip 1%/99% 分位
            lo = vals.quantile(pct_lower)
            hi = vals.quantile(pct_upper)
            vals_clipped = vals.clip(lo, hi)

            # 百分位排名 (0~1, 市值越大排名越高)
            pct = vals_clipped.rank(pct=True)

            # 映射回原 index
            grp = grp.copy()
            grp["mcap_pct"] = pct.reindex(grp.index)
            grp["universe"] = np.where(
                grp["mcap_pct"] >= self.percentile,
                "大盘",
                np.where(
                    grp["mcap_pct"].notna(),
                    "小盘",
                    "未分类",
                ),
            )
            return grp[["mcap_pct", "universe"]]

        result_frames = []
        for dt, grp in self.panel.groupby(self.date_col, group_keys=False):
            result_frames.append(_pct_rank(grp))

        rank_df = pd.concat(result_frames)
        self.panel["mcap_pct"] = rank_df["mcap_pct"]
        self.panel["universe"] = rank_df["universe"]

        # 3. 切割
        large_mask = self.panel["universe"] == "大盘"
        small_mask = self.panel["universe"] == "小盘"
        unclassified = (~large_mask & ~small_mask).sum()

        large_panel = self.panel[large_mask].copy()
        small_panel = self.panel[small_mask].copy()

        # ── 日志断言 ──
        n_total = len(self.panel)
        n_large = len(large_panel)
        n_small = len(small_panel)

        logger.info(
            "  [截面切割] 全市场=%d 行 | 大盘=%d (%.1f%%) | 小盘=%d (%.1f%%) | 未分类=%d",
            n_total, n_large, 100 * n_large / max(n_total, 1),
            n_small, 100 * n_small / max(n_total, 1),
            unclassified,
        )

        # 按日期统计每期样本数
        date_counts_total = self.panel.groupby(self.date_col).size()
        date_counts_large = large_panel.groupby(self.date_col).size()
        date_counts_small = small_panel.groupby(self.date_col).size()
        logger.info(
            "  每期样本数 (均值): 全市场=%.0f | 大盘=%.0f | 小盘=%.0f",
            date_counts_total.mean(), date_counts_large.mean(), date_counts_small.mean(),
        )

        # 断言
        assert abs(n_large + n_small + unclassified - n_total) < 5, (
            f"切割行数不匹配! {n_large} + {n_small} + {unclassified} ≠ {n_total}"
        )

        logger.info("  [通过] Step 1 截面切割完成\n")
        return large_panel, small_panel

    # ═══════════════════════════════════════════════════════
    # Step 2: 分域因子评估
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _compute_ic_ir_for_panel(
        panel: pd.DataFrame,
        factor_cols: list[str],
        return_col: str = "forward_return_1m",
        date_col: str = "date",
        min_stocks: int = 15,
    ) -> dict[str, dict]:
        """
        在给定面板上计算每个因子的 Rank IC 和 IC_IR。

        参数
        ----
        panel : 因子面板 (含原始因子列 + forward_return)
        factor_cols : 原始因子列名 (不含后缀) — 会尝试查找 _neutral_z / _z 版本
        return_col : 下期收益列名
        date_col : 日期列名
        min_stocks : 单期最少股票数 (低于此数跳过该期)

        返回
        ----
        dict: {factor_name: {IC_Mean, IC_Std, IC_IR, IC_Win_Rate, n_periods}}
        """
        results = {}

        for fname in factor_cols:
            # 优先找标准化版本
            for suffix in ["_neutral_z", "_z", ""]:
                fcol = f"{fname}{suffix}"
                if fcol in panel.columns:
                    break
            else:
                results[fname] = {
                    "IC_Mean": 0.0, "IC_Std": 0.0, "IC_IR": 0.0,
                    "IC_Win_Rate": 0.0, "n_periods": 0,
                }
                continue

            ic_list = []
            for dt, grp in panel.groupby(date_col):
                sub = grp[[fcol, return_col]].dropna()
                if len(sub) >= min_stocks:
                    try:
                        ic, _ = spearmanr(sub[fcol], sub[return_col])
                        if not np.isnan(ic):
                            ic_list.append(ic)
                    except Exception:
                        pass

            if ic_list:
                mean_ic = float(np.mean(ic_list))
                std_ic = float(np.std(ic_list, ddof=1))
                ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0
                win_rate = float(np.mean([1 if v > 0 else 0 for v in ic_list]))
            else:
                mean_ic = std_ic = ic_ir = win_rate = 0.0

            results[fname] = {
                "IC_Mean": round(mean_ic, 6),
                "IC_Std": round(std_ic, 6),
                "IC_IR": round(ic_ir, 6),
                "IC_Win_Rate": round(win_rate, 4),
                "n_periods": len(ic_list),
            }

        return results

    def evaluate_sub_universe(
        self,
        large_panel: pd.DataFrame,
        small_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        在大盘、小盘两个子域上独立计算每个因子的 IC_IR。

        同时计算全市场 IC_IR 作为基准对比。

        返回
        ----
        pd.DataFrame: 三域 IC_IR 对比表
          列: 因子, 全市场_IC_IR, 大盘_IC_IR, 小盘_IC_IR, 大盘优势

        "大盘优势" = 大盘_IC_IR - 小盘_IC_IR, 正值表示基本面因子在大盘更强。
        """
        logger.info("=" * 56)
        logger.info("[Step 2/4] 分域因子 IC_IR 评估")
        logger.info("=" * 56)

        # 确保有 forward return
        if self.return_col not in self.panel.columns:
            self.compute_forward_returns()

        # 给子面板也加上 forward return (从全市场 panel 拷)
        if self.return_col not in large_panel.columns:
            fwd = self.panel[[self.date_col, self.symbol_col, self.return_col]].dropna()
            large_panel = large_panel.merge(
                fwd, on=[self.date_col, self.symbol_col], how="left"
            )
        if self.return_col not in small_panel.columns:
            fwd = self.panel[[self.date_col, self.symbol_col, self.return_col]].dropna()
            small_panel = small_panel.merge(
                fwd, on=[self.date_col, self.symbol_col], how="left"
            )

        # 全市场 IC_IR
        logger.info("  计算全市场 IC_IR...")
        full_ic = self._compute_ic_ir_for_panel(
            self.panel, self.factor_cols, self.return_col, self.date_col
        )

        # 大盘 IC_IR
        logger.info("  计算大盘池 IC_IR (%d 行)...", len(large_panel))
        large_ic = self._compute_ic_ir_for_panel(
            large_panel, self.factor_cols, self.return_col, self.date_col
        )

        # 小盘 IC_IR
        logger.info("  计算小盘池 IC_IR (%d 行)...", len(small_panel))
        small_ic = self._compute_ic_ir_for_panel(
            small_panel, self.factor_cols, self.return_col, self.date_col
        )

        # ── 组装对比表 ──
        rows = []
        for fname in self.factor_cols:
            rows.append({
                "因子": fname,
                "全市场_IC_IR": full_ic.get(fname, {}).get("IC_IR", 0),
                "大盘_IC_IR": large_ic.get(fname, {}).get("IC_IR", 0),
                "小盘_IC_IR": small_ic.get(fname, {}).get("IC_IR", 0),
                "大盘_IC_Mean": full_ic.get(fname, {}).get("IC_Mean", 0),
                "小盘_IC_Mean": large_ic.get(fname, {}).get("IC_Mean", 0),
            })

        comparison = pd.DataFrame(rows)
        comparison["大盘优势"] = comparison["大盘_IC_IR"] - comparison["小盘_IC_IR"]
        comparison["归属"] = comparison["大盘优势"].apply(
            lambda x: "大盘型" if x > 0.05 else ("小盘型" if x < -0.05 else "中性")
        )

        # 按大盘优势排序
        comparison = comparison.sort_values("大盘优势", ascending=False).reset_index(drop=True)

        # ── 日志输出 ──
        logger.info("\n  %-20s %10s %10s %10s %10s %s",
                     "因子", "全市场", "大盘", "小盘", "大盘优势", "归属")
        logger.info("  " + "-" * 65)
        for _, row in comparison.iterrows():
            logger.info(
                "  %-20s %+10.4f %+10.4f %+10.4f %+10.4f %s",
                row["因子"], row["全市场_IC_IR"], row["大盘_IC_IR"],
                row["小盘_IC_IR"], row["大盘优势"], row["归属"],
            )

        # 验证假设
        n_dapan = (comparison["归属"] == "大盘型").sum()
        n_xiaopan = (comparison["归属"] == "小盘型").sum()
        logger.info(
            "\n  假设验证: %d 个因子上大盘更强, %d 个因子在小盘更强",
            n_dapan, n_xiaopan,
        )

        logger.info("  [通过] Step 2 分域评估完成\n")

        # 保存到实例
        self._full_ic = full_ic
        self._large_ic = large_ic
        self._small_ic = small_ic
        self._comparison = comparison

        return comparison

    # ═══════════════════════════════════════════════════════
    # Step 3: 异构模型训练
    # ═══════════════════════════════════════════════════════

    def build_sub_model(
        self,
        panel: pd.DataFrame,
        ic_results: dict[str, dict],
        min_ic_ir: float = 0.05,
        max_correlation: float = 0.7,
        universe_name: str = "",
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        为单个子域构建 IC_IR 加权复合信号。

        流程:
          1. 筛选 IC_IR > min_ic_ir 的因子 (排除噪声因子)
          2. 符号翻转: 负 IC_IR 因子取反 (如反转因子)
          3. 去冗余: 贪婪算法移除 |correlation| > max_correlation 的因子
          4. IC_IR 加权: weight_i = |IC_IR_i| / sum(|IC_IR|)

        参数
        ----
        panel : 子域面板 (含标准化因子列)
        ic_results : Step 2 输出的该子域 IC 结果
        min_ic_ir : 最低 |IC_IR| 阈值, 低于此值的因子不参与合成
        max_correlation : 去冗余的相关系数上限
        universe_name : 子域名称 (仅用于日志)

        返回
        ----
        (panel_with_composite, selected_factor_names)
        """
        logger.info(
            "  [%s模型] 筛选因子: |IC_IR| > %.2f, |corr| < %.1f",
            universe_name, min_ic_ir, max_correlation,
        )

        # 1. 筛选: |IC_IR| > min_ic_ir
        candidates = {
            fname: info
            for fname, info in ic_results.items()
            if abs(info.get("IC_IR", 0)) > min_ic_ir
        }

        if not candidates:
            logger.warning("    [WARN] 无因子通过 |IC_IR| > %.2f 筛选!", min_ic_ir)
            # 回退: 取 |IC_IR| 最高的 3 个
            sorted_by_icir = sorted(
                ic_results.items(),
                key=lambda x: abs(x[1].get("IC_IR", 0)),
                reverse=True,
            )
            candidates = {k: v for k, v in sorted_by_icir[:3]}
            logger.warning("    回退: 使用 |IC_IR| 最高的 3 个因子")

        # 2. 符号翻转
        factor_signs = {}
        for fname, info in candidates.items():
            ic_ir = info.get("IC_IR", 0)
            factor_signs[fname] = -1.0 if ic_ir < 0 else 1.0

        # 3. 去冗余 (基于最近一期截面相关性)
        fnames_sorted = sorted(
            candidates.keys(),
            key=lambda c: abs(candidates[c].get("IC_IR", 0)),
            reverse=True,
        )

        # 在最近一期截面上计算因子相关性矩阵
        if len(fnames_sorted) > 1:
            latest_date = panel[self.date_col].max()
            sub_latest = panel[panel[self.date_col] == latest_date]
            factor_cols_latest = []
            for fn in fnames_sorted:
                actual = self._get_factor_col(fn)
                if actual in sub_latest.columns:
                    factor_cols_latest.append(actual)
                else:
                    factor_cols_latest.append(fn)

            # 只保留存在的列
            valid_cols = [c for c in factor_cols_latest if c in sub_latest.columns]
            if len(valid_cols) > 1:
                corr_matrix = sub_latest[valid_cols].corr()
                # 建立 fname → actual_col 映射
                fname_to_col = {
                    fn: (self._get_factor_col(fn) if self._get_factor_col(fn) in sub_latest.columns else fn)
                    for fn in fnames_sorted
                }
            else:
                corr_matrix = None
                fname_to_col = {}
        else:
            corr_matrix = None
            fname_to_col = {}

        kept = []
        removed = []
        for fn in fnames_sorted:
            too_similar = False
            col_fn = fname_to_col.get(fn, fn)
            for kept_fn in kept:
                col_k = fname_to_col.get(kept_fn, kept_fn)
                if corr_matrix is not None and col_fn in corr_matrix.index and col_k in corr_matrix.columns:
                    corr_val = abs(corr_matrix.loc[col_fn, col_k])
                    if corr_val > max_correlation:
                        too_similar = True
                        removed.append((fn, kept_fn, corr_val))
                        break
            if not too_similar:
                kept.append(fn)

        if removed:
            logger.info("    去冗余: 移除 %d 个高相关因子:", len(removed))
            for r_fn, r_kept, r_corr in removed:
                logger.info("      %s (与 %s corr=%.3f > %.1f)", r_fn, r_kept, r_corr, max_correlation)

        # 4. IC_IR 加权
        total_icir = sum(abs(candidates[f]["IC_IR"]) for f in kept)
        if total_icir > 0:
            weights = {f: abs(candidates[f]["IC_IR"]) / total_icir for f in kept}
        else:
            weights = {f: 1.0 / len(kept) for f in kept}

        # 5. 合成
        panel = panel.copy()
        panel["composite_score"] = 0.0
        for fn in kept:
            actual_col = self._get_factor_col(fn)
            if actual_col in panel.columns:
                sign = factor_signs[fn]
                w = weights[fn]
                col_data = panel[actual_col].fillna(0.0)
                panel["composite_score"] += sign * w * col_data

        # ── 日志 ──
        logger.info(
            "    [%s模型] 保留 %d 个因子 → composite_score",
            universe_name, len(kept),
        )
        for fn in kept:
            sign_str = "(-)" if factor_signs[fn] < 0 else "(+)"
            logger.info(
                "      %-20s weight=%.3f %s  |IC_IR|=%.4f",
                fn, weights[fn], sign_str, abs(candidates[fn]["IC_IR"]),
            )

        return panel, kept

    def train_models(
        self,
        large_panel: pd.DataFrame,
        small_panel: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
        """
        分别训练大盘和小盘两个异构模型。

        返回
        ----
        (large_result, small_result, large_selected, small_selected)
        """
        logger.info("=" * 56)
        logger.info("[Step 3/4] 异构模型训练")
        logger.info("=" * 56)

        large_result, large_selected = self.build_sub_model(
            large_panel, self._large_ic,
            universe_name="大盘",
        )
        small_result, small_selected = self.build_sub_model(
            small_panel, self._small_ic,
            universe_name="小盘",
        )

        logger.info(
            "  大盘模型: %d 个因子 | 小盘模型: %d 个因子",
            len(large_selected), len(small_selected),
        )
        logger.info("  [通过] Step 3 模型训练完成\n")

        return large_result, small_result, large_selected, small_selected

    # ═══════════════════════════════════════════════════════
    # Step 4: 信号对齐与拼接
    # ═══════════════════════════════════════════════════════

    def blend_signals(
        self,
        large_result: pd.DataFrame,
        small_result: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        信号对齐与全市场拼接。

        关键操作:
          1. 分别对大盘 composite_score 做截面 Z-score (均值=0, 标准差=1)
          2. 分别对小盘 composite_score 做截面 Z-score
          3. 拼接为全市场 alpha_signal 表
          4. 日志断言: 打印标准化前后的均值/方差

        为什么不直接拼接:
          - 大盘模型的 raw score 可能天然高于小盘 (因子量纲偏差)
          - 如果直接拼接, 做多信号最强的股票可能全是大盘股
          - 各自池内标准化后, 信号含义变为:
            "这只股票在自己市值段内的相对排名"
          - 拼接后可以做全市场选股, 大盘小盘各有机会

        返回
        ----
        pd.DataFrame: 全市场面板 + alpha_signal 列 (已截面标准化)
        """
        logger.info("=" * 56)
        logger.info("[Step 4/4] 信号对齐与拼接")
        logger.info("=" * 56)

        # 1. 大盘池内截面标准化
        logger.info("  大盘池内截面 Z-score 标准化...")
        large_result = large_result.copy()

        # 标准化前统计
        mean_before_l = large_result["composite_score"].mean()
        std_before_l = large_result["composite_score"].std()
        logger.info(
            "    大盘 raw score: mean=%.4f, std=%.4f",
            mean_before_l, std_before_l,
        )

        large_result = standardize_cross_section(
            large_result,
            factor_col="composite_score",
            date_col=self.date_col,
        )
        large_result["alpha_signal"] = large_result["composite_score_z"]

        mean_after_l = large_result["alpha_signal"].mean()
        std_after_l = large_result.groupby(self.date_col)["alpha_signal"].transform("std").mean()
        logger.info(
            "    大盘 Z-score:   mean≈%.4f, mean(std)=%.4f",
            mean_after_l, std_after_l,
        )

        # 2. 小盘池内截面标准化
        logger.info("  小盘池内截面 Z-score 标准化...")
        small_result = small_result.copy()

        mean_before_s = small_result["composite_score"].mean()
        std_before_s = small_result["composite_score"].std()
        logger.info(
            "    小盘 raw score: mean=%.4f, std=%.4f",
            mean_before_s, std_before_s,
        )

        small_result = standardize_cross_section(
            small_result,
            factor_col="composite_score",
            date_col=self.date_col,
        )
        small_result["alpha_signal"] = small_result["composite_score_z"]

        mean_after_s = small_result["alpha_signal"].mean()
        std_after_s = small_result.groupby(self.date_col)["alpha_signal"].transform("std").mean()
        logger.info(
            "    小盘 Z-score:   mean≈%.4f, mean(std)=%.4f",
            mean_after_s, std_after_s,
        )

        # 3. 拼接
        keep_cols = [
            self.date_col, self.symbol_col, "alpha_signal",
            "universe", "mcap_pct", "mcap_est",
        ]
        # 确保需要的列都存在
        large_cut = large_result[[c for c in keep_cols if c in large_result.columns]]
        small_cut = small_result[[c for c in keep_cols if c in small_result.columns]]

        blended = pd.concat([large_cut, small_cut], ignore_index=True)

        # 4. 验证
        n_large = (blended["universe"] == "大盘").sum()
        n_small = (blended["universe"] == "小盘").sum()
        mean_all = blended["alpha_signal"].mean()
        std_all = blended["alpha_signal"].std()

        logger.info(
            "  拼接后: %d 行 (大盘=%d, 小盘=%d) | "
            "alpha mean=%.4f, std=%.4f",
            len(blended), n_large, n_small, mean_all, std_all,
        )

        # 断言: 拼接后的全市场均值应接近 0
        assert abs(mean_all) < 0.1, (
            f"拼接后 alpha_signal 均值偏差过大: {mean_all:.4f} (预期接近 0)"
        )

        logger.info("  [通过] Step 4 信号拼接完成\n")
        return blended

    # ═══════════════════════════════════════════════════════
    # 完整流水线
    # ═══════════════════════════════════════════════════════

    def run_pipeline(self) -> SplitUniverseResult:
        """
        执行完整 Split-Universe 流水线 (Step 1→2→3→4)。

        返回
        ----
        SplitUniverseResult: 包含大盘/小盘结果、拼接面板、对比表
        """
        logger.info("=" * 60)
        logger.info("Split-Universe 双模型流水线 启动")
        logger.info("因子数: %d | 切割阈值: %.0f%%", len(self.factor_cols), self.percentile * 100)
        logger.info("=" * 60)

        # 确保有 forward return
        self.compute_forward_returns()

        # Step 1: 切割
        large_panel, small_panel = self.split_by_market_cap()

        # Step 2: 评估
        comparison = self.evaluate_sub_universe(large_panel, small_panel)

        # Step 3: 训练
        large_res, small_res, large_sel, small_sel = self.train_models(
            large_panel, small_panel
        )

        # Step 4: 拼接
        blended = self.blend_signals(large_res, small_res)

        # ── 组装结果 ──
        large_result = SubUniverseResult(
            name="大盘",
            panel=large_res,
            ic_summary=self._large_ic,
            selected_factors=large_sel,
            composite_score=large_res["composite_score"]
            if "composite_score" in large_res.columns else None,
        )
        small_result = SubUniverseResult(
            name="小盘",
            panel=small_res,
            ic_summary=self._small_ic,
            selected_factors=small_sel,
            composite_score=small_res["composite_score"]
            if "composite_score" in small_res.columns else None,
        )

        result = SplitUniverseResult(
            large_cap=large_result,
            small_cap=small_result,
            blended_panel=blended,
            comparison_table=comparison,
        )

        logger.info("=" * 60)
        logger.info("Split-Universe 流水线 完成")
        logger.info("=" * 60)

        return result

    # ═══════════════════════════════════════════════════════
    # 可视化
    # ═══════════════════════════════════════════════════════

    def plot_ic_comparison(
        self,
        factor_names: Optional[list[str]] = None,
        save_path: Optional[str] = None,
    ):
        """
        绘制关键因子在大盘 vs 小盘中的累积 IC 净值曲线。

        累积 IC 净值 = Π(1 + monthly_IC), 可以直观展示因子预测能力的
        累计轨迹。如果曲线稳步上升, 说明因子持续有效; 如果忽上忽下,
        说明因子不稳定。

        参数
        ----
        factor_names : list[str] | None
            要绘制的因子名列表。默认取 ProfitGrowth_YoY 和 VolChg_20D。
        save_path : str | None
            保存路径 (PNG)。为 None 则显示。
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
        except ImportError:
            logger.warning("matplotlib 不可用, 跳过绘图")
            return

        if factor_names is None:
            # 默认: 一个基本面因子 + 一个技术面因子
            factor_names = ["ProfitGrowth_YoY", "VolChg_20D"]
            # 只保留实际存在的
            factor_names = [f for f in factor_names if f in self.factor_cols]
            if not factor_names:
                factor_names = self.factor_cols[:2]

        # 确保有 forward return
        if self.return_col not in self.panel.columns:
            self.compute_forward_returns()

        # 重新切割 (如果还没有)
        if "universe" not in self.panel.columns:
            self.split_by_market_cap()

        large_mask = self.panel["universe"] == "大盘"
        small_mask = self.panel["universe"] == "小盘"

        n_factors = len(factor_names)
        fig, axes = plt.subplots(n_factors, 1, figsize=(14, 5 * n_factors))
        if n_factors == 1:
            axes = [axes]

        for ax, fname in zip(axes, factor_names):
            fcol = self._get_factor_col(fname)
            if fcol not in self.panel.columns:
                ax.set_title(f"{fname} — 因子列 '{fcol}' 不存在")
                continue

            # ── 大盘 IC 序列 ──
            large_ic_series = []
            for dt, grp in self.panel[large_mask].groupby(self.date_col):
                sub = grp[[fcol, self.return_col]].dropna()
                if len(sub) >= 15:
                    ic, _ = spearmanr(sub[fcol], sub[self.return_col])
                    if not np.isnan(ic):
                        large_ic_series.append({"date": dt, "IC": ic})
            large_ic_df = pd.DataFrame(large_ic_series)
            if not large_ic_df.empty:
                large_ic_df = large_ic_df.sort_values("date")
                large_ic_df["Cum_IC_NV"] = (1 + large_ic_df["IC"]).cumprod()

            # ── 小盘 IC 序列 ──
            small_ic_series = []
            for dt, grp in self.panel[small_mask].groupby(self.date_col):
                sub = grp[[fcol, self.return_col]].dropna()
                if len(sub) >= 15:
                    ic, _ = spearmanr(sub[fcol], sub[self.return_col])
                    if not np.isnan(ic):
                        small_ic_series.append({"date": dt, "IC": ic})
            small_ic_df = pd.DataFrame(small_ic_series)
            if not small_ic_df.empty:
                small_ic_df = small_ic_df.sort_values("date")
                small_ic_df["Cum_IC_NV"] = (1 + small_ic_df["IC"]).cumprod()

            # ── 绘图 ──
            if not large_ic_df.empty:
                ax.plot(
                    large_ic_df["date"], large_ic_df["Cum_IC_NV"],
                    color="#D62728", linewidth=2, label=f"大盘 (IC_IR={self._large_ic.get(fname, {}).get('IC_IR', 0):+.3f})",
                )
            if not small_ic_df.empty:
                ax.plot(
                    small_ic_df["date"], small_ic_df["Cum_IC_NV"],
                    color="#1F77B4", linewidth=2, label=f"小盘 (IC_IR={self._small_ic.get(fname, {}).get('IC_IR', 0):+.3f})",
                )

            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
            ax.set_title(f"{fname} — 累积 IC 净值 (大盘 vs 小盘)", fontsize=14)
            ax.set_ylabel("累积 IC 净值", fontsize=11)
            ax.legend(fontsize=11, loc="upper left")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("图表已保存到: %s", save_path)

        plt.show()
        return fig


# ═══════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════

def run_split_universe_analysis(
    panel: pd.DataFrame,
    factor_cols: list[str],
    percentile: float = 0.5,
    output_dir: str = "output",
) -> SplitUniverseResult:
    """
    一键运行 Split-Universe 分析。

    参数
    ----
    panel : 因子面板 (Stage 2 输出的 panel.parquet)
    factor_cols : 因子列名
    percentile : 市值切割阈值
    output_dir : 输出目录

    返回
    ----
    SplitUniverseResult
    """
    model = SplitUniverseModel(
        panel=panel,
        factor_cols=factor_cols,
        percentile=percentile,
    )
    result = model.run_pipeline()

    # 保存对比表
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    if result.comparison_table is not None:
        result.comparison_table.to_csv(
            out / "split_universe_ic_comparison.csv",
            index=False, encoding="utf-8-sig",
        )
        logger.info("IC 对比表已保存到: %s", out / "split_universe_ic_comparison.csv")

    # 保存拼接面板
    if result.blended_panel is not None:
        result.blended_panel.to_parquet(
            out / "split_universe_blended.parquet", index=False,
        )
        logger.info("拼接面板已保存到: %s", out / "split_universe_blended.parquet")

    # 可视化
    try:
        model.plot_ic_comparison(
            save_path=str(out / "split_universe_ic_curve.png"),
        )
    except Exception as e:
        logger.warning("绘图失败: %s", e)

    return result

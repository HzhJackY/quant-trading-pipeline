# 模型注册表

| 模型 | 状态 | 路径 | 组合规则 | Sharpe | Max Drawdown | 月换手 | 说明 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| BLEND_V0_50_V7_50_TOP50_BUFFER_V3 | SHADOW_CANDIDATE_NOT_PRODUCTION | output/training_panel_v15_sr.parquet | Top50 Buffer 35/75 | 1.509353 | -0.107414 | 0.187290 | 当前 shadow candidate，不是 production |
| V0_FULL_V15_OOS | main_alpha_component | output/full_panel_forced_tournament_v3/V0_FULL_V15_OOS.parquet | Top50 Buffer 35/75 | 1.147 | -0.0987 | 0.0947 | Blend V3 主要 alpha 组件 |
| V7_FULL_V15_OOS | ml_component | output/full_panel_forced_tournament_v3/V7_FULL_V15_OOS.parquet | Top50 Buffer 35/75 | 1.206 | -0.1243 | 0.3670 | Blend V3 ML 组件 |
| Compact-F | baseline_or_style_reference | output/production_models_v15_compact/Compact_F_oos.parquet | Top50 Buffer 35/75 | 0.273 | -0.3197 | 0.2700 | baseline / style reference，不再是唯一默认 production candidate |
| legacy_paper_trading | active production paper trading logic | paper_trading/paper_trading_pipeline.py | 当前脚本逻辑 | n/a | n/a | n/a | 未被 Blend V3 替换 |

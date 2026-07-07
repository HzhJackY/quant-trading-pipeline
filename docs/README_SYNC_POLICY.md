# README 同步规则

README.md 不再承担实时状态管理，也不是 shadow NAV 或单次 audit 的状态日志。项目当前状态以 `config/project_status.yaml` 为机器可读源，以 `docs/CURRENT_STATUS.md` 为人工可读说明。

## 允许更新 README.md 的情况

1. 主模型身份变化。
2. production / shadow 状态变化。
3. 运行入口变化。
4. 目录结构变化。
5. 数据依赖变化。
6. 重大风险声明变化。

## 不更新 README.md 的情况

1. 单次实验完成。
2. 单次 audit 完成。
3. shadow NAV 单日更新。
4. Codex 修复工程小问题。
5. 中间模型表现变化。
6. CSMAR 初步表清单审计。

## 执行规则

- `scripts/check_readme_consistency.py` 只检查 README 漂移，不自动修改。
- 如检查结果为 `README_ACTION_REQUIRED`，先输出最小修改建议，再由人工确认是否修改。
- 日常状态更新写入 `config/project_status.yaml`、`docs/CURRENT_STATUS.md` 或 `output/blend_v3_governance_patch_v2/*.md`。

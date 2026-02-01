本文档说明 `diff.py` 生成的输出表（默认 `rank_percent_diff_summary.csv`）各列含义与计算方式。

**按季/周分组**
- 每一行对应一个 `Season + Week` 组合的结果。
- 先对同一周内所有选手做计算，再汇总成一行。

**输出字段说明**
- `Season`：赛季编号（来自输入表）。
- `Week`：周次（来自输入表）。
- `Participants`：该周参与人数。
- `Elimination_Event`：是否发生淘汰事件（1=是，0=否）。判断依据为输入表 `Status` 是否包含 “Eliminated”（大小写不敏感）。
- `Elim_Percent`：按“百分比法”计算的最差选手名单（可能多人并列），用 `; ` 分隔。
- `Elim_Rank`：按“排名法”计算的最差选手名单（可能多人并列），用 `; ` 分隔。
- `Elim_Changed`：若该周发生淘汰事件，且 `Elim_Percent` 与 `Elim_Rank` 不同，则为 1，否则为 0；若无淘汰事件固定为 0。
- `Rank_Diff_Mean`：两种方法下最终名次的平均绝对差，按 `(n-1)` 归一化（n 为人数），保留 4 位小数。
- `Diff_Score_Normalized`：综合差异分（见下方），保留 4 位小数。

**两种方法的计算逻辑**
- 百分比法：
  - 裁判分 `JudgeScore` → 归一化成百分比。
  - 观众分优先使用 `Predicted_Audience_Percent`；若缺失则用 `Predicted_Audience_Rank` 转成 `1/rank` 权重再归一化；两者都缺失则均分。
  - 合并百分比：`judge_percent + audience_percent`，再做 dense rank（高分排名更靠前）。
- 排名法：
  - 裁判分做 dense rank（高分排名更靠前）。
  - 观众排名优先使用 `Predicted_Audience_Rank`；若缺失则用观众百分比生成排名。
  - 合并排名：`judge_rank + audience_rank`，再做 dense rank（低分排名更靠前）。

**综合差异分（Diff_Score_Normalized）**
- 先计算 `Rank_Diff_Mean`（两种方法最终排名的平均绝对差）。
- 若该周有淘汰事件：`Rank_Diff_Mean` 权重 1.0，`Elim_Changed` 权重 4.0。
- 若无淘汰事件：`Rank_Diff_Mean` 权重 0.3，`Elim_Changed` 权重 0.0。
- 最终按权重归一化：
  - 有淘汰：`(1.0 * Rank_Diff_Mean + 4.0 * Elim_Changed) / (1.0 + 4.0)`
  - 无淘汰：`(0.3 * Rank_Diff_Mean) / 0.3`

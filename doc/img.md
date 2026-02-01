# 方法流程图（Rank / Percent）

> 规则背景见 `doc/rule.md`；Percent 赛季的损失函数定义见 `doc/loss.md`。

## 0. 总体流程（按赛季规则分支）

```mermaid
flowchart TD
  A[输入数据<br/>- 每周评委打分<br/>- 每周参赛者名单<br/>- 淘汰/退赛结果<br/>- (可选) 搜索热度] --> B{赛季规则类型?}

  B -->|Rank 赛季 (S1-2, S28+)| R0[Monte Carlo<br/>反推观众投票名次]
  B -->|Percent 赛季 (S3-27)| P0[Backprop Optimization<br/>反推观众投票百分比]

  R0 --> O[统一输出（写入结果表）<br/>- Predicted_Audience_Rank / Percent<br/>- Possible_Audience_Vote_Range（区间）]
  P0 --> O

  O --> U[不确定性量化（可选）<br/>区间宽度 + 归一化指标]
```

## 1. Rank 赛季：Monte Carlo（观众投票 Rank + 中心 50% 区间）

筛选条件（与实现一致）：
- **符合实际淘汰规则**：在 Rank 规则下，淘汰者不应“比安全者更好”（合并名次/评委分数 tie-break）  
- **跨周平滑约束**：`Rank_t ≤ Rank_{t-1} + floor(N_t/2)`（`N_t` 为当周参赛人数）

```mermaid
flowchart TD
  R1[按赛季逐周处理] --> R2[提取当周参赛者<br/>计算评委总分与评委名次]

  R2 --> R3{第 1 周?}
  R3 -->|是| R4[随机生成大量观众名次排列<br/>(permutation sampling)]
  R3 -->|否| R5[从上一周粒子抽样作为“父轨迹”]

  R5 --> R6[生成本周观众名次排列<br/>并施加跨周平滑约束<br/>Rank_t ≤ Rank_{t-1}+floor(N_t/2)]

  R4 --> R7[结构有效性筛选<br/>（符合淘汰规则 / tie-break）]
  R6 --> R7

  R7 -->|通过| R8[保留为有效粒子<br/>(trajectory/particle)]
  R7 -->|未通过| R9[丢弃并继续采样]
  R9 --> R4

  R8 --> R10[汇总每位选手的名次分布]
  R10 --> R11[统计量输出<br/>- Mean_Rank / Min_Rank / Max_Rank<br/>- 中心50%区间：P25–P75]

  R11 --> R12[名次→百分比换算<br/>用中心名次构造权重并归一化为100%]
  R12 --> R13[写入结果表<br/>Predicted_Audience_Rank/Percent<br/>Possible_Audience_Vote_Range]
```

## 2. Percent 赛季：机器学习（观众投票 Percent + Loss-bounded 区间）

核心思想：
- 设未知变量为当周观众投票百分比向量 `audience_p`
- 构造总损失 `L_total`（见 `doc/loss.md`），用反向传播找 **最小损失** 的 `audience_p`
- 用阈值 `L_threshold = 2 × L_min`（或一般形式 `k × L_min`）求 **Loss-bounded** 的百分比区间

```mermaid
flowchart TD
  P1[按赛季逐周处理] --> P2[提取当周参赛者<br/>计算评委百分比 judge_p]
  P2 --> P3[构造约束信息<br/>- eliminated_mask / safe_mask<br/>- prev_percent_map（跨周平滑）<br/>- (可选) trend_scores]

  P3 --> P4[初始化 audience_p（可学习变量）]
  P4 --> P5[计算 L_total（加权和）<br/>constraint + smooth + corr + reg + diversity + trend]
  P5 --> P6[梯度下降 / 反向传播更新 audience_p]
  P6 --> P7{收敛/到达步数?}
  P7 -->|否| P5
  P7 -->|是| P8[得到最优解<br/>audience_p* 与 L_min]

  P8 --> P9[区间估计（Loss-bounded）<br/>设 L_threshold = 2 × L_min]
  P9 --> P10[在 L_total ≤ L_threshold 下<br/>对每位选手求 percent 的 min/max]

  P10 --> P11[写入结果表<br/>Predicted_Audience_Percent<br/>Possible_Audience_Vote_Range（%区间）]
```

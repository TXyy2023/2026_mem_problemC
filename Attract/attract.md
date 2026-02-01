# 单变量吸引力效应（Attract）说明

本节说明 `Attract/compute_effects.py` 的计算逻辑与输出含义，用于论文/报告中的“变量效应解释”部分。

## 1. 数据与变量

- 名人属性来自：`2026_MCM_Problem_C_Data.csv`
  - `celebrity_name`
  - `ballroom_partner`
  - `celebrity_industry`
  - `celebrity_homestate`
  - `celebrity_homecountry/region`
  - `celebrity_age_during_season`
- 评委与观众指标来自：`MCM_Problem_C_Results_20260131_2256.csv`
  - **Rank 赛季**：
    - 观众：`Predicted_Audience_Rank`
    - 评委：`JudgeScore_Normalization`
  - **Percent 赛季**：
    - 观众：`Predicted_Audience_Percent`
    - 评委：`JudgeScore_Normalization`

## 1.1 输入 CSV 字段含义

### 2026_MCM_Problem_C_Data.csv（本模型使用字段）
- `celebrity_name`：明星姓名（主键之一）
- `ballroom_partner`：专业舞者/舞伴姓名
- `celebrity_industry`：明星行业类别（如 Athlete、Actor 等）
- `celebrity_homestate`：美国州（非美国可能为空）
- `celebrity_homecountry/region`：国家或地区
- `celebrity_age_during_season`：参赛赛季时年龄（数值）

### MCM_Problem_C_Results_20260131_2256.csv（本模型使用字段）
- `CelebrityName`：明星姓名（与 `celebrity_name` 对齐）
- `Season`：赛季编号
- `Week`：周次
- `RuleType`：规则类型（Rank / Percent）
- `JudgeScore`：当周评委原始总分
- `JudgeScore_Normalization`：当周评委分的归一化/排名值（可能带 % 或数值）
- `Predicted_Audience_Percent`：观众投票百分比预测值（Percent 赛季主用）
- `Predicted_Audience_Rank`：观众投票排名预测值（Rank 赛季主用）

### MCM_Problem_C_Results_20260131_2256.csv（其他字段，作记录与复现用）
- `Possible_Audience_Vote_Range`：模型约束下观众排名可能范围（如 1-6）
- `Loss_Total`：总损失
- `Loss_Constraint` / `Loss_Smooth` / `Loss_Corr` / `Loss_Reg` / `Loss_Diversity` / `Loss_Trend`：损失分项
- `Status`：当周状态（如 Safe 等）
- `Config_*`：优化与模型配置参数（超参数记录）
- `Percent_Predict` / `Rank_predict`：模型输出的预测百分比/排名（可能在部分行为空）

## 2. 归一化与可比性处理

由于 Rank 与 Percent 的尺度不同，且 Rank 越小表示越好，Percent 越大表示越好，本次在计算效应前做了统一归一化处理：

- 以 **(Season, Week, RuleType)** 为组，对该组内的评委与观众指标分别做 **min-max 归一化**。
- **Rank 赛季进行反向处理**：
  - 归一化后取 `1 - norm`，确保“值越大表示越好”。
- 这样得到的 `Judge_Norm` 与 `Audience_Norm` 均落在 [0,1]，可在不同赛季与规则之间比较。

## 3. 聚合与效应定义

1) **周级别归一化 → 名人级别聚合**
- 对每位名人（按 season）聚合所有参赛周：
  - `Judge_Mean`：评委归一化均值
  - `Audience_Mean`：观众归一化均值

2) **单变量效应（详细效应）**
- 对每个变量的每个取值（如某舞伴、某行业、某州、某国、某名人）：
  - 统计该组的 `Judge_Mean` 与 `Audience_Mean` 均值
  - 计算与整体平均的差值（Lift）

> Lift > 0 表示该取值对评委/观众表现有正向提升；Lift < 0 表示相对降低。

## 4. 输出文件说明

输出目录：`Attract/output/`

- `overall_means.csv`
  - Rank 与 Percent 的总体均值基准
- `effects_celebrity_name.csv`
- `effects_ballroom_partner.csv`
- `effects_celebrity_industry.csv`
- `effects_celebrity_homestate.csv`
- `effects_celebrity_homecountry_region.csv`
- `effects_celebrity_age_during_season.csv`
  - 以上文件均包含：
    - 样本数 N
    - 评委/观众归一化均值
    - Lift（相对总体的偏移）
    - 标准差（衡量稳定性）
- `age_linear_effects.csv`
  - 年龄对评委/观众的线性趋势（斜率 + 相关系数）

## 4.1 输出 CSV 字段含义

### overall_means.csv
- `RuleType`：Rank / Percent
- `N`：该规则下的样本数（名人-赛季层级）
- `Mean_Judge_Norm`：评委归一化均值
- `Mean_Audience_Norm`：观众归一化均值

### effects_*.csv（各单变量效应文件）
- `RuleType`：Rank / Percent
- `Value`：该变量的具体取值（如舞伴名、行业、州、国家/地区、名人名、年龄）
- `N`：该取值对应的样本数
- `Mean_Judge_Norm`：该取值下评委归一化均值
- `Mean_Audience_Norm`：该取值下观众归一化均值
- `Judge_Lift`：相对总体的评委偏移（Mean_Judge_Norm - overall）
- `Audience_Lift`：相对总体的观众偏移（Mean_Audience_Norm - overall）
- `Judge_Std` / `Audience_Std`：组内标准差（稳定性）

### age_linear_effects.csv
- `RuleType`：Rank / Percent
- `Target`：Judge / Audience
- `N`：参与拟合的样本量
- `Slope`：年龄每增加 1 岁对归一化评分的线性影响
- `Intercept`：截距
- `Correlation`：年龄与目标的相关系数

## 5. 结果解读建议

- 若某舞伴 `Judge_Lift` 明显为正而 `Audience_Lift` 接近 0，说明舞伴提升技术认可但不一定提升人气。
- 若某行业 `Audience_Lift` 显著为正而评委无差异，说明该行业更受观众偏好。
- 若某国/州的 Lift 为负，可能存在地域偏好不足或受众覆盖弱的趋势。
- 年龄线性结果可用于讨论“年龄偏好/歧视”是否存在。

## 6. 复现命令

```bash
python Attract/compute_effects.py \
  --data 2026_MCM_Problem_C_Data.csv \
  --results MCM_Problem_C_Results_20260131_2256.csv \
  --outdir Attract/output
```

# 损失函数计算说明

本项目对观众投票百分比 `audience_p` 进行反向传播优化，单周总损失为加权和：

```
L_total = α * L_constraint + β * L_smooth + γ * L_corr + δ * L_reg
```

其中权重来自 `PercentLossConfig`：

- α = `alpha_constraint`
- β = `beta_smooth`
- γ = `gamma_corr`
- δ = `delta_reg`

下文给出各项损失的具体计算方式（见 `percent_optimizer.py`）。

## 1. 约束损失 L_constraint

目的：保证“安全组”的总分不低于“淘汰组”，否则惩罚。

定义：

- 总分 `total = audience_p + judge_p`
- `safe_mask` 为安全组布尔掩码，`elim_mask` 为淘汰组布尔掩码
- margin 为 `constraint_margin`

计算：

```
violation = margin - (total_safe - total_elim)
L_constraint = mean( relu(violation)^2 )
```

若安全组或淘汰组为空，返回 0。

## 2. 平滑损失 L_smooth

目的：对跨周出现的选手加入软约束，要求下一周百分比不低于上一周的一半。

定义：

- `prev_percent_tensor` 为上一周的百分比，若不存在则为 -1
- 有效项满足 `prev_percent_tensor >= 0`

计算：

```
allowed = prev_percent_tensor / 2
violation = allowed - audience_p
L_smooth = mean( relu(violation[mask])^2 )
```

若无有效项，返回 0。

## 3. 相关性损失 L_corr

目的：让观众百分比与评委百分比方向一致（最大化相关系数）。

计算：

```
corr = mean((a - mean(a)) * (j - mean(j))) / (std(a) * std(j) + 1e-12)
L_corr = 1 - corr
```

其中 `a = audience_p`, `j = judge_p`。当样本数 < 2 时，返回 0。

## 4. 正则损失 L_reg

目的：让观众百分比分布靠近目标分布（依据软排序得到）。

步骤：

1) 软排序：
```
rank_i = 1 + sum_{j!=i} sigmoid((v_j - v_i) / tau)
```

2) 基于软排序生成目标分布：

- 正态分布型（`reg_type = "normal"`）：
```
mu = (n + 1) / 2
sigma = n * normal_sigma_factor
target_i = exp(-0.5 * ((rank_i - mu) / sigma)^2)
```

- 长尾分布型（`reg_type = "longtail"`）：
```
target_i = 1 / (rank_i + longtail_shift) ^ longtail_alpha
```

最后归一化：
```
target = target / sum(target)
```

3) 用 KL 散度形式计算：
```
L_reg = sum( audience_p * (log(audience_p + eps) - log(target + eps)) )
```

其中 `eps = 1e-12`。
 
## 5. Diversity loss L_diversity (repulsion)

Goal: penalize audience percentages that are too close within the same week.

Definition:
```
L_diversity = mean( exp(-|a_i - a_j| / sigma) ),  i < j
```

Where:
- `a_i` are the audience percentages for the week
- `sigma = diversity_sigma`

Updated total loss:
```
L_total = alpha_constraint * L_constraint
        + beta_smooth * L_smooth
        + gamma_corr * L_corr
        + delta_reg * L_reg
        + epsilon_diversity * L_diversity
```

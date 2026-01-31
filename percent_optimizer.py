from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover - handled by caller
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from tqdm import trange
except Exception:  # pragma: no cover - optional dependency
    trange = None


@dataclass
class PercentLossConfig:
    # Loss weights
    alpha_constraint: float = 50.0
    beta_smooth: float = 1.0
    gamma_corr: float = 1.0
    delta_reg: float = 0.5
    epsilon_diversity: float = 0.05

    # Optimization
    steps: int = 600
    lr: float = 0.08
    temperature: float = 1.0  # softmax temperature

    # Soft-rank
    rank_tau: float = 0.4

    # Regularization distribution
    reg_type: str = "longtail"  # "normal" or "longtail"
    normal_sigma_factor: float = 0.28  # sigma = n * factor
    longtail_alpha: float = 1.6
    longtail_shift: float = 0.8

    # Constraint margin
    constraint_margin: float = 0.0

    # Diversity (repulsion) loss
    diversity_sigma: float = 0.03

    # Device selection: "auto", "cpu", or "cuda"
    device: str = "auto"


def _get_device(config: PercentLossConfig) -> "torch.device":
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Check PyTorch CUDA install.")
    return torch.device(config.device)


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for backprop optimization but is not installed. "
            "Install it (e.g., `pip install torch`) and retry."
        ) from _TORCH_IMPORT_ERROR


def soft_rank(values: "torch.Tensor", tau: float = 1.0) -> "torch.Tensor":
    """
    Differentiable soft rank. Higher values -> better rank (smaller number).
    rank_i = 1 + sum_{j!=i} sigmoid((v_j - v_i) / tau)
    """
    diff = values[None, :] - values[:, None]
    P = torch.sigmoid(diff / tau)
    # Remove self-comparison contribution (sigmoid(0) = 0.5)
    ranks = 1.0 + P.sum(dim=1) - 0.5
    return ranks


def _target_distribution_from_ranks(
    ranks: "torch.Tensor",
    reg_type: str,
    normal_sigma_factor: float,
    longtail_alpha: float,
    longtail_shift: float,
) -> "torch.Tensor":
    n = ranks.shape[0]
    if reg_type == "normal":
        mu = (n + 1) / 2.0
        sigma = max(1e-3, n * normal_sigma_factor)
        weights = torch.exp(-0.5 * ((ranks - mu) / sigma) ** 2)
    elif reg_type == "longtail":
        weights = 1.0 / (ranks + longtail_shift) ** longtail_alpha
    else:
        raise ValueError(f"Unknown reg_type: {reg_type}")
    weights = weights / (weights.sum() + 1e-12)
    return weights


def loss_constraint(
    audience_p: "torch.Tensor",
    judge_p: "torch.Tensor",
    safe_mask: "torch.Tensor",
    elim_mask: "torch.Tensor",
    margin: float = 0.0,
) -> "torch.Tensor":
    if elim_mask.sum() == 0 or safe_mask.sum() == 0:
        return torch.tensor(0.0, device=audience_p.device)
    total = audience_p + judge_p
    total_s = total[safe_mask][:, None]
    total_e = total[elim_mask][None, :]
    violation = margin - (total_s - total_e)
    return torch.relu(violation).pow(2).mean()


def loss_smooth(
    audience_p: "torch.Tensor",
    prev_percent_tensor: "torch.Tensor",
) -> "torch.Tensor":
    """
    Soft rule (percent-based):
    For participants appearing in consecutive weeks,
    enforce: p_{t+1} >= p_t / 2.
    """
    mask = prev_percent_tensor >= 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=audience_p.device)
    allowed = prev_percent_tensor / 2.0
    violation = allowed - audience_p
    return torch.relu(violation[mask]).pow(2).mean()


def loss_corr(audience_p: "torch.Tensor", judge_p: "torch.Tensor") -> "torch.Tensor":
    if audience_p.shape[0] < 2:
        return torch.tensor(0.0, device=audience_p.device)
    a = audience_p - audience_p.mean()
    j = judge_p - judge_p.mean()
    denom = (a.std(unbiased=False) * j.std(unbiased=False)) + 1e-12
    corr = (a * j).mean() / denom
    return 1.0 - corr


def loss_reg(
    audience_p: "torch.Tensor",
    soft_ranks: "torch.Tensor",
    reg_type: str,
    normal_sigma_factor: float,
    longtail_alpha: float,
    longtail_shift: float,
) -> "torch.Tensor":
    target = _target_distribution_from_ranks(
        soft_ranks,
        reg_type=reg_type,
        normal_sigma_factor=normal_sigma_factor,
        longtail_alpha=longtail_alpha,
        longtail_shift=longtail_shift,
    )
    eps = 1e-12
    return (audience_p * (audience_p.add(eps).log() - target.add(eps).log())).sum()


def loss_diversity(audience_p: "torch.Tensor", sigma: float) -> "torch.Tensor":
    """
    Repulsion loss: penalize close audience percentages.
    """
    if audience_p.shape[0] < 2:
        return torch.tensor(0.0, device=audience_p.device)
    diff = torch.abs(audience_p[:, None] - audience_p[None, :])
    # Exclude diagonal and avoid counting pairs twice.
    mask = torch.triu(torch.ones_like(diff, dtype=torch.bool), diagonal=1)
    pair_penalty = torch.exp(-diff / max(sigma, 1e-6))
    return pair_penalty[mask].mean()


def _build_prev_percent_tensor(
    participant_names: List[str],
    prev_percent_map: Dict[str, float],
    device: "torch.device",
) -> "torch.Tensor":
    n = len(participant_names)
    prev_percent_tensor = torch.full((n,), -1.0, dtype=torch.float32, device=device)
    if prev_percent_map:
        for i, name in enumerate(participant_names):
            if name in prev_percent_map:
                prev_percent_tensor[i] = float(prev_percent_map[name])
    return prev_percent_tensor


def _compute_total_loss(
    audience_p: "torch.Tensor",
    judge_p: "torch.Tensor",
    safe_mask: "torch.Tensor",
    elim_mask: "torch.Tensor",
    prev_percent_tensor: "torch.Tensor",
    config: PercentLossConfig,
) -> "torch.Tensor":
    ranks = soft_rank(audience_p, tau=config.rank_tau)
    l_constraint = loss_constraint(
        audience_p, judge_p, safe_mask, elim_mask, margin=config.constraint_margin
    )
    l_smooth = loss_smooth(audience_p, prev_percent_tensor)
    l_corr = loss_corr(audience_p, judge_p)
    l_reg = loss_reg(
        audience_p,
        ranks,
        reg_type=config.reg_type,
        normal_sigma_factor=config.normal_sigma_factor,
        longtail_alpha=config.longtail_alpha,
        longtail_shift=config.longtail_shift,
    )
    l_div = loss_diversity(audience_p, sigma=config.diversity_sigma)
    return (
        config.alpha_constraint * l_constraint
        + config.beta_smooth * l_smooth
        + config.gamma_corr * l_corr
        + config.delta_reg * l_reg
        + config.epsilon_diversity * l_div
    )


def optimize_audience_percent(
    judge_percents: List[float],
    eliminated_mask: List[bool],
    safe_mask: List[bool],
    prev_percent_map: Dict[str, float],
    participant_names: List[str],
    config: PercentLossConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Optimize audience percentages for a single week using backprop.
    Returns (audience_percents, hard_ranks, loss_breakdown).
    """
    _require_torch()

    n = len(judge_percents)
    device = _get_device(config)

    judge_p = torch.tensor(judge_percents, dtype=torch.float32, device=device)
    elim_mask = torch.tensor(eliminated_mask, dtype=torch.bool, device=device)
    safe_mask = torch.tensor(safe_mask, dtype=torch.bool, device=device)

    # Init logits with judge percents for a reasonable starting point.
    init = np.log(np.array(judge_percents, dtype=np.float64) + 1e-6)
    logits = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32, device=device))
    opt = torch.optim.Adam([logits], lr=config.lr)

    # Build previous percent tensor aligned to current participants
    prev_percent_tensor = torch.full((n,), -1.0, dtype=torch.float32, device=device)
    if prev_percent_map:
        for i, name in enumerate(participant_names):
            if name in prev_percent_map:
                prev_percent_tensor[i] = float(prev_percent_map[name])

    step_iter = (
        trange(config.steps, desc="Optimize audience percents", leave=False)
        if trange is not None
        else range(config.steps)
    )
    for step_idx in step_iter:
        aud = torch.softmax(logits / config.temperature, dim=0)
        ranks = soft_rank(aud, tau=config.rank_tau)

        l_constraint = loss_constraint(
            aud, judge_p, safe_mask, elim_mask, margin=config.constraint_margin
        )
        l_smooth = loss_smooth(aud, prev_percent_tensor)
        l_corr = loss_corr(aud, judge_p)
        l_reg = loss_reg(
            aud,
            ranks,
            reg_type=config.reg_type,
            normal_sigma_factor=config.normal_sigma_factor,
            longtail_alpha=config.longtail_alpha,
            longtail_shift=config.longtail_shift,
        )
        l_div = loss_diversity(aud, sigma=config.diversity_sigma)

        total = (
            config.alpha_constraint * l_constraint
            + config.beta_smooth * l_smooth
            + config.gamma_corr * l_corr
            + config.delta_reg * l_reg
            + config.epsilon_diversity * l_div
        )

        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()

        if trange is not None and hasattr(step_iter, "set_postfix"):
            if step_idx % 10 == 0 or step_idx == config.steps - 1:
                step_iter.set_postfix(
                    lr=f"{config.lr:.3g}",
                    temp=f"{config.temperature:.3g}",
                    tau=f"{config.rank_tau:.3g}",
                    loss=f"{total.item():.4f}",
                    c=f"{l_constraint.item():.4f}",
                    s=f"{l_smooth.item():.4f}",
                    corr=f"{l_corr.item():.4f}",
                    reg=f"{l_reg.item():.4f}",
                    div=f"{l_div.item():.4f}",
                )

    with torch.no_grad():
        aud = torch.softmax(logits / config.temperature, dim=0)
        aud_np = aud.cpu().numpy()

    # Hard ranks (dense ranking) for reporting and smoothness carry-over
    order = np.argsort(-aud_np)
    hard_ranks = np.zeros(n, dtype=float)
    current_rank = 1
    prev_val = None
    for idx in order:
        val = aud_np[idx]
        if prev_val is None:
            hard_ranks[idx] = current_rank
        else:
            if not np.isclose(val, prev_val):
                current_rank += 1
            hard_ranks[idx] = current_rank
        prev_val = val

    # Loss breakdown (last iteration values recomputed for reporting)
    with torch.no_grad():
        aud_t = torch.tensor(aud_np, dtype=torch.float32, device=device)
        ranks_t = soft_rank(aud_t, tau=config.rank_tau)
        l_constraint = loss_constraint(
            aud_t, judge_p, safe_mask, elim_mask, margin=config.constraint_margin
        ).item()
        l_smooth = loss_smooth(aud_t, prev_percent_tensor).item()
        l_corr = loss_corr(aud_t, judge_p).item()
        l_reg = loss_reg(
            aud_t,
            ranks_t,
            reg_type=config.reg_type,
            normal_sigma_factor=config.normal_sigma_factor,
            longtail_alpha=config.longtail_alpha,
            longtail_shift=config.longtail_shift,
        ).item()
        l_div = loss_diversity(aud_t, sigma=config.diversity_sigma).item()

    loss_breakdown = {
        "constraint": l_constraint,
        "smooth": l_smooth,
        "corr": l_corr,
        "reg": l_reg,
        "diversity": l_div,
        "total": (
            config.alpha_constraint * l_constraint
            + config.beta_smooth * l_smooth
            + config.gamma_corr * l_corr
            + config.delta_reg * l_reg
            + config.epsilon_diversity * l_div
        ),
    }

    return aud_np, hard_ranks, loss_breakdown


def optimize_audience_percent_ranges_loss_bounded(
    judge_percents: List[float],
    eliminated_mask: List[bool],
    safe_mask: List[bool],
    prev_percent_map: Dict[str, float],
    participant_names: List[str],
    config: PercentLossConfig,
    base_audience_percents: np.ndarray,
    min_total_loss: float,
    loss_slack_ratio: float = None,
    loss_threshold_multiplier: float = 1.5,
    steps: int = 250,
    lr: float = None,
    penalty_base: float = 50.0,
    penalty_growth: float = 10.0,
    attempts: int = 3,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute min/max audience percents for each participant while enforcing
    total_loss <= min_total_loss * (1 + loss_slack_ratio).
    Returns (min_percents, max_percents, loss_threshold).
    """
    _require_torch()

    n = len(judge_percents)
    if n == 0:
        return np.array([]), np.array([]), 0.0

    device = _get_device(config)
    judge_p = torch.tensor(judge_percents, dtype=torch.float32, device=device)
    elim_mask = torch.tensor(eliminated_mask, dtype=torch.bool, device=device)
    safe_mask_t = torch.tensor(safe_mask, dtype=torch.bool, device=device)
    prev_percent_tensor = _build_prev_percent_tensor(
        participant_names=participant_names,
        prev_percent_map=prev_percent_map,
        device=device,
    )

    base_audience = np.clip(np.asarray(base_audience_percents, dtype=np.float64), 1e-8, 1.0)
    base_audience = base_audience / base_audience.sum()
    base_logits = np.log(base_audience)

    if loss_threshold_multiplier is None:
        slack = 0.0 if loss_slack_ratio is None else float(loss_slack_ratio)
        loss_threshold_multiplier = 1.0 + slack
    else:
        loss_threshold_multiplier = float(loss_threshold_multiplier)

    loss_threshold = float(min_total_loss) * loss_threshold_multiplier
    lr = float(config.lr if lr is None else lr)

    def _optimize_target(target_idx: int, direction: str) -> float:
        best_val = None
        best_violation = float("inf")

        for attempt in range(attempts):
            penalty = penalty_base * (penalty_growth ** attempt)
            logits = torch.nn.Parameter(
                torch.tensor(base_logits, dtype=torch.float32, device=device)
            )
            opt = torch.optim.Adam([logits], lr=lr)

            for _ in range(steps):
                aud = torch.softmax(logits / config.temperature, dim=0)
                total_loss = _compute_total_loss(
                    aud, judge_p, safe_mask_t, elim_mask, prev_percent_tensor, config
                )
                violation = torch.relu(total_loss - loss_threshold)
                if direction == "min":
                    obj = aud[target_idx] + penalty * violation
                else:
                    obj = -aud[target_idx] + penalty * violation

                opt.zero_grad(set_to_none=True)
                obj.backward()
                opt.step()

            with torch.no_grad():
                aud = torch.softmax(logits / config.temperature, dim=0)
                total_loss = _compute_total_loss(
                    aud, judge_p, safe_mask_t, elim_mask, prev_percent_tensor, config
                )
                violation = max(0.0, total_loss.item() - loss_threshold)
                candidate = aud[target_idx].item()

            if violation <= 1e-6:
                if best_val is None:
                    best_val = candidate
                else:
                    if direction == "min":
                        best_val = min(best_val, candidate)
                    else:
                        best_val = max(best_val, candidate)
            elif violation < best_violation:
                best_violation = violation
                best_val = candidate

        return float(best_val) if best_val is not None else float(base_audience[target_idx])

    min_vals = np.zeros(n, dtype=float)
    max_vals = np.zeros(n, dtype=float)
    target_iter = trange(n, desc="Range opt", leave=False) if trange is not None else range(n)
    for i in target_iter:
        min_vals[i] = _optimize_target(i, "min")
        max_vals[i] = _optimize_target(i, "max")

    return min_vals, max_vals, loss_threshold

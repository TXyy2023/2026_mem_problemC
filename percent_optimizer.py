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


@dataclass
class PercentLossConfig:
    # Loss weights
    alpha_constraint: float = 50.0
    beta_smooth: float = 1.0
    gamma_corr: float = 1.0
    delta_reg: float = 0.5

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
    device = torch.device("cpu")

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

    for _ in range(config.steps):
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

        total = (
            config.alpha_constraint * l_constraint
            + config.beta_smooth * l_smooth
            + config.gamma_corr * l_corr
            + config.delta_reg * l_reg
        )

        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()

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

    loss_breakdown = {
        "constraint": l_constraint,
        "smooth": l_smooth,
        "corr": l_corr,
        "reg": l_reg,
        "total": (
            config.alpha_constraint * l_constraint
            + config.beta_smooth * l_smooth
            + config.gamma_corr * l_corr
            + config.delta_reg * l_reg
        ),
    }

    return aud_np, hard_ranks, loss_breakdown

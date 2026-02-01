import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np


DEFAULT_INPUT = "MCM_Problem_C_Results_20260201_1348.csv"
DEFAULT_OUTPUT = "rank_percent_diff_summary.csv"


@dataclass
class WeightConfig:
    rank_weight: float = 1.0
    elim_weight: float = 4.0
    no_elim_rank_weight: float = 0.3


def dense_rank(values: pd.Series, higher_better: bool = True) -> pd.Series:
    """
    Dense rank: 1,1,2,3...
    """
    ascending = not higher_better
    return values.rank(method="dense", ascending=ascending)


def normalize_percent(percent: pd.Series) -> pd.Series:
    total = percent.sum(skipna=True)
    if total <= 0:
        return percent
    return percent / total * 100.0


def derive_audience_percent(group: pd.DataFrame) -> pd.Series:
    pct = pd.to_numeric(group.get("Predicted_Audience_Percent"), errors="coerce")
    if pct.isna().all():
        rank = pd.to_numeric(group.get("Predicted_Audience_Rank"), errors="coerce")
        if rank.isna().all():
            return pd.Series([100.0 / len(group)] * len(group), index=group.index)
        weight = 1.0 / rank
        return normalize_percent(weight)
    if pct.isna().any():
        rank = pd.to_numeric(group.get("Predicted_Audience_Rank"), errors="coerce")
        if rank.notna().any():
            weight = 1.0 / rank
            fill_pct = normalize_percent(weight)
            pct = pct.fillna(fill_pct)
        else:
            pct = pct.fillna(0.0)
    return normalize_percent(pct)


def derive_audience_rank(group: pd.DataFrame, audience_percent: pd.Series) -> pd.Series:
    rank = pd.to_numeric(group.get("Predicted_Audience_Rank"), errors="coerce")
    if rank.isna().all():
        return dense_rank(audience_percent, higher_better=True)
    if rank.isna().any():
        fill_rank = dense_rank(audience_percent, higher_better=True)
        rank = rank.fillna(fill_rank)
    return rank


def pick_elimination_set(names: pd.Series, totals: pd.Series, higher_better: bool) -> List[str]:
    if higher_better:
        worst_val = totals.min()
    else:
        worst_val = totals.max()
    elim = names[totals == worst_val].tolist()
    return sorted(elim)


def analyze_group(group: pd.DataFrame, weights: WeightConfig) -> Dict[str, object]:
    n = len(group)
    names = group["CelebrityName"]

    judge_scores = pd.to_numeric(group["JudgeScore"], errors="coerce").fillna(0.0)
    judge_percent = normalize_percent(judge_scores)
    judge_rank = dense_rank(judge_scores, higher_better=True)

    audience_percent = derive_audience_percent(group)
    audience_rank = derive_audience_rank(group, audience_percent)

    percent_total = judge_percent + audience_percent
    rank_total = judge_rank + audience_rank

    rank_percent = dense_rank(percent_total, higher_better=True)
    rank_rank = dense_rank(rank_total, higher_better=False)

    if n <= 1:
        rank_diff = 0.0
    else:
        rank_diff = (rank_percent.sub(rank_rank).abs().mean()) / (n - 1)

    status = group.get("Status", pd.Series([""] * n, index=group.index))
    elimination_event = status.astype(str).str.contains("Eliminated", case=False, na=False).any()

    elim_percent = pick_elimination_set(names, percent_total, higher_better=True)
    elim_rank = pick_elimination_set(names, rank_total, higher_better=False)

    elim_changed = int(elim_percent != elim_rank) if elimination_event else 0

    if elimination_event:
        rank_weight = weights.rank_weight
        elim_weight = weights.elim_weight
    else:
        rank_weight = weights.no_elim_rank_weight
        elim_weight = 0.0

    denom = rank_weight + elim_weight
    if denom <= 0:
        score = 0.0
    else:
        score = (rank_weight * rank_diff + elim_weight * elim_changed) / denom

    return {
        "Season": int(group["Season"].iloc[0]),
        "Week": int(group["Week"].iloc[0]),
        "Participants": n,
        "Elimination_Event": int(elimination_event),
        "Elim_Percent": "; ".join(elim_percent),
        "Elim_Rank": "; ".join(elim_rank),
        "Elim_Changed": int(elim_changed),
        "Rank_Diff_Mean": round(float(rank_diff), 4),
        "Diff_Score_Normalized": round(float(score), 4),
    }


def analyze(input_path: str, output_path: str, weights: WeightConfig) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    required = {"CelebrityName", "Season", "Week", "JudgeScore"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    results = []
    for (_, _), group in df.groupby(["Season", "Week"], sort=True):
        if group.empty:
            continue
        results.append(analyze_group(group, weights))

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    return out_df


def main():
    parser = argparse.ArgumentParser(
        description="Compare Rank vs Percent voting outcomes with a normalized difference score."
    )
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT, help="Input CSV file")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Output CSV file")
    parser.add_argument("--rank-weight", type=float, default=1.0, help="Weight for rank difference")
    parser.add_argument("--elim-weight", type=float, default=4.0, help="Weight for elimination change")
    parser.add_argument(
        "--no-elim-rank-weight",
        type=float,
        default=0.3,
        help="Rank weight when no elimination event in that week",
    )
    args = parser.parse_args()

    weights = WeightConfig(
        rank_weight=args.rank_weight,
        elim_weight=args.elim_weight,
        no_elim_rank_weight=args.no_elim_rank_weight,
    )

    out_df = analyze(args.input, args.output, weights)
    overall = out_df["Diff_Score_Normalized"].mean() if not out_df.empty else 0.0
    print(f"Wrote: {args.output}")
    print(f"Overall normalized difference score: {overall:.4f}")


if __name__ == "__main__":
    main()

import argparse
import os

import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None
from calculate_audience_votes import (
    INPUT_FILE,
    OUTPUT_PREFIX,
    PERCENT_LOSS_CONFIG,
    get_weekly_participants,
    parse_result_status,
)
from percent_optimizer import optimize_audience_percent_ranges_loss_bounded


def _select_prev_output_file():
    candidates = []
    for name in os.listdir("."):
        if name.startswith(f"{OUTPUT_PREFIX}_") and name.endswith(".csv"):
            candidates.append(name)
    if not candidates:
        return None
    return max(candidates, key=lambda p: os.path.getmtime(p))


def _build_ml_map(ml_df: pd.DataFrame) -> dict:
    ml_map = {}
    for _, row in ml_df.iterrows():
        key = (
            row.get("CelebrityName"),
            int(row.get("Season")),
            int(row.get("Week")),
            row.get("RuleType"),
        )
        ml_map[key] = {
            "Predicted_Audience_Percent": row.get("Predicted_Audience_Percent"),
            "Loss_Total": row.get("Loss_Total"),
        }
    return ml_map


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def _process_season_percent_range(
    season: int,
    df: pd.DataFrame,
    score_cols: list,
    ml_map: dict,
    loss_threshold_multiplier: float,
    show_week_progress: bool = False,
) -> dict:
    rule_type = "Percent"
    if season <= 2 or season >= 28:
        rule_type = "Rank"
    if rule_type != "Percent":
        return {}

    prev_percent_map = {}
    weeks = set()
    for c in score_cols:
        try:
            week_num = int(c.split("_")[0].replace("week", ""))
            weeks.add(week_num)
        except Exception:
            pass
    sorted_weeks = sorted(list(weeks))
    last_week = sorted_weeks[-1] if sorted_weeks else None

    range_map = {}
    week_iter = (
        tqdm(sorted_weeks, desc=f"Season {season} Weeks", leave=False)
        if (show_week_progress and tqdm is not None)
        else sorted_weeks
    )
    for week in week_iter:
        participants = get_weekly_participants(df, season, week, score_cols, last_week)
        if not participants:
            continue

        total_J = sum(p["total_score"] for p in participants)
        judge_percents = [
            (p["total_score"] / total_J) if total_J > 0 else 0.0 for p in participants
        ]
        eliminated_mask = [p["is_eliminated_this_week"] for p in participants]
        safe_mask = [
            (not p["is_eliminated_this_week"]) and p["status"] != "Withdrew"
            for p in participants
        ]
        names = [p["name"] for p in participants]

        # Build base audience percents and loss from ML outputs
        base_audience = []
        week_loss_total = None
        for name in names:
            key = (name, int(season), int(week), rule_type)
            cached = ml_map.get(key)
            if cached:
                base_audience.append(float(cached.get("Predicted_Audience_Percent", 0.0)) / 100.0)
                if week_loss_total is None:
                    val = cached.get("Loss_Total")
                    week_loss_total = None if val is None else float(val)
            else:
                base_audience.append(0.0)

        if week_loss_total is None:
            print(f"Warning: missing Loss_Total for season {season} week {week}; skip range opt.")
            continue

        base_sum = sum(base_audience)
        if base_sum <= 0:
            print(f"Warning: missing Predicted_Audience_Percent for season {season} week {week}; skip.")
            continue
        base_audience = [x / base_sum for x in base_audience]

        week_range_min, week_range_max, _ = optimize_audience_percent_ranges_loss_bounded(
            judge_percents=judge_percents,
            eliminated_mask=eliminated_mask,
            safe_mask=safe_mask,
            prev_percent_map=prev_percent_map,
            participant_names=names,
            config=PERCENT_LOSS_CONFIG,
            base_audience_percents=base_audience,
            min_total_loss=week_loss_total,
            loss_threshold_multiplier=loss_threshold_multiplier,
        )

        for idx_p, name in enumerate(names):
            min_p = week_range_min[idx_p] * 100.0
            max_p = week_range_max[idx_p] * 100.0
            range_map[(name, int(season), int(week), rule_type)] = f"{min_p:.1f}%-{max_p:.1f}%"

        prev_percent_map = {names[i]: float(base_audience[i]) for i in range(len(names))}

    return range_map


def main(
    ml_input: str = None,
    loss_threshold_multiplier: float = 1.5,
    season_workers: int = 1,
    force_parallel: bool = False,
):
    if ml_input is None:
        ml_input = _select_prev_output_file()
    if ml_input is None or not os.path.exists(ml_input):
        raise FileNotFoundError("No ML output CSV found. Provide --ml-input.")

    print(f"Loading raw data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    score_cols = [c for c in df.columns if "judge" in c and "score" in c and "week" in c]
    df = parse_result_status(df, score_cols)

    print(f"Loading ML output from {ml_input}...")
    ml_df = pd.read_csv(ml_input)
    ml_map = _build_ml_map(ml_df)

    seasons = sorted(df["season"].unique())
    range_map = {}

    if season_workers is None or int(season_workers) < 1:
        season_workers = 1

    if season_workers > 1:
        use_cuda = False
        if PERCENT_LOSS_CONFIG.device == "cuda":
            use_cuda = True
        elif PERCENT_LOSS_CONFIG.device == "auto" and _cuda_available():
            use_cuda = True
        if use_cuda:
            if force_parallel:
                print("Warning: CUDA enabled; running multi-process on a single GPU may be slower or OOM.")
            else:
                print("Info: CUDA enabled; forcing --season-workers=1 to avoid multi-process GPU contention.")
                season_workers = 1

    if season_workers == 1:
        season_iter = tqdm(seasons, desc="Seasons", leave=True) if tqdm is not None else seasons
        for season in season_iter:
            season_map = _process_season_percent_range(
                season=season,
                df=df,
                score_cols=score_cols,
                ml_map=ml_map,
                loss_threshold_multiplier=loss_threshold_multiplier,
                show_week_progress=True,
            )
            range_map.update(season_map)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=int(season_workers)) as ex:
            futures = {
                ex.submit(
                    _process_season_percent_range,
                    season,
                    df,
                    score_cols,
                    ml_map,
                    loss_threshold_multiplier,
                ): season
                for season in seasons
            }
            season_progress = tqdm(total=len(seasons), desc="Seasons", leave=True) if tqdm is not None else None
            for fut in as_completed(futures):
                range_map.update(fut.result())
                if season_progress is not None:
                    season_progress.update(1)
            if season_progress is not None:
                season_progress.close()

    # Update Possible_Audience_Vote_Range in ML output
    for i, row in ml_df.iterrows():
        key = (
            row.get("CelebrityName"),
            int(row.get("Season")),
            int(row.get("Week")),
            row.get("RuleType"),
        )
        new_range = range_map.get(key)
        if new_range is not None:
            ml_df.at[i, "Possible_Audience_Vote_Range"] = new_range

    out_name = ml_input
    ml_df.to_csv(out_name, index=False)
    print(f"Percent range results saved to {out_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute percent-range from ML output.")
    parser.add_argument("--ml-input", dest="ml_input", type=str, default=None, help="Path to ML output CSV.")
    parser.add_argument(
        "--loss-threshold-multiplier",
        dest="loss_threshold_multiplier",
        type=float,
        default=1,
        help="Loss threshold multiplier for loss-bounded percent ranges.",
    )
    parser.add_argument(
        "--season-workers",
        dest="season_workers",
        type=int,
        default=1,
        help="Parallel workers for season-level processing (CPU-side).",
    )
    parser.add_argument(
        "--force-parallel",
        dest="force_parallel",
        action="store_true",
        help="Allow multi-process even when CUDA is enabled (may be slower or OOM).",
    )
    args = parser.parse_args()
    main(
        ml_input=args.ml_input,
        loss_threshold_multiplier=float(args.loss_threshold_multiplier),
        season_workers=int(args.season_workers),
        force_parallel=bool(args.force_parallel),
    )

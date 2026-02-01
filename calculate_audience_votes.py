import pandas as pd
import numpy as np
import warnings
import re
from datetime import datetime

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from percent_optimizer import (
    PercentLossConfig,
    optimize_audience_percent,
)

warnings.filterwarnings('ignore')

INPUT_FILE = '2026_MCM_Problem_C_Data.csv'
OUTPUT_PREFIX = 'MCM_Problem_C_Results'
OUTPUT_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M')
OUTPUT_FILE = f'{OUTPUT_PREFIX}_{OUTPUT_TIMESTAMP}.csv'

# Percent-rule optimization config (loss modularized in percent_optimizer.py)
PERCENT_LOSS_CONFIG = PercentLossConfig(
    alpha_constraint=240.0,  # large penalty for violating elimination constraints
    beta_smooth=6.0,
    gamma_corr=5.0,
    delta_reg=5,
    epsilon_diversity=5,
    steps=2000,
    lr=0.005,
    temperature=0.8,
    rank_tau=0.45,
    reg_type="longtail",
    normal_sigma_factor=0.28,
    longtail_alpha=1.5,
    longtail_shift=0.8,
    constraint_margin=0.0,
    diversity_sigma=0.03,
    device="cuda",
)

def build_week_cols_map(score_cols):
    week_cols_map = {}
    for c in score_cols:
        try:
            week_num = int(c.split('_')[0].replace('week', ''))
        except Exception:
            continue
        week_cols_map.setdefault(week_num, []).append(c)
    return week_cols_map

def get_last_scored_week(row, week_cols_map):
    last_week = None
    for week_num in sorted(week_cols_map.keys()):
        week_scores = []
        for c in week_cols_map[week_num]:
            val = row.get(c)
            try:
                val = float(val)
                if not pd.isna(val) and val > 0:
                    week_scores.append(val)
            except Exception:
                continue
        if week_scores:
            last_week = week_num
    return last_week

def parse_result_status(df, score_cols):
    """
    Parse 'results' column to determine Exit Week for each participant.
    """
    # Initialize ExitWeek with a large number (Survived)
    df['ExitWeek'] = 999 
    df['Status'] = 'Safe'
    df['ElimRank'] = None
    week_cols_map = build_week_cols_map(score_cols)
    
    for idx, row in df.iterrows():
        res = str(row['results']).lower()
        placement_raw = row.get('placement')
        try:
            placement = int(float(placement_raw)) if placement_raw is not None and str(placement_raw).strip() != "" else None
        except Exception:
            placement = None
        if 'eliminated week' in res:
            try:
                week = int(re.search(r'week (\d+)', res).group(1))
                df.at[idx, 'ExitWeek'] = week
                df.at[idx, 'Status'] = 'Eliminated'
                if placement is not None:
                    df.at[idx, 'ElimRank'] = placement
            except:
                pass
        elif 'withdrew' in res:
            # Need to find when they withdrew.
            # Heuristic: Check last week with scores.
            # We will handle this by checking scores later.
            df.at[idx, 'Status'] = 'Withdrew' 
            # We'll set ExitWeek dynamically based on missing scores.
        elif 'place' in res:
            # Finalist: use placement to determine elimination in final week
            last_week = get_last_scored_week(row, week_cols_map)
            if last_week is not None:
                df.at[idx, 'ExitWeek'] = last_week
            if placement == 1:
                df.at[idx, 'Status'] = 'Winner'
            else:
                df.at[idx, 'Status'] = 'Eliminated'
                if placement is not None:
                    df.at[idx, 'ElimRank'] = placement
            
    return df

def get_weekly_participants(df, season, week, score_cols, last_week):
    """
    Get participants who danced this week.
    Returns list of dicts with info.
    """
    season_df = df[df['season'] == season]
    week_cols = [c for c in score_cols if c.startswith(f'week{week}_')]
    
    participants = []
    
    for idx, row in season_df.iterrows():
        scores = []
        for c in week_cols:
            val = row[c]
            try:
                val = float(val)
                if not pd.isna(val) and val > 0:
                    scores.append(val)
            except:
                pass
        
        if scores:
            total_score = sum(scores)
            p_data = {
                'id': idx,
                'name': row['celebrity_name'],
                'total_score': total_score,
                'status': row['Status'],
                'exit_week': row['ExitWeek'],
                'actual_result': row['results'],
                'placement': row.get('placement'),
                'elim_rank': row.get('ElimRank')
            }
            
            # Determine outcome for THIS week
            # If ExitWeek == week -> Eliminated
            # If ExitWeek > week -> Safe
            # If Withdrew: Check if they have scores next week?
            # Usually Withdrew people dance then leave, or leave before dancing. 
            # If they have scores this week, they competed.
            # If they withdrew this week, did they take an elimination spot?
            # Usually Withdrew == No Elimination, or Elimination proceeds?
            # We treat Withdrew as "Neutral" (Not S, Not E) for constraints?
            # Or if they withdrew, they are E?
            
            # Refined Logic:
            # E = {p | ExitWeek == week AND Status == Eliminated}
            # S = {p | ExitWeek > week OR Status in [Winner, ...]}
            
            is_eliminated = (p_data['exit_week'] == week and p_data['status'] == 'Eliminated')
            # Check for finalists outcome
            if 'place' in str(p_data['actual_result']).lower() and week == last_week:
                 # Final week logic
                 pass
            
            p_data['is_eliminated_this_week'] = is_eliminated
            
            # Handle Withdrew logic dynamically
            # If 'Withdrew' and this is their last week of scores (assumed if ExitWeek is not set to K)
            # Actually, standard logic: E vs S constraints only apply to E and S.
            # Withdrew people are ignored for constraints.
            
            participants.append(p_data)
            
    return participants

def solve_rank_rule_inverse(target_p, participants, n_participants):
    """
    Find valid range [MinA, MaxA] for target_p.
    Constraint: S beats E.
    Total(s) <= Total(e).
    """
    S = [p for p in participants if not p['is_eliminated_this_week'] and p['status'] != 'Withdrew']
    E = [p for p in participants if p['is_eliminated_this_week']]
    
    # If no elimination, range is 1-N
    if not E:
        return 1, n_participants
    
    # If everyone eliminated? (Final week?)
    # Usually Final week is Top Place vs Lower Place.
    # Keep simple: If E exists, constraints exist.
    
    # Calculate Judge Ranks
    # Sort participants by Score Descending
    sorted_p = sorted(participants, key=lambda x: x['total_score'], reverse=True)
    # Assign Ranks
    for i, p in enumerate(sorted_p):
        # Handle ties
        if i > 0 and p['total_score'] == sorted_p[i-1]['total_score']:
            p['j_rank'] = sorted_p[i-1]['j_rank']
        else:
            p['j_rank'] = i + 1
            
    # Solve for target_p
    # Iterate all possible ranks a in 1..N
    valid_a = []
    
    # Determine target's role (S or E)
    target_role = 'S' if target_p in S else ('E' if target_p in E else 'Ignore')
    if target_role == 'Ignore':
        return 1, n_participants # Withdrew, no constraints on them
        
    for a_test in range(1, n_participants + 1):
        # Can target_p have Audience Rank a_test?
        # Construct pool
        pool = list(range(1, n_participants + 1))
        pool.remove(a_test)
        
        # We need to assign pool to Others (S+E excluding target)
        # To satisfy max(Total(S)) <= min(Total(E))
        # Greedy Assignment:
        # Give S the BEST ranks (Smallest numbers) from pool.
        # Give E the WORST ranks (Largest numbers) from pool.
        # If this BEST CASE assignment fails, then Impossible.
        
        # S_others: S excluding target (if target in S)
        # E_others: E excluding target (if target in E)
        S_others = [p for p in S if p != target_p]
        E_others = [p for p in E if p != target_p]
        
        # Sort S_others by J_rank (Asc)? Doesn't matter for sets, we just need to pair.
        # Actually it DOES matter.
        # To minimize max(Total(S)), we should pair Worst J_rank with Best A_rank?
        # max(J+A). To minimize max sum, match Large J with Small A. (Reverse sort).
        
        pool_asc = sorted(pool)
        
        # Assign to S_others first (Smallest A's) to help them verify "S <= E".
        # Minimize max(S):
        # S_others sorted by J desc (Worst J first).
        # Pair with Best A (Smallest).
        
        # Check S totals
        temp_S_totals = []
        
        if target_p in S:
            temp_S_totals.append(target_p['j_rank'] + a_test)
            
        S_others_sorted = sorted(S_others, key=lambda x: x['j_rank'], reverse=True) # Large J first
        # Take needed number of A's
        num_s = len(S_others)
        s_ranks = pool_asc[:num_s]
        remaining_pool = pool_asc[num_s:]
        
        for k, p in enumerate(S_others_sorted):
            temp_S_totals.append(p['j_rank'] + s_ranks[k])
            
        # Assign to E_others (Largest A's) to help them (maximize E total) verify "S <= E"?
        # Wait constraint is max(S) <= min(E).
        # We want min(E) to be LARGE.
        # So pair E with Largest A's.
        
        # Minimize min(Total(E))? No, we want Checks to PASS.
        # Pass Condition: max(S) <= min(E).
        # To make it pass, we want LEFT side LOW, RIGHT side HIGH.
        # Left Side (S): We gave them Smallest A. Correct.
        # Right Side (E): We give them Largest A. Correct.
        
        temp_E_totals = []
        if target_p in E:
            temp_E_totals.append(target_p['j_rank'] + a_test)
            
        E_others_sorted = sorted(E_others, key=lambda x: x['j_rank'], reverse=False) # Small J first
        # Why Small J first? 
        # We want to MAXIMIZE min(E).
        # Pair Small J with Large A? Or Small J with Small A?
        # J=[1, 10]. A=[10, 100].
        # 1+100=101, 10+10=20. min=20.
        # 1+10=11, 10+100=110. min=11.
        # To MAXIMIZE min(E), we should "balance" sums.
        # Pair Small J with Large A.
        # So E sorted Asc (Small J), pair with Desc A (Large A).
        
        num_e = len(E_others)
        # Take LARGEST available from remaining
        e_ranks = sorted(remaining_pool, reverse=True)[:num_e]
        
        for k, p in enumerate(E_others_sorted):
            temp_E_totals.append(p['j_rank'] + e_ranks[k])
            
        # Check Constraint
        max_s = max(temp_S_totals) if temp_S_totals else -float('inf')
        min_e = min(temp_E_totals) if temp_E_totals else float('inf')
        
        if max_s <= min_e:
            valid_a.append(a_test)
            
    if not valid_a:
        return 0, 0 # Should not happen unless inconsistent data
        
    return min(valid_a), max(valid_a)

def _process_season(
    season: int,
    df: pd.DataFrame,
    score_cols: list,
    is_ml: bool,
):
    results = []

    # Determine Rule Type
    rule_type = 'Percent'
    if season <= 2 or season >= 28:
        rule_type = 'Rank'

    prev_percent_map = {}

    # Get weeks
    weeks = set()
    for c in score_cols:
        try:
            week_num = int(c.split('_')[0].replace('week', ''))
            weeks.add(week_num)
        except Exception:
            pass
    sorted_weeks = sorted(list(weeks))
    last_week = sorted_weeks[-1] if sorted_weeks else None

    for week in sorted_weeks:
        participants = get_weekly_participants(df, season, week, score_cols, last_week)
        if not participants:
            continue

        n_p = len(participants)

        # Percent rule optimization (backprop) computed once per week
        week_aud_pcts = None
        week_aud_ranks = None
        week_loss = None
        prev_percent_map_for_week = prev_percent_map

        if rule_type == 'Percent' and is_ml:
            total_J = sum(p['total_score'] for p in participants)
            judge_percents = [
                (p['total_score'] / total_J) if total_J > 0 else 0.0
                for p in participants
            ]
            eliminated_mask = [p['is_eliminated_this_week'] for p in participants]
            safe_mask = [
                (not p['is_eliminated_this_week']) and p['status'] != 'Withdrew'
                for p in participants
            ]
            names = [p['name'] for p in participants]

            week_aud_pcts, week_aud_ranks, week_loss = optimize_audience_percent(
                judge_percents=judge_percents,
                eliminated_mask=eliminated_mask,
                safe_mask=safe_mask,
                prev_percent_map=prev_percent_map_for_week,
                participant_names=names,
                config=PERCENT_LOSS_CONFIG,
            )

            # Update previous week map for smoothness in next week
            prev_percent_map = {
                names[i]: float(week_aud_pcts[i]) for i in range(len(names))
            }

        for idx_p, p in enumerate(participants):
            val_range = ""
            j_norm = ""
            pred_aud_pct = None
            pred_aud_rank = None
            loss_total = None
            loss_constraint = None
            loss_smooth = None
            loss_corr = None
            loss_reg = None
            loss_diversity = None

            if rule_type == 'Rank':
                min_r, max_r = solve_rank_rule_inverse(p, participants, n_p)
                val_range = f"{min_r}-{max_r}"
                sorted_temp = sorted(participants, key=lambda x: x['total_score'], reverse=True)
                my_j = 0

                # Dense Ranking Logic
                current_rank = 0
                prev_score = None

                for i, x in enumerate(sorted_temp):
                    if x['total_score'] != prev_score:
                        current_rank += 1
                    x['drank'] = current_rank
                    prev_score = x['total_score']

                    if x['id'] == p['id']:
                        my_j = x['drank']
                j_norm = str(my_j)

            else:
                val_range = ""
                total_s = sum(x['total_score'] for x in participants)
                j_p = (p['total_score'] / total_s) * 100 if total_s > 0 else 0
                j_norm = f"{j_p:.1f}%"

                # Predicted audience percent/rank from backprop optimization
                if is_ml and week_aud_pcts is not None:
                    pred_aud_pct = float(week_aud_pcts[idx_p] * 100.0)
                    pred_aud_rank = int(round(week_aud_ranks[idx_p]))
                    if week_loss is not None:
                        loss_total = float(week_loss['total'])
                        loss_constraint = float(week_loss['constraint'])
                        loss_smooth = float(week_loss['smooth'])
                        loss_corr = float(week_loss['corr'])
                        loss_reg = float(week_loss['reg'])
                        loss_diversity = float(week_loss['diversity'])

            status_out = 'Safe'
            if p['status'] == 'Withdrew':
                status_out = 'Withdrew'
            elif p['is_eliminated_this_week']:
                if p.get('elim_rank') is not None:
                    status_out = f"Eliminated (Place {int(p['elim_rank'])})"
                else:
                    status_out = 'Eliminated'

            results.append({
                'CelebrityName': p['name'],
                'Season': season,
                'Week': week,
                'RuleType': rule_type,
                'JudgeScore': p['total_score'],
                'JudgeScore_Normalization': j_norm,
                'Possible_Audience_Vote_Range': val_range,
                'Predicted_Audience_Percent': pred_aud_pct,
                'Predicted_Audience_Rank': pred_aud_rank,
                'Loss_Total': loss_total,
                'Loss_Constraint': loss_constraint,
                'Loss_Smooth': loss_smooth,
                'Loss_Corr': loss_corr,
                'Loss_Reg': loss_reg,
                'Loss_Diversity': loss_diversity,
                'Status': status_out
            })

    return results


def main(is_ml=True, season_workers: int = 1):
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    score_cols = [c for c in df.columns if 'judge' in c and 'score' in c and 'week' in c]
    df = parse_result_status(df, score_cols)
    seasons = sorted(df['season'].unique())
    
    results = []
    global sorted_weeks # Hack for global access in get_participants
    
    total_weeks = 0
    for season in seasons:
        week_set = set()
        for c in score_cols:
            try:
                week_num = int(c.split('_')[0].replace('week', ''))
                week_set.add(week_num)
            except:
                pass
        total_weeks += len(week_set)

    week_progress = tqdm(total=total_weeks, desc="Total progress", leave=True) if tqdm is not None else None

    if season_workers is None or int(season_workers) < 1:
        season_workers = 1

    if season_workers == 1:
        for season in seasons:
            results.extend(_process_season(season, df, score_cols, is_ml))
            if week_progress is not None:
                # Approximate progress: mark all weeks for this season as done.
                season_weeks = set()
                for c in score_cols:
                    try:
                        week_num = int(c.split('_')[0].replace('week', ''))
                        season_weeks.add(week_num)
                    except Exception:
                        pass
                week_progress.update(len(season_weeks))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=int(season_workers)) as ex:
            futures = {
                ex.submit(_process_season, season, df, score_cols, is_ml): season
                for season in seasons
            }
            for fut in as_completed(futures):
                results.extend(fut.result())
                if week_progress is not None:
                    season_weeks = set()
                    for c in score_cols:
                        try:
                            week_num = int(c.split('_')[0].replace('week', ''))
                            season_weeks.add(week_num)
                        except Exception:
                            pass
                    week_progress.update(len(season_weeks))
    if week_progress is not None:
        week_progress.close()
                
    out_df = pd.DataFrame(results)
    if is_ml:
        config_payload = {
            'alpha_constraint': PERCENT_LOSS_CONFIG.alpha_constraint,
            'beta_smooth': PERCENT_LOSS_CONFIG.beta_smooth,
            'gamma_corr': PERCENT_LOSS_CONFIG.gamma_corr,
            'delta_reg': PERCENT_LOSS_CONFIG.delta_reg,
            'epsilon_diversity': PERCENT_LOSS_CONFIG.epsilon_diversity,
            'steps': PERCENT_LOSS_CONFIG.steps,
            'lr': PERCENT_LOSS_CONFIG.lr,
            'temperature': PERCENT_LOSS_CONFIG.temperature,
            'rank_tau': PERCENT_LOSS_CONFIG.rank_tau,
            'reg_type': PERCENT_LOSS_CONFIG.reg_type,
            'normal_sigma_factor': PERCENT_LOSS_CONFIG.normal_sigma_factor,
            'longtail_alpha': PERCENT_LOSS_CONFIG.longtail_alpha,
            'longtail_shift': PERCENT_LOSS_CONFIG.longtail_shift,
            'constraint_margin': PERCENT_LOSS_CONFIG.constraint_margin,
            'diversity_sigma': PERCENT_LOSS_CONFIG.diversity_sigma,
        }
        for key, value in config_payload.items():
            out_df[f'Config_{key}'] = value
        print("PercentLossConfig:", config_payload)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Calculate audience vote ranges.")
    parser.add_argument('--is-ml', dest='is_ml', action='store_true', help='Run ML optimization for percent rule.')
    parser.add_argument('--no-ml', dest='is_ml', action='store_false', help='Skip ML optimization for percent rule.')
    parser.add_argument(
        '--season-workers',
        dest='season_workers',
        type=int,
        default=1,
        help='Parallel workers for season-level processing (CPU-side).',
    )
    parser.set_defaults(is_ml=True)
    args = parser.parse_args()
    main(
        is_ml=bool(args.is_ml),
        season_workers=int(args.season_workers),
    )

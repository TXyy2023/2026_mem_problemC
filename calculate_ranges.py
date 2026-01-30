import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

INPUT_FILE = '/Users/a1234/Desktop/美赛/2026_MCM_Problem_C_Data.csv'
OUTPUT_FILE = '/Users/a1234/Desktop/美赛/MCM_Problem_C_Results.csv'

def calculate_best_rank_ranking_rule(my_judge_rank, other_judge_ranks, n_participants):
    """
    Calculate best rank (min rank) for a candidate under Ranking Rule.
    Strategy: Pair largest available Audience Rank with smallest available Other Judge Rank to neutralize.
    """
    my_score = my_judge_rank + 1
    others_j = sorted(other_judge_ranks) # Asc
    available_audience = sorted(list(range(2, n_participants + 1))) # Asc
    
    matches = 0
    j_ptr = len(others_j) - 1
    a_ptr = 0
    
    while j_ptr >= 0 and a_ptr < len(available_audience):
        # Can we save J[j_ptr] (Easiest to save) with A[a_ptr] (Strongest available)?
        if others_j[j_ptr] + available_audience[a_ptr] >= my_score:
             matches += 1
             j_ptr -= 1
             a_ptr += 1
        else:
             # A[a_ptr] is too weak for even the easiest J.
             # It acts as a "loss" (someone beats me).
             a_ptr += 1
             
    better_than_me = len(others_j) - matches
    return 1 + better_than_me

def calculate_worst_rank_ranking_rule(my_judge_rank, other_judge_ranks, n_participants):
    """
    Calculate worst rank (max rank) for a candidate under Ranking Rule.
    Strategy: Pair smallest available Other Judge Rank with largest usable Audience Rank < MyScore to Maximize Losses.
    """
    my_score = my_judge_rank + n_participants
    others_j = sorted(other_judge_ranks) # Asc
    available_audience = sorted(list(range(1, n_participants))) # 1 to N-1 (Asc)
    
    matches = 0
    # Process J from smallest (strongest candidates)
    # Try to pair with LARGEST feasible A
    
    j_ptr = 0
    a_ptr = len(available_audience) - 1
    
    while j_ptr < len(others_j) and a_ptr >= 0:
        if others_j[j_ptr] + available_audience[a_ptr] < my_score:
            matches += 1
            j_ptr += 1
            a_ptr -= 1
        else:
            # A[a_ptr] is too big. No J can handle it.
            a_ptr -= 1
            
    return 1 + matches

def main():
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Identify score columns
    score_cols = [c for c in df.columns if 'judge' in c and 'score' in c and 'week' in c]
    
    results = []

    # Get all unique seasons
    seasons = sorted(df['season'].unique())
    
    for season in seasons:
        season_df = df[df['season'] == season]
        
        # Identify active weeks for this season
        # We need to scan columns to see which weeks actually have data for this season
        # Or simpler: Iterating Weeks 1 to 11 (max week in dataset appears to be 11)
        # Actually max week depends on data.
        
        # Extract week numbers from columns
        weeks = set()
        for c in score_cols:
            # format: weekX_judgeY_score
            try:
                week_num = int(c.split('_')[0].replace('week', ''))
                weeks.add(week_num)
            except:
                pass
        
        sorted_weeks = sorted(list(weeks))
        
        for week in sorted_weeks:
            # Filter active participants for this week
            # Condition: At least one judge score is not NA and > 0?
            # Actually, scores are separate. We sum them up?
            # Rules say "Judge Ranking" based on "Total Judge Score".
            # So we sum the judges for that week.
            
            week_cols = [c for c in score_cols if c.startswith(f'week{week}_')]
            
            valid_participants = []
            
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
                
                # If no valid scores, skip (eliminated or didn't dance)
                if not scores:
                    continue
                    
                total_score = sum(scores)
                valid_participants.append({
                    'celebrity_name': row['celebrity_name'],
                    'total_score': total_score
                })
            
            if not valid_participants:
                continue
                
            n_participants = len(valid_participants)
            
            # Sort by total score descending (Highest is Best)
            valid_participants.sort(key=lambda x: x['total_score'], reverse=True)
            
            # Assign Judge Ranks
            # Process ties correctly for ranks (dense? standard?)
            # Usually strict rank: 1, 2, 2, 4.
            # Let's use standard ranking (scipy rank method 'min').
            # But let's write manual loop for clarity.
            current_rank = 1
            for i in range(n_participants):
                if i > 0 and valid_participants[i]['total_score'] == valid_participants[i-1]['total_score']:
                    # Same score, same rank
                    valid_participants[i]['judge_rank'] = valid_participants[i-1]['judge_rank']
                else:
                    valid_participants[i]['judge_rank'] = i + 1
            
            # Also calculate percentage (for Percent Rule)
            total_season_score = sum(p['total_score'] for p in valid_participants)
            for p in valid_participants:
                p['judge_percent'] = p['total_score'] / total_season_score if total_season_score > 0 else 0
            
            # Determine Rule Type
            # Rank Rule: 1-2, 28-34
            # Percent Rule: 3-27
            rule_type = 'Percent'
            if season <= 2 or season >= 28:
                rule_type = 'Rank'
            
            # Calculate Ranges
            scores_list = [p['total_score'] for p in valid_participants]
            judge_ranks_list = [p['judge_rank'] for p in valid_participants]
            
            for i, p in enumerate(valid_participants):
                my_rank_range = ""
                my_percent_range = ""
                
                if rule_type == 'Rank':
                    # Prepare input for greedy algo
                    my_j = p['judge_rank']
                    others_j = judge_ranks_list[:i] + judge_ranks_list[i+1:]
                    
                    min_r = calculate_best_rank_ranking_rule(my_j, others_j, n_participants)
                    max_r = calculate_worst_rank_ranking_rule(my_j, others_j, n_participants)
                    
                    possible_result = f"{min_r}-{max_r}"
                    judge_norm = int(p['judge_rank'])
                else:
                    # Percent Rule
                    budget = 1.0
                    my_jp = p['judge_percent']
                    
                    others_jp = [x['judge_percent'] for k, x in enumerate(valid_participants) if k != i]
                    
                    costs = []
                    for ojp in others_jp:
                         c = max(0.0, (my_jp - ojp) + 1e-9)
                         costs.append(c)
                    
                    costs.sort()
                    count_worse = 0 # Worse than me (Strictly Better Score)
                    
                    current_budget = 1.0
                    for c in costs:
                        if current_budget >= c:
                            current_budget -= c
                            count_worse += 1
                        else:
                            break
                    
                    # Percent Range output
                    # Just [J%, J%+100%]
                    min_p = p['judge_percent'] * 100
                    max_p = min_p + 100
                    possible_result = f"{min_p:.1f}%-{max_p:.1f}%"
                    judge_norm = f"{min_p:.1f}%"

                results.append({
                    'CelebrityName': p['celebrity_name'],
                    'Season': season,
                    'Week': week,
                    'RuleType': rule_type,
                    'JudgeScore': p['total_score'],
                    'JudgeScore_Normalization': judge_norm,
                    'Possible_Result': possible_result
                })

    # Save
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()

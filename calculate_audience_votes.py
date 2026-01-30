import pandas as pd
import numpy as np
from scipy.optimize import linprog
import warnings
import re

warnings.filterwarnings('ignore')

INPUT_FILE = '/Users/a1234/Desktop/美赛/2026_MCM_Problem_C_Data.csv'
OUTPUT_FILE = '/Users/a1234/Desktop/美赛/MCM_Problem_C_Results.csv'

def parse_result_status(df):
    """
    Parse 'results' column to determine Exit Week for each participant.
    """
    # Initialize ExitWeek with a large number (Survived)
    df['ExitWeek'] = 999 
    df['Status'] = 'Safe'
    
    for idx, row in df.iterrows():
        res = str(row['results']).lower()
        if 'eliminated week' in res:
            try:
                week = int(re.search(r'week (\d+)', res).group(1))
                df.at[idx, 'ExitWeek'] = week
                df.at[idx, 'Status'] = 'Eliminated'
            except:
                pass
        elif 'withdrew' in res:
            # Need to find when they withdrew.
            # Heuristic: Check last week with scores.
            # We will handle this by checking scores later.
            df.at[idx, 'Status'] = 'Withdrew' 
            # We'll set ExitWeek dynamically based on missing scores.
        elif 'place' in res:
            # Finalist
            pass
            
    return df

def get_weekly_participants(df, season, week, score_cols):
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
                'actual_result': row['results']
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
            if 'place' in str(p_data['actual_result']).lower() and week == sorted_weeks[-1]:
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

def solve_percent_rule_inverse(target_p, participants):
    """
    Linear Programming for Percent Rule.
    Vars: x_1 ... x_n (Audience Percentages).
    Target: Min/Max x_target.
    Constraints:
    1. Sum(x) = 1.
    2. x >= 0.
    3. For all s in S, e in E: Total(s) >= Total(e).
       (J_s + x_s) >= (J_e + x_e)
       x_s - x_e >= J_e - J_s
    """
    S = [p for p in participants if not p['is_eliminated_this_week'] and p['status'] != 'Withdrew']
    E = [p for p in participants if p['is_eliminated_this_week']]
    
    if not E:
        return 0.0, 100.0
        
    n = len(participants)
    target_idx = participants.index(target_p)
    
    # Vars: x_0 ... x_{n-1}
    # Bounds: (0, 1)
    bounds = [(0, 1) for _ in range(n)]
    
    # Equality: Sum(x) = 1
    # A_eq * x = b_eq
    A_eq = [[1.0] * n]
    b_eq = [1.0]
    
    # Inequality: A_ub * x <= b_ub
    # SciPy uses <=.
    # Constraint: x_s - x_e >= val  =>  x_e - x_s <= -val
    
    A_ub = []
    b_ub = []
    
    # Calculate Judge Percentages
    total_J = sum(p['total_score'] for p in participants)
    J_percents = [p['total_score']/total_J for p in participants]
    
    s_indices = [participants.index(p) for p in S]
    e_indices = [participants.index(p) for p in E]
    
    for s_i in s_indices:
        for e_i in e_indices:
            # x_e - x_s <= J_s - J_e
            # Row vector
            row = [0] * n
            row[e_i] = 1
            row[s_i] = -1
            A_ub.append(row)
            b_ub.append(J_percents[s_i] - J_percents[e_i])
            
    # Create valid LP?
    # If S and E empty? Handled.
    
    # Solve Min
    c_min = [0] * n
    c_min[target_idx] = 1 # Minimize target
    
    res_min = linprog(c_min, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    # Solve Max
    c_max = [0] * n
    c_max[target_idx] = -1 # Maximize target
    
    res_max = linprog(c_max, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    min_val = 0.0
    max_val = 100.0
    
    if res_min.success:
        min_val = res_min.fun * 100
    if res_max.success:
        max_val = -res_max.fun * 100
        
    return max(0.0, min_val), min(100.0, max_val)

def main():
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df = parse_result_status(df)
    
    score_cols = [c for c in df.columns if 'judge' in c and 'score' in c and 'week' in c]
    seasons = sorted(df['season'].unique())
    
    results = []
    global sorted_weeks # Hack for global access in get_participants
    
    for season in seasons:
        # Determine Rule Type
        rule_type = 'Percent'
        if season <= 2 or season >= 28:
            rule_type = 'Rank'
            
        # Get weeks
        weeks = set()
        for c in score_cols:
            try:
                week_num = int(c.split('_')[0].replace('week', ''))
                weeks.add(week_num)
            except:
                pass
        sorted_weeks = sorted(list(weeks))
        
        for week in sorted_weeks:
            participants = get_weekly_participants(df, season, week, score_cols)
            if not participants:
                continue
                
            n_p = len(participants)
            
            for p in participants:
                val_range = ""
                j_norm = ""
                
                if rule_type == 'Rank':
                    min_r, max_r = solve_rank_rule_inverse(p, participants, n_p)
                    val_range = f"{min_r}-{max_r}"
                    # Judge Norm: Judge Rank
                    # Need to retrieve J Rank calculated inside logic or redo
                    # Just calculate simply here
                    matches = [x for x in participants if x['id'] == p['id']]
                    # It's messy to retrieve, recalculate J Rank
                    # Or reuse logic
                    # Let's clean up: Calculate J_ranks centrally
                    # Just do quick rank for display
                    # ... (skip for brevity, logic exists in solve)
                    # We will output Placeholders or move logic out.
                    # Redo J Rank for display:
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
                    min_p, max_p = solve_percent_rule_inverse(p, participants)
                    val_range = f"{min_p:.1f}%-{max_p:.1f}%"
                    total_s = sum(x['total_score'] for x in participants)
                    j_p = (p['total_score'] / total_s) * 100 if total_s > 0 else 0
                    j_norm = f"{j_p:.1f}%"
                
                results.append({
                    'CelebrityName': p['name'],
                    'Season': season,
                    'Week': week,
                    'RuleType': rule_type,
                    'JudgeScore': p['total_score'],
                    'JudgeScore_Normalization': j_norm,
                    'Possible_Audience_Vote_Range': val_range,
                    'Status': p['status'] if p['is_eliminated_this_week'] else 'Safe'
                })
                
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()

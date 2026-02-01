
import pandas as pd
import numpy as np
import os
import random

# Configuration
INPUT_FILE = '2026_MCM_Problem_C_Data.csv'
OUTPUT_FILE = 'MCM_Problem_C_MonteCarlo_Results.csv'
NUM_PARTICLES = 1000   # Number of particles (trajectories) per season
SEASONS_RANK = [1, 2] + list(range(28, 35)) # Assuming 34 is current max? Just simplified logic later.

def load_data(filepath):
    """Load and preprocess data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    return df

def get_season_data(df, season):
    """Filter data for a specific season."""
    return df[df['season'] == season].copy()

def parse_status(row, week):
    """Determine status for a specific week."""
    # Simplified logic: consistent with calculate_audience_votes.py
    res = str(row['results']).lower()
    exit_week = 999
    if 'eliminated week' in res:
        try:
            import re
            exit_week = int(re.search(r'week (\d+)', res).group(1))
        except:
            pass
    elif 'third' in res or '3rd' in res:
         # Finalist logic could be complex, assuming they stay till end
         pass
    
    # Determine if eliminated THIS week
    is_eliminated = (exit_week == week)
    is_safe = (exit_week > week)
    # What if they withdrew?
    status = 'Safe'
    if is_eliminated:
        status = 'Eliminated'
    elif 'withdrew' in res:
        # Check if they have scores?
        # If no score, they are not in participants list usually.
        pass
        
    return status, is_eliminated

def check_structure_validity(ranks_dict, participants, week_data):
    """
    Check if the assigned ranks are consistent with the Elimination constraints.
    Constraint: Sum(JudgeRank + AudRank) for Eliminated >= Sum(JudgeRank + AudRank) for Safe.
    Essentially, those eliminated should be at the bottom (highest Rank Sum).
    
    participants: list of player IDs (or names)
    ranks_dict: {player_id: audience_rank}
    week_data: dict {player_id: {judge_score: ..., status: ...}}
    """
    
    # diverse rank logic or judge score?
    # Rank Rule: Total Rank = Judge Rank + Audience Rank.
    # We need Judge Rank.
    # Scores are given. We calculate Judge Rank from Scores.
    
    # 1. Calculate Judge Ranks
    # Sort by Score Descending
    p_list = [p for p in participants if p in week_data]
    if not p_list: return False
    
    # Sort by score desc
    sorted_by_score = sorted(p_list, key=lambda p: week_data[p]['judge_score'], reverse=True)
    
    judge_ranks = {}
    current_rank = 1
    for i, p in enumerate(sorted_by_score):
        if i > 0 and week_data[p]['judge_score'] < week_data[sorted_by_score[i-1]]['judge_score']:
            current_rank = i + 1
        judge_ranks[p] = current_rank
        
    # 2. Calculate Total Ranks
    total_ranks = [] # (player, total_rank, status)
    
    for p in p_list:
        if p not in ranks_dict:
             return False # Should not happen
        t_rank = judge_ranks[p] + ranks_dict[p]
        status = week_data[p]['status']
        total_ranks.append({'p': p, 'total': t_rank, 'status': status})
        
    # 3. Validation
    # Eliminated players must be "worse" than Safe players.
    # Worse = Higher Total Rank.
    # Strict inequality? Usually E >= S is minimal req. 
    # Real elim rule: The maximum total rank is eliminated.
    # So max(Total(E)) >= max(Total(S)) ? No.
    # All E should be >= All S? No, usually just the bottom 1 or 2 are eliminated.
    # So the set of Eliminated players must have the Highest Total Ranks.
    # i.e. Minimum(Total(E)) >= Maximum(Total(S)) ?
    # Let's say: 
    # Sort all by Total Rank Descending.
    # The top K (where K is num eliminated) should be the Eliminated ones.
    # If there is a tie at the boundary, tie-breaker applies.
    # Tie-breaker: Lower Judge Score is saved? Or Eliminated?
    # Usually: Judge Score breaks tie. Lower Judge Score -> Eliminated?
    # "The couple with the lowest combined total of judges' points and audience votes is eliminated."
    # "If there is a tie... the couple with the lowest judges' score is eliminated."
    
    # So: Sort by Total (Desc), then JudgeScore (Asc - because low score is bad).
    # Wait, Low Judge Score is Bad. So if Totals are equal, the one with Lower Score is WORSE (higher effective rank).
    # So Sort Key: (Total Rank Desc, JudgeScore Asc (Low is bad -> Top of sorted list)).
    
    sorted_final = sorted(total_ranks, key=lambda x: (x['total'], -week_data[x['p']]['judge_score']), reverse=True)
    # Why negative judge score?
    # We want "Worse" players at index 0.
    # Worse = High Total.
    # Tie = Low Judge Score.
    # So (Total DESC, Judge Score ASC).
    # Python sorts tuples elementwise.
    # (10, 20) vs (10, 15).
    # We want (10, 15) to come BEFORE (10, 20) because 15 < 20 (Worse).
    # So Sort DESCENDING? 
    # (10, 20) > (10, 15). Use total DESC.
    # If we want (10, 15) first:
    # Key = (Total, -JudgeScore).
    # (10, -15) vs (10, -20). 
    # -15 > -20. So (10, 15) comes FIRST in descending sort? 
    # Wait: -15 > -20.
    # So (10, -15) is "Greater". 
    # Descending sort puts Greater first.
    # So (10, 15) comes first.
    # Is (10, 15) WORSE or BETTER than (10, 20)?
    # Low score (15) is Worse.
    # So we want Worse to be at the top (Eliminated).
    # So yes, (10, 15) should be considered "Higher/Worse" than (10, 20).
    # So Correct.
    
    # Identify expected status based on rank
    num_eliminated = sum(1 for x in total_ranks if x['status'] == 'Eliminated')
    if num_eliminated == 0:
        return True # No constraints if no one goes home
        
    # The top num_eliminated in the sorted list MUST be the actual eliminated ones.
    # Or rather, the actual eliminated ones must allow for this.
    # Wait, if there are ties at the boundary, multiple people COULD be eliminated.
    # But we know who WAS eliminated.
    # So, check if the actual eliminated people are indeed the worst `num_eliminated` candidates?
    # Actually, simply:
    # For every E in Eliminated, and S in Safe:
    # E is WorseThanOrEqual S.
    # IsRankWorse(E, S) should be True.
    
    # Let's use the Pairwise check as it's more robust to specific "Bottom 2" vs "Bottom 1" ambiguity.
    # Constraint: Generally, an Eliminated person should not have a Better rank than a Safe person.
    # i.e. For all e in E, s in S:
    #   Total(e) >= Total(s)
    #   (And if Total(e) == Total(s), then Judge(e) <= Judge(s)).
    
    elim_list = [x for x in total_ranks if x['status'] == 'Eliminated']
    safe_list = [x for x in total_ranks if x['status'] == 'Safe']
    
    for e in elim_list:
        e_judge = week_data[e['p']]['judge_score']
        for s in safe_list:
            s_judge = week_data[s['p']]['judge_score']
            
            # Check if S is strictly worse than E (Impossible)
            # S Worse E means: S_Total > E_Total OR (S_Total == E_Total and S_Judge < E_Judge)
            # If S is Worse than E, then S should have been eliminated instead of E (or with E).
            # This is invalid.
            
            s_is_worse = False
            if s['total'] > e['total']:
                s_is_worse = True
            elif s['total'] == e['total']:
                # Tie breaker: Lower judge score is worse.
                if s_judge < e_judge:
                    s_is_worse = True
            
            if s_is_worse:
                return False
                
    return True

def monte_carlo_season(df, season):
    """Run MC for a single season."""
    print(f"Processing Season {season}...")
    
    # Get all potential participants for the season
    season_df = get_season_data(df, season)
    score_cols = [c for c in df.columns if 'judge' in c and 'score' in c and 'week' in c]
    
    # Identify weeks
    weeks = set()
    for c in score_cols:
        try:
             week_num = int(c.split('_')[0].replace('week', ''))
             col_name = f"week{week_num}_judge1_score"
             # Check if season has data for this week
             if not season_df[col_name].dropna().empty:
                 weeks.add(week_num)
        except: pass
    sorted_weeks = sorted(list(weeks))
    
    # Particles: List of dicts.
    # Each dict: {player_name: rank_history_list}
    # Actually, we proceed week by week.
    # Particle_t = {player_name: current_rank}
    # We maintain a list of Particles (List of Dicts).
    # And we need to store History to check the "Filter".
    # So Particle = { 'history': {player: [r1, r2...]}, 'current_ranks': {player: r_t} }
    
    particles = [] 
    # Initialization
    # We execute Week 1.
    
    results_collector = []
    
    for week_idx, week in enumerate(sorted_weeks):
        # 1. Identify Participants for this week
        # Must have a score > 0
        week_score_cols = [c for c in score_cols if c.startswith(f'week{week}_')]
        
        week_data = {} # name -> {score, status}
        p_names = []
        
        for idx, row in season_df.iterrows():
            # Calculate total score
            scores = []
            for c in week_score_cols:
                try:
                    v = float(row[c])
                    if v > 0: scores.append(v)
                except: pass
            
            if scores:
                total_score = sum(scores)
                status, is_elim = parse_status(row, week)
                name = row['celebrity_name']
                week_data[name] = {'judge_score': total_score, 'status': status, 'is_elim': is_elim}
                p_names.append(name)
        
        if not p_names: continue
        
        num_p = len(p_names)
        print(f"  Week {week}: {num_p} participants.")
        
        # 2. Generate Particles
        new_particles = []
        
        # Determine Limit for Filter
        # Rank_t <= Rank_{t-1} + floor(N/2)
        filter_limit_add = num_p // 2
        
        # If Week 1, Generate Random N! (sampled)
        if week_idx == 0:
            attempts = 0
            while len(new_particles) < NUM_PARTICLES and attempts < NUM_PARTICLES * 100:
                attempts += 1
                # Random permutation
                ranks = list(range(1, num_p + 1))
                random.shuffle(ranks)
                p_ranks = {name: r for name, r in zip(p_names, ranks)}
                
                if check_structure_validity(p_ranks, p_names, week_data):
                    # Valid
                    new_particles.append(p_ranks)
        else:
            # Propagate from previous particles
            # Previous particles contain ranks for Previous Candidates.
            # Current candidates might be fewer (eliminations).
            # We filter previous particles?
            # Or we just take previous particles and extend them?
            
            # Note: We need M new particles.
            # We can sample from old particles, then generate new step.
            
            if not particles:
                print("  No surviving particles from previous week. Stopping.")
                break
                
            attempts = 0
            while len(new_particles) < NUM_PARTICLES and attempts < NUM_PARTICLES * 50:
                attempts += 1
                # Sample a parent
                parent = random.choice(particles) # {name: rank}
                
                # Check constraints for each player to define domain
                # Domain = [1, N] INTERSECT [1, prev_rank + limit]
                # Note: prev_rank refers to rank in Previous Week.
                # Does player exist in parent?
                
                # Build Domain
                domains = {} # index in p_names -> list of valid ranks
                valid_assignment_possible = True
                
                for i, name in enumerate(p_names):
                    max_r = num_p
                    if name in parent:
                        prev_r = parent[name] # Note: Previous rank was out of N_prev
                        # The constraint uses the rank NUMBER.
                        constraint = prev_r + filter_limit_add
                        max_r = min(num_p, constraint)
                    
                    possible = list(range(1, max_r + 1))
                    if not possible:
                        valid_assignment_possible = False
                        break
                    domains[name] = possible
                    
                if not valid_assignment_possible:
                    continue
                    
                # Generate a permutation consistent with domains
                # Generating a valid permutation with specific constraints is hard (Permanent).
                # Simple Rejection Sampling?
                # Or greedy with backtracking?
                # Or just Shuffle [1..N] and check?
                # Given N ~ 10-13, Shuffle is 10^9... 
                # Rejection sampling might be slow if constraints are tight.
                
                # Optimized Generator:
                # Fill positions [1..N] one by one?
                # Or assign Ranks to Persons?
                # Persons have domains.
                # Sort persons by domain size (MRV heuristic).
                
                # Try simplistic rejection for now (speed check?)
                # If N=10, 50% constraint -> Domains are large.
                # Shuffle 1..N. Check if valid.
                
                # Heuristic: 
                # Create a random permutation.
                # Check if each person's assigned rank is in their domain.
                
                current_ranks_list = list(range(1, num_p + 1))
                random.shuffle(current_ranks_list)
                
                # Check domains
                is_valid_perm = True
                rank_map = {}
                for i, name in enumerate(p_names):
                    r = current_ranks_list[i]
                    if r > (parent.get(name, 999) + filter_limit_add):
                        is_valid_perm = False
                        break
                    rank_map[name] = r
                
                if is_valid_perm:
                    # Check Elimination Constraint
                    if check_structure_validity(rank_map, p_names, week_data):
                        new_particles.append(rank_map)

        particles = new_particles
        print(f"  Generated {len(particles)} valid particles.")
        
        # Record Stats
        if particles:
            for name in p_names:
                all_r = [p[name] for p in particles]
                mean_r = np.mean(all_r)
                min_r = np.min(all_r)
                max_r = np.max(all_r)
                # 95% CI
                low_95 = np.percentile(all_r, 25)
                high_95 = np.percentile(all_r, 75)
                
                results_collector.append({
                    'Season': season,
                    'Week': week,
                    'CelebrityName': name,
                    'Mean_Rank': mean_r,
                    'Min_Rank': min_r,
                    'Max_Rank': max_r,
                    'CI_Lower': low_95,
                    'CI_Upper': high_95, 
                    'Status': week_data[name]['status'],
                    'RuleType': 'Rank'
                })
        else:
            print("  Simulation died.")
            
    return results_collector

def main():
    df = load_data(INPUT_FILE)
    all_results = []
    
    for season in SEASONS_RANK:
        try:
            res = monte_carlo_season(df, season)
            all_results.extend(res)
        except Exception as e:
            print(f"Error in Season {season}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save
    out = pd.DataFrame(all_results)
    out.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()

import unittest

def calculate_best_rank_ranking_rule(my_judge_rank, other_judge_ranks, n_participants):
    """
    Calculate best rank (min rank) for a candidate under Ranking Rule.
    My Score = my_judge + 1 (best aud).
    We want to MINIMIZE count of others with Score_other < My Score.
    This means MAXIMIZE count of others with Score_other >= My Score.
    Strategy: Satisfy "Score_other >= My Score" for EASIEST cases first.
    High Judge Rank -> Easiest to make Score High.
    Large Audience Rank -> Strongest tool to make Score High.
    """
    my_score = my_judge_rank + 1
    
    # We want to maximize pairs (j, a) such that j + a >= my_score
    # But wait, tie breaking? 
    # If S_other < S_my: They beat me.
    # If S_other == S_my: Tie. Usually doesn't push me down in dense rank.
    # So I only care about S_other < S_my. 
    # So I want to AVOID j + a < my_score.
    
    # Others judge ranks
    others_j = sorted(other_judge_ranks) # Smallest (strongest) to Largest (weakest)
    
    # Available audience ranks for others: 2, 3, ..., N
    available_audience = list(range(2, n_participants + 1))
    available_audience.sort(reverse=True) # Largest (weakest) to Smallest (strongest)
    
    # Greedy Strategy:
    # Try to "save" the strong opponents (Small J) from beating me.
    # To stop J from beating me, I need J + A >= my_score.
    # If I use a very large A, I make J+A very large.
    # I should use the SMALLEST A that is sufficient to make J+A >= my_score?
    # No, let's reverse:
    # I have strong opponents (Small J). They are THREATS.
    # I need to neutralize Threats.
    # To neutralize a Small J, I need a Large A.
    # So: Take smallest J (Biggest Threat). Pair with Largest A.
    # If J_min + A_max < my_score: This threat CANNOT be neutralized. He beats me.
    # If J_min + A_max >= my_score: Threat neutralized.
    # But should I use A_max? 
    # Suppose J_min = 1, MyScore = 4. I check A=3. 1+3=4 >= 4. Neutralized.
    # Suppose J_next = 1. A_next = 2. 1+2=3 < 4. Beaten.
    # What if I had A= [3, 2].
    # If I used A=2 for J_min? 1+2 = 3 < 4. Fail.
    # So I MUST use A >= 3.
    # So, yes, use the largest available A's to neutralize the smallest J's.
    
    beats_me_count = 0
    
    # Pointers
    # We want to pair Smallest J with Largest A to check if we CAN neutralize.
    # If we can, we assume we use "appropriate" A. 
    # Actually, simpler: Measure how many CANNOT be neutralized.
    # But A's are consumed.
    
    # Let's try: Match Smallest J with Largest A.
    # If sum < my_score: He beats me. No A can save him. Discard this J (he wins) and discard this A?
    # No, if he beats me anyway, I should give him the SMALLEST A (best rank) to save the Large A (worst rank/high val) for someone else?
    # YES. If J_i is unmatched, give him "useless" small A.
    
    # Refined Greedy:
    # J sorted asc (1, 2, ...). A sorted asc (2, 3, ...).
    # Iterate J from Smallest (Hardest to neutralize).
    # Iterate A from Largest (Best neutralizer).
    # If J[small] + A[large] >= my_score:
    #    We CAN neutralize J[small]. We use A[large].
    #    Wait, if A[large] is overkill?
    #    Ex: MyScore=10. J=5. A_large=100. A_mid=5.
    #    Both neutralize. Using A=100 wastes it.
    #    So we should find Simple Matching Max Flow type logic.
    #    We want to MAXIMIZE pairs (j, a) s.t. j+a >= my_score.
    #    This is "Maximize number of pairs >= K".
    #    Standard Greedy for this:
    #    Sort J (asc), Sort A (asc).
    #    Iterate J from Largest (Easiest to satisfy).
    #    Pair with Smallest valid A.
    #    If J_large + A_small >= my_score: Match! (We used cheapest A for easiest J).
    #    
    #    Let's re-verify logic:
    #    We want to MINIMIZE "Better than me" => MAXIMIZE "Worse or Equal".
    #    So yes, maximize count(J+A >= my_score).
    
    others_j = sorted(other_judge_ranks) # Asc
    available_audience = sorted(list(range(2, n_participants + 1))) # Asc
    
    count_neutralized = 0
    
    # Check from largest J (easy)
    # Use matching with available_audience
    
    a_idx = 0 # Smallest A
    # Iterate J from largest to smallest?
    # Or J from smallest?
    # If we process Largest J first. We can satisfy it with small A.
    # This leaves Large A for Small J. This is OPTIMAL.
    # Because Small J NEEDS Large A.
    
    # So: Iterate J from Largest down to Smallest?
    # Let's trace: My=6. J=[1, 5]. A=[2, 8].
    # J_large = 5. Needs >= 1. A_small=2 works. (5+2=7>=6). Used A=2.
    # J_small = 1. Needs >= 5. Remaining A=[8]. (1+8=9>=6). Used A=8.
    # Total neutralized: 2. Rank 1.
    
    # Trace 2: My=6. J=[1, 5, 9]. A=[2, 3, 4].
    # J=9. Need >=-3. A=2 works. Match.
    # J=5. Need >=1. A=3 works. Match.
    # J=1. Need >=5. A=4 (1+4=5? No 1+4=5. >=? Strict? tie is allowed).
    # If 5 < 6. 5 is better. 
    # MyScore=6. J+A=5. He beats me.
    # So we need J+A >= 6 (assuming tie is okay) OR J+A > 6?
    # If Tie (Score 6 vs 6): Rank shared. I am still Rank 1 (Top).
    # So >= 6 is success.
    
    my_score = my_judge_rank + 1
    others_j = sorted(other_judge_ranks) # Asc
    available_audience = sorted(list(range(2, n_participants + 1))) # Asc
    
    # Iterate J from Largest (easiest)
    matches = 0
    # Available A set?
    # Simple algorithm:
    # j_ptr = len(others) - 1 (Largest J)
    # a_ptr = 0 (Smallest A)
    # while j_ptr >= 0 and a_ptr < len(A):
    #    if J[j_ptr] + A[a_ptr] >= my_score:
    #        matches += 1 (Saved!)
    #        j_ptr -= 1
    #        a_ptr += 1 (Used smallest valid A)
    #    else:
    #        # A[a_ptr] is too small even for J[j_ptr].
    #        # But J[j_ptr] is the EASIEST remaining J.
    #        # So A[a_ptr] is useless for ANY remaining J.
    #        # Discard A.
    #        a_ptr += 1
    
    j_ptr = len(others_j) - 1
    a_ptr = 0
    while j_ptr >= 0 and a_ptr < len(available_audience):
        # We try to satisfy J[j_ptr] with smallest possible A
        if others_j[j_ptr] + available_audience[a_ptr] >= my_score:
             matches += 1
             j_ptr -= 1
             a_ptr += 1
        else:
             # This A is too weak for even the easiest J.
             # It will definitely fail to save anyone (since other Js are harder).
             # So this A corresponds to a "Loss" (someone beating me).
             # But wait, does it matter WHO beats me? No.
             # Just that we can't form a >= pair with this A.
             a_ptr += 1
             
    better_than_me = len(others_j) - matches
    return 1 + better_than_me

def calculate_worst_rank_ranking_rule(my_judge_rank, other_judge_ranks, n_participants):
    """
    Worst Rank:
    My Score = my_judge + N.
    Maximize count of J + A < My Score (Strictly Better?).
    If J+A == MyScore, we tie. 
    If there are K people strictly better, and L people tied.
    Dense rank: I am Rank 1+K.
    So we want to MAXIMIZE Strictly Better (Score_other < My_Score).
    Condition: J + A < My_Score.
    
    Maximize pairs < K.
    Greedy:
    Sort J (asc - strongest). Sort A (asc - strongest).
    We want to form Strong pairs.
    Smallest J (strongest) + Smallest A (strongest) -> Most likely to be < K.
    But is it optimal?
    J=[1, 10]. A=[1, 10]. Limit 15.
    1+1=2 (ok). 10+10=20 (fail). Count 1.
    1+10=11 (ok). 10+1=11 (ok). Count 2.
    Mixed is better.
    
    Greedy Strategy for Max Counts < K:
    Smallest J (Strongest). Need Largest possible A that keeps it < K.
    Why Largest? To save Small A for Weaker J (Large J).
    TRACE: J=[1, 10]. A=[1, 10]. Limit 15.
    J=1. Max valid A? 10. 1+10=11 < 15. OK. Match. Used J=1, A=10.
    J=10. Left A=1. 10+1=11 < 15. Match. Used J=10, A=1.
    Total 2. Matches intuition.
    
    Algorithm:
    Sort J (Asc). Sort A (Asc).
    Iterate J from Smallest.
    Find Largest A such that J + A < K.
    """
    my_score = my_judge_rank + n_participants
    others_j = sorted(other_judge_ranks) # Asc
    available_audience = sorted(list(range(1, n_participants))) # 1 to N-1
    
    matches = 0
    # Process J from smallest (strongest candidates)
    # Try to pair with LARGEST feasible A
    
    # Pointers
    # Just iterate J. Use binary search or two pointers for A?
    # A is sorted asc.
    # For J[i], we want largest A[k] < Limit - J[i].
    # If we find it, match and remove A[k].
    
    # Efficient:
    # J from Smallest.
    # A from Largest.
    # if J[small] + A[large] < K:
    #    Match! (This A is strong enough (numerically small enough? No A is rank).
    #    Wait. A_large is WEAK rank (big number).
    #    If even with Weak Rank, J is strong enough to beat me...
    #    Then SURELY this J can beat me with Strong Rank.
    #    But we want to save Strong Rank (Small A) for Weak J.
    #    So yes, use Weak Rank (Large A) if possible.
    #    So: if J[small] + A[large] < K -> Match.
    # else:
    #    J[small] + A[large] >= K.
    #    This A is too big (too weak).
    #    So J[small] cannot carry this Weak A.
    #    But maybe J[small] can carry a Stronger A (Small A).
    #    Actually, since we process J from Smallest (Strongest),
    #    and A from Largest (Weakest).
    #    If J[small] cannot carry A[large], then NOBODY can carry A[large] (since J are getting weaker/larger).
    #    So A[large] is useless. Discard it.
    
    j_ptr = 0
    a_ptr = len(available_audience) - 1
    
    while j_ptr < len(others_j) and a_ptr >= 0:
        if others_j[j_ptr] + available_audience[a_ptr] < my_score:
            matches += 1
            j_ptr += 1
            a_ptr -= 1
        else:
            # A[a_ptr] is too big. No J can handle it (since J are increasing).
            a_ptr -= 1
            
    return 1 + matches

class TestRankCalculations(unittest.TestCase):
    def test_best_rank_ranking_rule(self):
        # Case 1: Easy win
        # My J=1. Others=[2]. N=2.
        # MyScore = 2.
        # A_avail = [2].
        # J=2 + A=2 = 4 >= 2. Match. Better=0. Rank 1.
        self.assertEqual(calculate_best_rank_ranking_rule(1, [2], 2), 1)
        
        # Case 2: Impossible to win
        # My J=10. Others=[1]. N=2.
        # MyScore = 11.
        # A_avail = [2].
        # J=1 + A=2 = 3. 3 < 11. No match. Better=1. Rank 2.
        self.assertEqual(calculate_best_rank_ranking_rule(10, [1], 2), 2)
        
        # Case 3: Mixed
        # My J=3. Others=[1, 2, 5]. N=4.
        # MyScore = 4.
        # A_avail = [2, 3, 4].
        # Target: J + A >= 4.
        # J_large=5. A_small=2. 5+2=7>=4. Match. (J=5 settled). Used A=2.
        # J_mid=2. Need A>=2. Avail=[3,4]. A_small=3. 2+3=5>=4. Match. Used A=3.
        # J_small=1. Need A>=3. Avail=[4]. 1+4=5>=4. Match. Used A=4.
        # All matched. Rank 1.
        self.assertEqual(calculate_best_rank_ranking_rule(3, [1, 2, 5], 4), 1)
        
        # Case 4: Semi-Fail
        # My J=2. Others=[1, 1]. N=3.
        # MyScore = 3.
        # A_avail = [2, 3].
        # J_large=1. Need A>=2. A=2 works. 1+2=3>=3. Match. Used A=2.
        # J_small=1. Need A>=2. Avail=[3]. A=3 works. 1+3=4>=3. Match.
        # Rank 1. (Tie is allowed).
        self.assertEqual(calculate_best_rank_ranking_rule(2, [1, 1], 3), 1)

    def test_worst_rank_ranking_rule(self):
        # Case 1: Easy loss
        # My J=10. Others=[1]. N=2.
        # MyScore (Worst) = 10 + 2 = 12.
        # A_avail = [1].
        # J=1 + A=1 = 2 < 12. Match. Rank 2.
        self.assertEqual(calculate_worst_rank_ranking_rule(10, [1], 2), 2)
        
        # Case 2: Saving grace
        # My J=1. Others=[2]. N=2.
        # MyScore = 1 + 2 = 3.
        # A_avail = [1].
        # J=2 + A=1 = 3. Not < 3. 
        # Rank 1.
        self.assertEqual(calculate_worst_rank_ranking_rule(1, [2], 2), 1)
        
        # Case 3: Mixed
        # My J=4. Others=[1, 2, 5]. N=4.
        # MyScore = 4 + 4 = 8.
        # A_avail = [1, 2, 3].
        # J=1. Max A such that 1+A < 8? 
        # A=3 -> 4 < 8. OK. Match. Used A=3.
        # J=2. Left [1, 2]. Max A s.t. 2+A < 8? A=2 -> 4 < 8. Match. Used A=2.
        # J=5. Left [1]. 5+1=6 < 8. Match. Used A=1.
        # Matches=3. Rank 4.
        self.assertEqual(calculate_worst_rank_ranking_rule(4, [1, 2, 5], 4), 4)

if __name__ == '__main__':
    unittest.main()

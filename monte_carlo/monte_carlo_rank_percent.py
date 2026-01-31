import argparse
import os

import pandas as pd

DEFAULT_MC_INPUT = 'MCM_Problem_C_MonteCarlo_Results.csv'
DEFAULT_RESULT_DATA = 'MCM_Problem_C_Results_20260131_1700.csv'


def rank_to_percent(rank, n):
    """
    Convert rank (1 is best) to percent in [0, 100].
    Example: rank=1 => 100%, rank=n => 100/n.
    """
    rank_s = pd.to_numeric(rank, errors='coerce')
    n_s = pd.to_numeric(n, errors='coerce')
    percent = (n_s - rank_s + 1) / n_s * 100.0
    return percent.where(n_s > 0)


def build_percent_columns(df):
    required = {'Season', 'Week', 'CelebrityName', 'Mean_Rank'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Participants per season/week
    counts = (
        df.groupby(['Season', 'Week'])['CelebrityName']
        .nunique()
        .rename('Participants')
        .reset_index()
    )

    out = df.merge(counts, on=['Season', 'Week'], how='left')

    # Compute percent columns based on rank statistics (vectorized)
    out['Percent_Mean'] = rank_to_percent(out['Mean_Rank'], out['Participants'])

    if 'Min_Rank' in out.columns:
        out['Percent_Max'] = rank_to_percent(out['Min_Rank'], out['Participants'])
    if 'Max_Rank' in out.columns:
        out['Percent_Min'] = rank_to_percent(out['Max_Rank'], out['Participants'])

    if 'CI_Lower' in out.columns:
        # Lower rank => higher percent; CI_Lower is better rank bound
        out['Percent_Upper'] = rank_to_percent(out['CI_Lower'], out['Participants'])
    if 'CI_Upper' in out.columns:
        out['Percent_Lower'] = rank_to_percent(out['CI_Upper'], out['Participants'])

    return out


def build_normalized_percent(df):
    """
    Build a normalized percent per Season+Week such that sum(percent)=100.
    Method 3: use CI_Lower/CI_Upper to form a central rank and invert it as weight.
    """
    required = {'Season', 'Week', 'CelebrityName'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    if 'CI_Lower' in out.columns and 'CI_Upper' in out.columns:
        out['Rank_predict'] = (out['CI_Lower'] + out['CI_Upper']) / 2.0
    else:
        # Fallback to Mean_Rank if CI columns missing
        if 'Mean_Rank' not in out.columns:
            raise ValueError("Missing CI_Lower/CI_Upper and Mean_Rank for fallback.")
        out['Rank_predict'] = out['Mean_Rank']

    # Weight: better rank -> larger weight
    out['Weight'] = 1.0 / out['Rank_predict']

    # Normalize within each Season+Week
    weight_sum = out.groupby(['Season', 'Week'])['Weight'].transform('sum')
    out['Percent_Predict'] = out['Weight'] / weight_sum * 100.0

    return out


def update_result_data(result_df, mc_df):
    mc_pred = build_normalized_percent(mc_df)
    mc_pred = mc_pred[['Season', 'Week', 'CelebrityName', 'Percent_Predict', 'Rank_predict']]

    merged = result_df.merge(
        mc_pred,
        on=['Season', 'Week', 'CelebrityName'],
        how='left',
        suffixes=('', '_mc'),
    )

    rank_mask = merged.get('RuleType', pd.Series(False, index=merged.index)) == 'Rank'

    if 'Predicted_Audience_Percent' in merged.columns and 'Percent_Predict_mc' in merged.columns:
        merged.loc[rank_mask, 'Predicted_Audience_Percent'] = merged.loc[rank_mask, 'Percent_Predict_mc']
        merged['Predicted_Audience_Percent'] = pd.to_numeric(
            merged['Predicted_Audience_Percent'], errors='coerce'
        ).round(2)

    if 'Predicted_Audience_Rank' in merged.columns and 'Rank_predict_mc' in merged.columns:
        merged.loc[rank_mask, 'Predicted_Audience_Rank'] = merged.loc[rank_mask, 'Rank_predict_mc']

    # Ensure per-week Predicted_Audience_Percent sums to exactly 100 after rounding for Rank rows
    if 'Predicted_Audience_Percent' in merged.columns and 'RuleType' in merged.columns:
        rank_groups = merged[merged['RuleType'] == 'Rank'].groupby(['Season', 'Week']).groups
        for (season, week), idx in rank_groups.items():
            group_vals = merged.loc[idx, 'Predicted_Audience_Percent']
            if not group_vals.notna().any():
                continue
            group_sum = group_vals.sum(skipna=True)
            diff = round(100.0 - group_sum, 2)
            if diff != 0:
                max_idx = group_vals.idxmax(skipna=True)
                merged.at[max_idx, 'Predicted_Audience_Percent'] = round(
                    merged.at[max_idx, 'Predicted_Audience_Percent'] + diff, 2
                )

    drop_cols = [c for c in merged.columns if c.endswith('_mc')]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    return merged


def main():
    parser = argparse.ArgumentParser(description='Update result_data CSV with Monte Carlo percent/rank predictions.')
    parser.add_argument('-m', '--mc-input', default=DEFAULT_MC_INPUT, help='Monte Carlo input CSV file')
    parser.add_argument('-r', '--result-data', default=DEFAULT_RESULT_DATA, help='Result data CSV file to update')
    parser.add_argument('-o', '--output', default=None, help='Optional output CSV file (defaults to result_data)')
    args = parser.parse_args()

    if not os.path.exists(args.mc_input):
        raise FileNotFoundError(f"Monte Carlo input file not found: {args.mc_input}")
    if not os.path.exists(args.result_data):
        raise FileNotFoundError(f"Result data file not found: {args.result_data}")

    mc_df = pd.read_csv(args.mc_input)
    result_df = pd.read_csv(args.result_data)

    out = update_result_data(result_df, mc_df)

    output_path = args.output or args.result_data
    out.to_csv(output_path, index=False)
    print(f"Wrote: {output_path}")


if __name__ == '__main__':
    main()

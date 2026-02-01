#!/usr/bin/env python3
import argparse
import csv
import math
import os
import statistics
from collections import defaultdict

def parse_float(value):
    if value is None:
        return None
    v = str(value).strip()
    if v == "" or v.upper() == "N/A":
        return None
    v = v.replace("%", "").replace(",", "")
    try:
        return float(v)
    except ValueError:
        return None

def minmax_norm(val, minv, maxv, invert=False):
    if val is None or minv is None or maxv is None:
        return None
    if maxv == minv:
        return 0.5
    norm = (val - minv) / (maxv - minv)
    if invert:
        norm = 1.0 - norm
    if norm < 0.0:
        norm = 0.0
    if norm > 1.0:
        norm = 1.0
    return norm

def safe_mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return statistics.mean(vals)

def safe_stdev(values):
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0
    return statistics.pstdev(vals)

def load_attributes(path):
    attrs = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("celebrity_name", "").strip()
            season = row.get("season", "").strip()
            if name == "" or season == "":
                continue
            try:
                season_i = int(float(season))
            except ValueError:
                continue
            attrs[(name, season_i)] = {
                "celebrity_name": name,
                "ballroom_partner": row.get("ballroom_partner", "").strip() or "Unknown",
                "celebrity_industry": row.get("celebrity_industry", "").strip() or "Unknown",
                "celebrity_homestate": row.get("celebrity_homestate", "").strip() or "Unknown",
                "celebrity_homecountry/region": row.get("celebrity_homecountry/region", "").strip() or "Unknown",
                "celebrity_age_during_season": parse_float(row.get("celebrity_age_during_season", "")),
            }
    return attrs

def load_results(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rule = row.get("RuleType", "").strip()
            if rule not in ("Rank", "Percent"):
                continue
            name = row.get("CelebrityName", "").strip()
            season = row.get("Season", "").strip()
            week = row.get("Week", "").strip()
            if name == "" or season == "" or week == "":
                continue
            try:
                season_i = int(float(season))
                week_i = int(float(week))
            except ValueError:
                continue

            if rule == "Rank":
                judge_raw = parse_float(row.get("JudgeScore_Normalization", ""))
                audience_raw = parse_float(row.get("Predicted_Audience_Rank", ""))
                invert = True
            else:
                judge_raw = parse_float(row.get("JudgeScore_Normalization", ""))
                audience_raw = parse_float(row.get("Predicted_Audience_Percent", ""))
                invert = False

            rows.append({
                "celebrity_name": name,
                "season": season_i,
                "week": week_i,
                "rule": rule,
                "judge_raw": judge_raw,
                "audience_raw": audience_raw,
                "invert": invert,
            })
    return rows

def build_group_minmax(rows):
    grouped = defaultdict(lambda: {"judge": [], "audience": []})
    for r in rows:
        key = (r["season"], r["week"], r["rule"])
        if r["judge_raw"] is not None:
            grouped[key]["judge"].append(r["judge_raw"])
        if r["audience_raw"] is not None:
            grouped[key]["audience"].append(r["audience_raw"])

    minmax = {}
    for key, vals in grouped.items():
        judge_vals = vals["judge"]
        aud_vals = vals["audience"]
        minmax[key] = {
            "judge": (min(judge_vals), max(judge_vals)) if judge_vals else (None, None),
            "audience": (min(aud_vals), max(aud_vals)) if aud_vals else (None, None),
        }
    return minmax

def normalize_rows(rows, minmax):
    for r in rows:
        key = (r["season"], r["week"], r["rule"])
        mm = minmax.get(key, {})
        jmin, jmax = mm.get("judge", (None, None))
        amin, amax = mm.get("audience", (None, None))
        r["judge_norm"] = minmax_norm(r["judge_raw"], jmin, jmax, invert=r["invert"])
        r["audience_norm"] = minmax_norm(r["audience_raw"], amin, amax, invert=r["invert"])
    return rows

def aggregate_by_celebrity(rows):
    agg = defaultdict(lambda: {"judge": [], "audience": [], "weeks": set(), "rule": None})
    for r in rows:
        key = (r["celebrity_name"], r["season"], r["rule"])
        agg[key]["rule"] = r["rule"]
        if r["judge_norm"] is not None:
            agg[key]["judge"].append(r["judge_norm"])
        if r["audience_norm"] is not None:
            agg[key]["audience"].append(r["audience_norm"])
        agg[key]["weeks"].add(r["week"])

    records = []
    for (name, season, rule), vals in agg.items():
        records.append({
            "celebrity_name": name,
            "season": season,
            "rule": rule,
            "judge_mean": safe_mean(vals["judge"]),
            "audience_mean": safe_mean(vals["audience"]),
            "weeks": len(vals["weeks"]),
        })
    return records

def join_attributes(records, attrs):
    joined = []
    missing = 0
    for r in records:
        key = (r["celebrity_name"], r["season"])
        attr = attrs.get(key)
        if not attr:
            missing += 1
            continue
        out = dict(r)
        out.update(attr)
        joined.append(out)
    return joined, missing

def compute_overall(records):
    overall = {}
    for rule in ("Rank", "Percent"):
        jvals = [r["judge_mean"] for r in records if r["rule"] == rule and r["judge_mean"] is not None]
        avals = [r["audience_mean"] for r in records if r["rule"] == rule and r["audience_mean"] is not None]
        overall[rule] = {
            "judge_mean": safe_mean(jvals),
            "audience_mean": safe_mean(avals),
            "n": len([r for r in records if r["rule"] == rule])
        }
    return overall

def linear_regression(xs, ys):
    vals = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(vals)
    if n < 2:
        return None
    xbar = sum(x for x, _ in vals) / n
    ybar = sum(y for _, y in vals) / n
    ss_xy = sum((x - xbar) * (y - ybar) for x, y in vals)
    ss_xx = sum((x - xbar) ** 2 for x, _ in vals)
    ss_yy = sum((y - ybar) ** 2 for _, y in vals)
    if ss_xx == 0:
        return None
    slope = ss_xy / ss_xx
    intercept = ybar - slope * xbar
    r = 0.0
    if ss_xx > 0 and ss_yy > 0:
        r = ss_xy / math.sqrt(ss_xx * ss_yy)
    return {
        "n": n,
        "slope": slope,
        "intercept": intercept,
        "r": r,
    }

def write_effects(records, overall, var_name, out_path):
    rows_out = []
    for rule in ("Rank", "Percent"):
        groups = defaultdict(list)
        for r in records:
            if r["rule"] != rule:
                continue
            val = r.get(var_name)
            if val is None or val == "":
                val = "Unknown"
            groups[val].append(r)

        for val, recs in groups.items():
            judge_vals = [x["judge_mean"] for x in recs if x["judge_mean"] is not None]
            aud_vals = [x["audience_mean"] for x in recs if x["audience_mean"] is not None]
            jmean = safe_mean(judge_vals)
            amean = safe_mean(aud_vals)
            rows_out.append({
                "RuleType": rule,
                "Value": val,
                "N": len(recs),
                "Mean_Judge_Norm": jmean,
                "Mean_Audience_Norm": amean,
                "Judge_Lift": None if jmean is None else jmean - overall[rule]["judge_mean"],
                "Audience_Lift": None if amean is None else amean - overall[rule]["audience_mean"],
                "Judge_Std": safe_stdev(judge_vals),
                "Audience_Std": safe_stdev(aud_vals),
            })

    def sort_key(row):
        if isinstance(row["Value"], (int, float)):
            return (row["RuleType"], row["Value"])
        return (row["RuleType"], str(row["Value"]))

    rows_out.sort(key=sort_key)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "RuleType", "Value", "N",
            "Mean_Judge_Norm", "Mean_Audience_Norm",
            "Judge_Lift", "Audience_Lift",
            "Judge_Std", "Audience_Std",
        ])
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

def write_overall(overall, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["RuleType", "N", "Mean_Judge_Norm", "Mean_Audience_Norm"])
        writer.writeheader()
        for rule, vals in overall.items():
            writer.writerow({
                "RuleType": rule,
                "N": vals["n"],
                "Mean_Judge_Norm": vals["judge_mean"],
                "Mean_Audience_Norm": vals["audience_mean"],
            })

def write_age_linear(records, out_path):
    rows_out = []
    for rule in ("Rank", "Percent"):
        xs = [r.get("celebrity_age_during_season") for r in records if r["rule"] == rule]
        ys_j = [r.get("judge_mean") for r in records if r["rule"] == rule]
        ys_a = [r.get("audience_mean") for r in records if r["rule"] == rule]
        reg_j = linear_regression(xs, ys_j)
        reg_a = linear_regression(xs, ys_a)
        if reg_j:
            rows_out.append({
                "RuleType": rule,
                "Target": "Judge",
                "N": reg_j["n"],
                "Slope": reg_j["slope"],
                "Intercept": reg_j["intercept"],
                "Correlation": reg_j["r"],
            })
        if reg_a:
            rows_out.append({
                "RuleType": rule,
                "Target": "Audience",
                "N": reg_a["n"],
                "Slope": reg_a["slope"],
                "Intercept": reg_a["intercept"],
                "Correlation": reg_a["r"],
            })

    if not rows_out:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["RuleType", "Target", "N", "Slope", "Intercept", "Correlation"])
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

def sanitize_filename(name):
    return name.replace("/", "_")

def main():
    parser = argparse.ArgumentParser(description="Compute detailed single-variable effects for DWTS data.")
    parser.add_argument("--data", required=True, help="Path to 2026_MCM_Problem_C_Data.csv")
    parser.add_argument("--results", required=True, help="Path to MCM_Problem_C_Results_20260131_2256.csv")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args()

    attrs = load_attributes(args.data)
    rows = load_results(args.results)
    minmax = build_group_minmax(rows)
    normalize_rows(rows, minmax)
    agg = aggregate_by_celebrity(rows)
    joined, missing = join_attributes(agg, attrs)

    overall = compute_overall(joined)

    write_overall(overall, os.path.join(args.outdir, "overall_means.csv"))

    variables = [
        "celebrity_name",
        "ballroom_partner",
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry/region",
        "celebrity_age_during_season",
    ]

    for var in variables:
        safe_var = sanitize_filename(var)
        out_path = os.path.join(args.outdir, f"effects_{safe_var}.csv")
        write_effects(joined, overall, var, out_path)

    write_age_linear(joined, os.path.join(args.outdir, "age_linear_effects.csv"))

    if missing:
        print(f"Warning: {missing} aggregated rows missing attributes and were skipped.")
    print(f"Done. Outputs written to {args.outdir}")

if __name__ == "__main__":
    main()

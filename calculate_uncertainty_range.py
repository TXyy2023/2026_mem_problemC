import argparse
import csv
import os


DEFAULT_PREFIX = "MCM_Problem_C_Results_uncertainty.csv"
DEFAULT_EXT = ".csv"


def _select_latest_results_file():
    candidates = []
    for name in os.listdir("."):
        if name.startswith(DEFAULT_PREFIX) and name.endswith(DEFAULT_EXT):
            candidates.append(name)
    if not candidates:
        return None
    return max(candidates, key=lambda p: os.path.getmtime(p))


def _parse_range(range_str):
    if range_str is None:
        return None
    text = str(range_str).strip()
    if not text:
        return None
    clean = text.replace("%", "").replace(",", "").strip()
    parts = [p.strip() for p in clean.split("-") if p.strip() != ""]
    if len(parts) < 2:
        return None
    try:
        low = float(parts[0])
        high = float(parts[1])
    except Exception:
        return None
    if low > high:
        low, high = high, low
    return low, high


def _build_output_row(row, low, high, width, index):
    return {
        "CelebrityName": row.get("CelebrityName", ""),
        "Season": row.get("Season", ""),
        "Week": row.get("Week", ""),
        "RuleType": row.get("RuleType", ""),
        "Possible_Audience_Vote_Range": row.get("Possible_Audience_Vote_Range", ""),
        "Uncertainty_Range_Min": "" if low is None else f"{low:.4f}".rstrip("0").rstrip("."),
        "Uncertainty_Range_Max": "" if high is None else f"{high:.4f}".rstrip("0").rstrip("."),
        "Uncertainty_Range_Width": "" if width is None else f"{width:.4f}".rstrip("0").rstrip("."),
        "Uncertainty_Index": "" if index is None else f"{index:.6f}".rstrip("0").rstrip("."),
    }


def main(input_path=None, output_path=None):
    if input_path is None:
        input_path = _select_latest_results_file()
    if input_path is None or not os.path.exists(input_path):
        raise FileNotFoundError("No results CSV found. Provide --input.")

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_uncertainty{ext}"

    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Pre-compute participants per Season+Week+RuleType for rank normalization.
    participants_map = {}
    for row in rows:
        season = str(row.get("Season", "")).strip()
        week = str(row.get("Week", "")).strip()
        rule = str(row.get("RuleType", "")).strip().casefold()
        key = (season, week, rule)
        name = str(row.get("CelebrityName", "")).strip()
        if not name:
            continue
        participants_map.setdefault(key, set()).add(name)

    output_rows = []
    for row in rows:
        range_str = row.get("Possible_Audience_Vote_Range", "")
        parsed = _parse_range(range_str)
        low = high = width = index = None
        if parsed is not None:
            low, high = parsed
            width = high - low
            rule = str(row.get("RuleType", "")).strip().casefold()
            is_percent = (rule == "percent") or ("%" in str(range_str))
            if is_percent:
                index = width / 100.0
            else:
                season = str(row.get("Season", "")).strip()
                week = str(row.get("Week", "")).strip()
                key = (season, week, rule)
                participants = len(participants_map.get(key, []))
                denom = (participants - 1) if participants and participants > 1 else None
                index = (width / denom) if denom else None

        output_rows.append(_build_output_row(row, low, high, width, index))

    fieldnames = [
        "CelebrityName",
        "Season",
        "Week",
        "RuleType",
        "Possible_Audience_Vote_Range",
        "Uncertainty_Range_Min",
        "Uncertainty_Range_Max",
        "Uncertainty_Range_Width",
        "Uncertainty_Index",
    ]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Uncertainty metrics saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute uncertainty metrics from Possible_Audience_Vote_Range.")
    parser.add_argument("--input", dest="input_path", type=str, default="MCM_Problem_C_Results_20260201_1748.csv", help="Path to results CSV.")
    parser.add_argument("--output", dest="output_path", type=str, default="MCM_Problem_C_Results_20260201_1748_uncertainty.csv", help="Path to output CSV.")
    args = parser.parse_args()
    main(input_path=args.input_path, output_path=args.output_path)

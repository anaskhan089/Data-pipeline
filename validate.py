"""
validate.py - Step 3 of the pipeline: clean + validate the raw CSV files.

Reads   data/raw/*.csv
Writes  data/processed/<table>_valid.csv      records that are safe to load
        data/processed/<table>_rejected.csv   records with a reject_reason
        data/processed/validation_issues.csv  every finding (error / warning / fixed)

Three levels of findings:
  fixed    format problem we can repair safely (e.g. 31.03.2025 -> 2025-03-31)
  warning  record is usable but incomplete (e.g. empty comment)
  error    record is NOT safe to use -> rejected, owner must correct it

Rule: we never invent values. A progress of 130% is rejected, not "fixed" to 100.

Usage:
  python validate.py
  python validate.py --raw data/raw --out data/processed
"""
import argparse
import os

import pandas as pd


INT_COLS = ["budget_eur", "budget_spent_eur", "forecast_final_cost_eur", "delay_days",
            "severity", "likelihood", "risk_score"]


class Issues:
    """Collects findings for one table."""

    def __init__(self, table, id_col, df):
        self.table, self.id_col, self.df = table, id_col, df
        self.rows = []

    def flag(self, mask, issue, level="error"):
        mask = mask.fillna(False).astype(bool)
        for i in self.df.index[mask]:
            self.rows.append({
                "table": self.table, "row": i,
                "record_id": self.df.at[i, self.id_col],
                "level": level, "issue": issue,
            })

    def frame(self):
        return pd.DataFrame(self.rows, columns=["table", "row", "record_id", "level", "issue"])


# ---------- helpers ----------
def read_table(raw_dir, name):
    # Read everything as text first, so bad values never crash the load.
    return pd.read_csv(os.path.join(raw_dir, f"{name}.csv"), dtype=str)


def tidy_text(df):
    """CLEANING: trim spaces, turn empty strings into real missing values."""
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df.replace({"": pd.NA})


def to_number(df, col, issues):
    """CLEANING: text -> number. Non-numeric text is an error."""
    raw = df[col]
    num = pd.to_numeric(raw, errors="coerce")
    issues.flag(raw.notna() & num.isna(), f"{col}_not_numeric")
    df[col] = num


def fix_dates(df, col, issues):
    """CLEANING: accept YYYY-MM-DD and DD.MM.YYYY, store as YYYY-MM-DD."""
    iso = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
    de = pd.to_datetime(df[col], format="%d.%m.%Y", errors="coerce")
    issues.flag(iso.isna() & de.notna(), "bad_date_format", level="fixed")
    issues.flag(iso.isna() & de.isna(), "bad_date_format", level="error")
    df[col] = iso.fillna(de).dt.strftime("%Y-%m-%d")


def split_valid_rejected(df, issues_df):
    errors = issues_df[issues_df.level == "error"]
    bad_rows = sorted(set(errors.row))
    rejected = df.loc[bad_rows].copy()
    rejected["reject_reason"] = errors.groupby("row").issue.apply(lambda s: "; ".join(s))
    valid = df.drop(index=bad_rows)
    return valid, rejected


# ---------- table validators ----------
def validate_projects(df):
    df = tidy_text(df.copy())
    iss = Issues("projects", "project_id", df)
    to_number(df, "budget_eur", iss)
    fix_dates(df, "start_date", iss)
    fix_dates(df, "planned_end_date", iss)
    iss.flag(df.project_id.duplicated(keep="first"), "duplicate_row")
    iss.flag(df.budget_eur <= 0, "budget_not_positive")
    iss.flag(df.planned_end_date < df.start_date, "end_before_start")
    iss.flag(df.project_owner.isna(), "missing_owner", level="warning")
    return df, iss.frame()


def validate_reports(df, project_ids):
    df = tidy_text(df.copy())
    iss = Issues("monthly_reports", "report_id", df)
    for col in ["planned_progress_pct", "actual_progress_pct", "budget_spent_eur",
                "forecast_final_cost_eur", "delay_days"]:
        to_number(df, col, iss)
    fix_dates(df, "report_month", iss)

    iss.flag(df.actual_progress_pct.isna(), "missing_progress")
    iss.flag(df.actual_progress_pct > 100, "progress_over_100")
    iss.flag(df.actual_progress_pct < 0, "progress_negative")
    iss.flag(df.budget_spent_eur < 0, "negative_budget_spent")
    iss.flag(df.report_id.duplicated(keep="first"), "duplicate_row")
    iss.flag(~df.project_id.isin(project_ids), "unknown_project")
    iss.flag(df.status_comment.isna(), "missing_comment", level="warning")
    return df, iss.frame()


def validate_risks(df, project_ids):
    df = tidy_text(df.copy())
    iss = Issues("risks", "risk_id", df)
    for col in ["severity", "likelihood", "risk_score"]:
        to_number(df, col, iss)
    fix_dates(df, "raised_month", iss)

    bad_sev = ~df.severity.between(1, 5)
    bad_lik = ~df.likelihood.between(1, 5)
    iss.flag(bad_sev & df.severity.notna(), "severity_out_of_range")
    iss.flag(bad_lik & df.likelihood.notna(), "likelihood_out_of_range")
    # Only check the score when inputs are valid (avoids double-flagging).
    ok_inputs = ~bad_sev & ~bad_lik
    iss.flag(ok_inputs & (df.risk_score != df.severity * df.likelihood), "risk_score_mismatch")
    iss.flag(df.risk_id.duplicated(keep="first"), "duplicate_row")
    iss.flag(~df.project_id.isin(project_ids), "unknown_project")
    iss.flag(df.mitigation.isna(), "missing_mitigation", level="warning")
    return df, iss.frame()


def validate_gates(df, project_ids):
    df = tidy_text(df.copy())
    iss = Issues("gate_reviews", "gate_review_id", df)
    fix_dates(df, "review_date", iss)
    iss.flag(~df.decision.isin(["go", "conditional", "hold"]), "invalid_decision")
    iss.flag(~df.project_id.isin(project_ids), "unknown_project")
    return df, iss.frame()


# ---------- self-check against the answer key ----------
def compare_with_answer_key(all_issues, log_path):
    if not os.path.exists(log_path):
        return
    log = pd.read_csv(log_path)
    injected = set(zip(log.table, log.record_id, log.issue))
    found = set(zip(all_issues.table, all_issues.record_id, all_issues.issue))
    print("\nSelf-check against dq_issues_log.csv (answer key)")
    print(f"{'issue':<26}{'injected':>9}{'caught':>8}")
    for issue in sorted(set(log.issue)):
        inj = {x for x in injected if x[2] == issue}
        print(f"{issue:<26}{len(inj):>9}{len(inj & found):>8}")
    missed = injected - found
    extra = found - injected
    print(f"\nMissed: {len(missed)} | Extra findings not in the log: {len(extra)}")
    for m in sorted(missed)[:10]:
        print("  missed:", m)
    for e in sorted(extra)[:10]:
        print("  extra :", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    projects, p_iss = validate_projects(read_table(args.raw, "projects"))
    ok_projects = set(projects.loc[~projects.index.isin(
        p_iss[p_iss.level == "error"].row), "project_id"])

    results = {"projects": (projects, p_iss)}
    results["monthly_reports"] = validate_reports(read_table(args.raw, "monthly_reports"), ok_projects)
    results["risks"] = validate_risks(read_table(args.raw, "risks"), ok_projects)
    results["gate_reviews"] = validate_gates(read_table(args.raw, "gate_reviews"), ok_projects)

    all_issues = []
    print(f"{'table':<18}{'in':>6}{'valid':>7}{'rejected':>10}{'warnings':>10}{'fixed':>7}")
    for name, (df, iss) in results.items():
        valid, rejected = split_valid_rejected(df, iss)
        for c in INT_COLS:  # whole numbers must be written without ".0" for BigQuery
            if c in valid.columns:
                valid[c] = valid[c].round().astype("Int64")
        valid.to_csv(os.path.join(args.out, f"{name}_valid.csv"), index=False)
        rejected.to_csv(os.path.join(args.out, f"{name}_rejected.csv"), index=False)
        all_issues.append(iss)
        lv = iss.level.value_counts()
        print(f"{name:<18}{len(df):>6}{len(valid):>7}{len(rejected):>10}"
              f"{lv.get('warning', 0):>10}{lv.get('fixed', 0):>7}")

    all_issues = pd.concat(all_issues, ignore_index=True)
    all_issues.drop(columns="row").to_csv(os.path.join(args.out, "validation_issues.csv"), index=False)
    print(f"\nFiles written to: {os.path.abspath(args.out)}")
    compare_with_answer_key(all_issues, os.path.join(args.raw, "dq_issues_log.csv"))


if __name__ == "__main__":
    main()

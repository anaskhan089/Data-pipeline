"""
Synthetic data generator: construction project reporting pipeline.

Creates 5 CSV files in --out (default: ./data/raw):
  projects.csv          master data (20 projects)
  monthly_reports.csv   monthly project-owner questionnaires
  risks.csv             risk register
  gate_reviews.csv      gate review decisions
  zones.csv             placeholder for phase 2 (BIM / point cloud progress)
  dq_issues_log.csv     answer key: which dirty records were injected

Deliberately dirty records (missing values, duplicates, bad dates,
impossible values) are injected so your validation layer has real work to do.

Usage:
  python generate_data.py                  # clean + dirty mix, seed 42
  python generate_data.py --no-dirty       # fully clean data
  python generate_data.py --seed 7 --projects 30
"""
import argparse
import math
import os
import random
from datetime import date

import numpy as np
import pandas as pd

REPORT_END = date(2026, 8, 31)  # last reporting month

PROJECT_TYPES = ["Residential", "Commercial", "Infrastructure", "Industrial", "Public building"]
CITIES = ["Dresden", "Leipzig", "Berlin", "Munich", "Hamburg", "Frankfurt", "Stuttgart", "Cologne"]
OWNERS = ["A. Becker", "M. Schneider", "L. Fischer", "J. Weber", "S. Hoffmann",
          "T. Wagner", "K. Richter", "N. Klein", "P. Wolf", "E. Neumann"]
BUILDERS = ["Nordbau GmbH", "Elbe Construction AG", "Steinwerk Bau", "Alpha Bau Group",
            "Rheinland Hochbau", "Saxon Builders GmbH"]

RISK_CATEGORIES = {
    "Schedule": ["Delayed permit approval", "Subcontractor delay", "Late design changes"],
    "Cost": ["Material price increase", "Scope creep", "Currency / inflation impact"],
    "Supply chain": ["Steel delivery delay", "Concrete supplier shortage", "Long lead time for facade elements"],
    "Weather": ["Prolonged frost period", "Heavy rain delaying groundwork"],
    "Safety": ["Scaffolding inspection finding", "Site accident near-miss"],
    "Resources": ["Skilled labour shortage", "Key site manager left"],
}
MITIGATIONS = [
    "Escalated to steering committee", "Alternative supplier being qualified",
    "Schedule buffer added", "Additional crew contracted",
    "Weekly monitoring with builder", "Change request submitted",
]
COMMENTS = {
    "on_track": ["Works proceeding as planned.", "No major issues this month.",
                 "Milestone reached on time.", "Minor weather impact, recovered."],
    "slipping": ["Some delays due to subcontractor availability.", "Permit still pending, small impact on schedule.",
                 "Material delivery late, workaround in place.", "Progress slightly behind plan."],
    "troubled": ["Significant delay, recovery plan needed.", "Budget overrun expected, escalation required.",
                 "Major supplier problem, work stopped in parts.", "Design changes causing rework."],
}
GATES = [("G1", "Concept approved", 0.0), ("G2", "Design approved", 0.15),
         ("G3", "Construction start", 0.35), ("G4", "Handover readiness", 0.90)]


def s_curve(t):
    """Planned progress (0..1) for normalised time t in 0..1 (smooth S-shape)."""
    t = min(max(t, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def month_range(start, end):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield date(y, m, 1)
        m += 1
        if m > 12:
            y, m = y + 1, 1


def build_projects(n, rng):
    rows = []
    profiles = rng.choice(["on_track", "slipping", "troubled"], size=n, p=[0.5, 0.3, 0.2])
    for i in range(n):
        start = date(2024, 1, 1)
        start = date(2024 + rng.integers(0, 2), int(rng.integers(1, 13)), 1)
        duration = int(rng.integers(18, 37))
        end_y, end_m = divmod(start.month - 1 + duration, 12)
        planned_end = date(start.year + end_y, end_m + 1, 1)
        rows.append({
            "project_id": f"P{i + 1:03d}",
            "project_name": f"{rng.choice(CITIES)} {rng.choice(PROJECT_TYPES)} {i + 1:02d}",
            "project_type": rng.choice(PROJECT_TYPES),
            "city": rng.choice(CITIES),
            "project_owner": rng.choice(OWNERS),
            "builder": rng.choice(BUILDERS),
            "start_date": start,
            "planned_end_date": planned_end,
            "budget_eur": int(round(rng.uniform(5, 120) * 1e6, -4)),
            "_profile": profiles[i],  # internal only, dropped before saving
        })
    return pd.DataFrame(rows)


def build_reports(projects, rng):
    rows = []
    cost_factor = {"on_track": 1.00, "slipping": 1.08, "troubled": 1.22}
    lag = {"on_track": 0.00, "slipping": 0.12, "troubled": 0.28}
    for _, p in projects.iterrows():
        months = list(month_range(p.start_date, min(REPORT_END, p.planned_end_date)))
        total = max(len(months), 1)
        for k, m in enumerate(months):
            t = (k + 1) / total
            planned = s_curve(t) * 100
            drift = lag[p._profile] * (k + 1) / total  # gap grows over time
            actual = max(0, planned * (1 - drift) + rng.normal(0, 1.2))
            actual = min(actual, 100)
            cf = 1 + (cost_factor[p._profile] - 1) * (k + 1) / total
            spent = p.budget_eur * (actual / 100) * cf * rng.uniform(0.97, 1.03)
            forecast = p.budget_eur * cf * rng.uniform(0.98, 1.05)
            rows.append({
                "report_id": f"R{len(rows) + 1:05d}",
                "project_id": p.project_id,
                "report_month": m,
                "planned_progress_pct": round(planned, 1),
                "actual_progress_pct": round(actual, 1),
                "budget_spent_eur": int(spent),
                "forecast_final_cost_eur": int(forecast),
                "delay_days": int(max(0, (planned - actual) * 3 + rng.normal(0, 2))),
                "status_comment": rng.choice(COMMENTS[p._profile]),
                "submitted_on_time": bool(rng.random() > (0.05 if p._profile == "on_track" else 0.25)),
            })
    return pd.DataFrame(rows)


def build_risks(projects, rng):
    rows = []
    for _, p in projects.iterrows():
        n_risks = int({"on_track": rng.integers(2, 5), "slipping": rng.integers(4, 8),
                       "troubled": rng.integers(6, 11)}[p._profile])
        months = list(month_range(p.start_date, min(REPORT_END, p.planned_end_date)))
        for _ in range(n_risks):
            cat = rng.choice(list(RISK_CATEGORIES))
            hi = {"on_track": 3, "slipping": 4, "troubled": 5}[p._profile]
            severity = int(rng.integers(1, hi + 1))
            likelihood = int(rng.integers(1, hi + 1))
            rows.append({
                "risk_id": f"K{len(rows) + 1:05d}",
                "project_id": p.project_id,
                "raised_month": months[int(rng.integers(0, len(months)))],
                "category": cat,
                "description": rng.choice(RISK_CATEGORIES[cat]),
                "severity": severity,
                "likelihood": likelihood,
                "risk_score": severity * likelihood,
                "mitigation": rng.choice(MITIGATIONS),
                "status": rng.choice(["open", "mitigating", "closed"], p=[0.4, 0.3, 0.3]),
            })
    return pd.DataFrame(rows)


def build_gates(projects, reports, rng):
    rows = []
    for _, p in projects.iterrows():
        rep = reports[reports.project_id == p.project_id]
        for gate, name, threshold in GATES:
            reached = rep[rep.actual_progress_pct >= threshold * 100]
            if reached.empty:
                continue
            gate_date = reached.report_month.iloc[0]
            if p._profile == "on_track":
                dec = rng.choice(["go", "conditional"], p=[0.85, 0.15])
            elif p._profile == "slipping":
                dec = rng.choice(["go", "conditional", "hold"], p=[0.45, 0.4, 0.15])
            else:
                dec = rng.choice(["go", "conditional", "hold"], p=[0.2, 0.4, 0.4])
            rows.append({
                "gate_review_id": f"G{len(rows) + 1:04d}",
                "project_id": p.project_id,
                "gate": gate,
                "gate_name": name,
                "review_date": gate_date,
                "decision": dec,
            })
    return pd.DataFrame(rows)


def build_zones(projects, rng):
    """Placeholder for phase 2: element/zone level progress (BIM vs scan)."""
    rows = []
    for _, p in projects.iterrows():
        for z in ["Foundation", "Structure", "Envelope", "MEP", "Interior"]:
            rows.append({"zone_id": f"{p.project_id}-{z[:3].upper()}",
                         "project_id": p.project_id, "zone_name": z,
                         "bim_element_count": int(rng.integers(50, 800)),
                         "built_pct": None})  # filled in phase 2
    return pd.DataFrame(rows)


def inject_dirt(projects, reports, risks, rng):
    """Corrupt a small share of records and log every change (answer key)."""
    log = []
    reports = reports.copy()
    reports["report_month"] = reports["report_month"].astype(object)
    idx = rng.permutation(reports.index)

    def take(n):
        nonlocal idx
        out, idx = idx[:n], idx[n:]
        return out

    n = max(3, int(len(reports) * 0.02))
    for i in take(n):  # missing progress
        log.append(("monthly_reports", reports.at[i, "report_id"], "missing_progress"))
        reports.at[i, "actual_progress_pct"] = np.nan
    for i in take(n):  # impossible progress
        log.append(("monthly_reports", reports.at[i, "report_id"], "progress_over_100"))
        reports.at[i, "actual_progress_pct"] = float(rng.integers(105, 140))
    for i in take(n):  # negative budget
        log.append(("monthly_reports", reports.at[i, "report_id"], "negative_budget_spent"))
        reports.at[i, "budget_spent_eur"] = -abs(reports.at[i, "budget_spent_eur"])
    for i in take(n):  # wrong date format
        d = reports.at[i, "report_month"]
        log.append(("monthly_reports", reports.at[i, "report_id"], "bad_date_format"))
        reports.at[i, "report_month"] = d.strftime("%d.%m.%Y")
    for i in take(n):  # empty comment
        log.append(("monthly_reports", reports.at[i, "report_id"], "missing_comment"))
        reports.at[i, "status_comment"] = ""
    dups = reports.loc[take(n)]
    for rid in dups.report_id:
        log.append(("monthly_reports", rid, "duplicate_row"))
    reports = pd.concat([reports, dups], ignore_index=True)

    risks = risks.copy()
    for i in rng.choice(risks.index, size=max(2, int(len(risks) * 0.03)), replace=False):
        log.append(("risks", risks.at[i, "risk_id"], "severity_out_of_range"))
        risks.at[i, "severity"] = 9
    for i in rng.choice(risks.index, size=max(2, int(len(risks) * 0.03)), replace=False):
        log.append(("risks", risks.at[i, "risk_id"], "missing_mitigation"))
        risks.at[i, "mitigation"] = None

    projects = projects.copy()
    for i in rng.choice(projects.index, size=2, replace=False):
        log.append(("projects", projects.at[i, "project_id"], "missing_owner"))
        projects.at[i, "project_owner"] = None

    return projects, reports, risks, pd.DataFrame(log, columns=["table", "record_id", "issue"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--projects", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-dirty", action="store_true", help="skip injecting bad records")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    projects = build_projects(args.projects, rng)
    reports = build_reports(projects, rng)
    risks = build_risks(projects, rng)
    gates = build_gates(projects, reports, rng)
    zones = build_zones(projects, rng)

    dq = pd.DataFrame(columns=["table", "record_id", "issue"])
    if not args.no_dirty:
        projects, reports, risks, dq = inject_dirt(projects, reports, risks, rng)

    os.makedirs(args.out, exist_ok=True)
    projects.drop(columns="_profile").to_csv(f"{args.out}/projects.csv", index=False)
    reports.to_csv(f"{args.out}/monthly_reports.csv", index=False)
    risks.to_csv(f"{args.out}/risks.csv", index=False)
    gates.to_csv(f"{args.out}/gate_reviews.csv", index=False)
    zones.to_csv(f"{args.out}/zones.csv", index=False)
    dq.to_csv(f"{args.out}/dq_issues_log.csv", index=False)

    print(f"projects: {len(projects)} | reports: {len(reports)} | risks: {len(risks)} "
          f"| gate reviews: {len(gates)} | zones: {len(zones)} | injected issues: {len(dq)}")
    print(f"Files written to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

"""
load_to_bigquery.py - Step 4: processed CSVs -> GCS -> BigQuery.

Reads   data/processed/*_valid.csv and *_rejected.csv   (output of validate.py)
Uploads them to  gs://<BUCKET>/processed/
Loads them into  BigQuery dataset construction_reporting:
    projects, monthly_reports, risks, gate_reviews      (clean tables, typed)
    <table>_rejected                                    (quarantine, all text)

Safe to re-run: every load REPLACES the table (WRITE_TRUNCATE), so no duplicates.

Needs config.py with PROJECT_ID, BUCKET, REGION and a prior
`gcloud auth application-default login`.

Usage:
  python load_to_bigquery.py
"""
import csv
import os

from google.cloud import bigquery, storage

from config import BUCKET, PROJECT_ID, REGION

DATASET = "construction_reporting"
PROCESSED_DIR = "data/processed"
GCS_PREFIX = "processed"

S = bigquery.SchemaField
SCHEMAS = {
    "projects": [
        S("project_id", "STRING"), S("project_name", "STRING"), S("project_type", "STRING"),
        S("city", "STRING"), S("project_owner", "STRING"), S("builder", "STRING"),
        S("start_date", "DATE"), S("planned_end_date", "DATE"), S("budget_eur", "INT64"),
    ],
    "monthly_reports": [
        S("report_id", "STRING"), S("project_id", "STRING"), S("report_month", "DATE"),
        S("planned_progress_pct", "FLOAT64"), S("actual_progress_pct", "FLOAT64"),
        S("budget_spent_eur", "INT64"), S("forecast_final_cost_eur", "INT64"),
        S("delay_days", "INT64"), S("status_comment", "STRING"), S("submitted_on_time", "BOOL"),
    ],
    "risks": [
        S("risk_id", "STRING"), S("project_id", "STRING"), S("raised_month", "DATE"),
        S("category", "STRING"), S("description", "STRING"), S("severity", "INT64"),
        S("likelihood", "INT64"), S("risk_score", "INT64"), S("mitigation", "STRING"),
        S("status", "STRING"),
    ],
    "gate_reviews": [
        S("gate_review_id", "STRING"), S("project_id", "STRING"), S("gate", "STRING"),
        S("gate_name", "STRING"), S("review_date", "DATE"), S("decision", "STRING"),
    ],
}


def upload_to_gcs(files):
    """Copy local files into the bucket's processed/ folder."""
    bucket = storage.Client(project=PROJECT_ID).bucket(BUCKET)
    for name in files:
        bucket.blob(f"{GCS_PREFIX}/{name}").upload_from_filename(os.path.join(PROCESSED_DIR, name))
        print(f"  uploaded gs://{BUCKET}/{GCS_PREFIX}/{name}")


def load_table(bq, filename, table, schema):
    """Load one CSV from GCS into one BigQuery table (replacing old content)."""
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        allow_quoted_newlines=True,
    )
    table_id = f"{PROJECT_ID}.{DATASET}.{table}"
    uri = f"gs://{BUCKET}/{GCS_PREFIX}/{filename}"
    bq.load_table_from_uri(uri, table_id, job_config=job_config).result()
    print(f"  {table:<28}{bq.get_table(table_id).num_rows:>6} rows")


def header_as_strings(path):
    """Schema for rejected files: every column is text (they hold bad values)."""
    with open(path, newline="") as f:
        return [S(col, "STRING") for col in next(csv.reader(f))]


def main():
    files = []
    for name in SCHEMAS:
        files += [f"{name}_valid.csv", f"{name}_rejected.csv"]
    missing = [f for f in files if not os.path.exists(os.path.join(PROCESSED_DIR, f))]
    if missing:
        raise SystemExit(f"Missing files {missing}. Run validate.py first.")

    print("1) Uploading to GCS")
    upload_to_gcs(files)

    print("2) Creating dataset (if needed)")
    bq = bigquery.Client(project=PROJECT_ID, location=REGION)
    dataset = bigquery.Dataset(f"{PROJECT_ID}.{DATASET}")
    dataset.location = REGION
    bq.create_dataset(dataset, exists_ok=True)

    print("3) Loading into BigQuery")
    for name, schema in SCHEMAS.items():
        load_table(bq, f"{name}_valid.csv", name, schema)
        rejected_schema = header_as_strings(os.path.join(PROCESSED_DIR, f"{name}_rejected.csv"))
        load_table(bq, f"{name}_rejected.csv", f"{name}_rejected", rejected_schema)

    print(f"\nDone. In the console: BigQuery > {PROJECT_ID} > {DATASET}")


if __name__ == "__main__":
    main()

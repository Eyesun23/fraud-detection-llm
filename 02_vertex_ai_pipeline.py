"""
Enron Fraud Detection Pipeline — Vertex AI Batch
Focus: Earnings Manipulation / Concealment of Financial Losses

Usage:
    python enron3.py

Requirements:
    pip install google-cloud-aiplatform google-cloud-storage requests pandas
"""

import json
import os
import re
import time
import uuid
import pandas as pd
from pathlib import Path

import vertexai
from vertexai.batch_prediction import BatchPredictionJob
from google.cloud import storage


GCP_PROJECT     = os.environ.get("GCP_PROJECT", "enron-project23")
GCP_LOCATION    = os.environ.get("GCP_LOCATION", "us-central1")
GCS_BUCKET      = os.environ.get("GCS_BUCKET", "enron-project23-batch")

MODEL           = "gemini-2.5-flash"       
HIGH_RISK_THRESHOLD  = 18       # Wn ≥ 18 = High Probability
SUSPICIOUS_THRESHOLD = 12       # 12 ≤ Wn < 18 = Suspicious
DATE_FROM = "1999-01-01"     # Focus window start (LJM/Raptors creation)
DATE_TO   = "2002-12-31"     # Focus window end (post-bankruptcy filings)

CORPUS_PATH     = "data/emros_curpose.json" 
SAVE_PATH       = Path("data")


LEXICONS = {
    "deception": [
        "off-balance", "off balance", "raptor", "chewco", "ljm",
        "not for public", "mark to market", "mark-to-market",
        "special purpose", "spe", "creative compliance", "move offline",
        "don't cc", "do not cc", "mislead", "hide losses", "hiding losses",
        "conceal", "inflate earnings", "manipulate earnings",
        "earnings manipulation", "restate", "restatement",
        "accounting irregularity", "cook the books", "window dressing",
        "adjust the numbers", "adjusting numbers",
    ],
    "coercion": [
        "prc", "rank and yank", "by eod", "by end of day",
        "strictly confidential", "under the radar", "keep this between",
        "don't tell", "performance review", "your job", "your career",
        "whistleblower", "silence", "don't loop in legal",
        "exclude legal", "don't cc legal", "don't forward",
    ],
    "self_dealing": [
        "personal equity", "side deal", "fastow", "allocation",
        "friends and family", "personal account", "my account",
        "related party", "conflict of interest", "management fee",
        "insider trading", "exercise options", "private placement",
        "fee arrangement", "kickback",
    ],
    "manipulation": [
        "death star", "fat boy", "ricochet", "get shorty",
        "load shift", "load shifting", "blackout", "congestion",
        "artificial shortage", "wash trade", "wash trading",
        "round trip", "price cap", "price caps", "price fixing",
        "phantom load", "phantom energy", "scheduling error",
        "ferc", "iso",
    ],
}


def keyword_score(text: str) -> dict:
    lower = text.lower()
    hits, matched = {}, {}
    for cat, terms in LEXICONS.items():
        found = [t for t in terms if t in lower]
        hits[cat] = len(found)
        matched[cat] = found
    total = sum(hits.values())
    cats_hit = sum(1 for v in hits.values() if v > 0)
    return {"hits": hits, "matched": matched, "total": total, "cats_hit": cats_hit}




RUBRIC = {
    "q1":  "Does this email discuss adjusting financial numbers, using SPEs/entities like Raptors or LJM to hide financial losses, or manipulating mark-to-market valuations?",
    "q2":  "Is there an instruction to move the conversation offline, keep it under the radar, or exclude Legal/Compliance?",
    "q3":  "Is the email exploring creative compliance or loopholes to justify questionable accounting or avoid a restatement?",
    "q4":  "Does this email use pressure tactics, ultimatums, or threats to force an action related to financial reporting?",
    "q5":  "Is there an attempt to silence a whistleblower or suppress internal concerns about accounting irregularities?",
    "q6":  "Does this involve self-dealing — related parties (LJM/Fastow), personal enrichment, or conflicts of interest?",
    "q7":  "Is there evidence of using non-public financial information to make personal investment decisions (insider trading)?",
    "q8":  "Does the email mention market manipulation strategies (Death Star, Fat Boy, Ricochet, load shifting) to influence energy prices?",
    "q9":  "Is there discussion of coordinating to create artificial congestion, wash trades, or bypass FERC price caps?",
    "q10": "Does the email show awareness that the activity is illegal, improper, or needs to be concealed from regulators?",
}

SYSTEM_PROMPT = (
    "You are a senior forensic investigator specializing in corporate fraud. "
    "Evaluate the provided email strictly against the rubric. "
    "Respond ONLY with valid JSON — no markdown, no preamble."
)


def build_prompt(email: dict) -> str:
    questions = "\n".join(f"{k}: {v}" for k, v in RUBRIC.items())
    return f"""Evaluate this corporate email for evidence of financial wrongdoing.

EMAIL:
From: {email['from']}
To: {email['to']}
Date: {email['date']}
Subject: {email['subject']}
Body:
{email['body']}

SCORING SCALE:
0=Absent | 1=Minimal/vague | 2=Low/circumstantial | 3=Moderate/clear | 4=High/strong | 5=Smoking gun

RUBRIC:
{questions}

Return ONLY this JSON (no other text):
{{
  "scores": {{"q1":int,"q2":int,"q3":int,"q4":int,"q5":int,"q6":int,"q7":int,"q8":int,"q9":int,"q10":int}},
  "total_score": int,
  "primary_wrongdoing": "deception|coercion|self_dealing|manipulation|none",
  "rationale": "One to two sentence summary of the key evidence found."
}}"""



def build_batch_jsonl(emails: list[dict]) -> str:
    """Build a JSONL string of Vertex AI batch prediction requests."""
    lines = []
    for e in emails:
        req = {
            "request": {
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": build_prompt(e)}]}],
                "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.2},
            },
            "_email_id": e["id"],  
        }
        lines.append(json.dumps(req))
    return "\n".join(lines)


def upload_to_gcs(content: str, bucket_name: str, blob_name: str) -> str:
    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(content, content_type="application/jsonl")
    uri = f"gs://{bucket_name}/{blob_name}"
    print(f"  Uploaded {len(content.encode())/1024/1024:.1f} MB → {uri}")
    return uri


def run_batch_job(input_uri: str, output_uri_prefix: str, model: str) -> BatchPredictionJob:
    print(f"  Submitting batch job (model={model})…")
    job = BatchPredictionJob.submit(
        source_model=model,
        input_dataset=input_uri,
        output_uri_prefix=output_uri_prefix,
    )
    print(f"  Job: {job.resource_name}  state={job.state.name}")

    while not job.has_ended:
        time.sleep(30)
        job.refresh()
        print(f"  … {job.state.name}")

    if job.has_succeeded:
        print(f"  Job succeeded.")
    else:
        raise RuntimeError(f"Batch job failed: {job.state.name}\n{job.error}")
    return job


def download_batch_results(output_uri_prefix: str) -> dict[str, dict]:
    client = storage.Client(project=GCP_PROJECT)
    prefix_path = output_uri_prefix.replace(f"gs://{GCS_BUCKET}/", "")
    bucket = client.bucket(GCS_BUCKET)

    results = {}
    blobs = list(bucket.list_blobs(prefix=prefix_path))
    print(f"  Downloading {len(blobs)} output file(s)…")

    for blob in blobs:
        if not blob.name.endswith(".jsonl"):
            continue
        content = blob.download_as_text()
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                email_id = obj.get("_email_id") or obj.get("request", {}).get("_email_id")
                candidates = obj.get("response", {}).get("candidates", [])
                if not candidates:
                    continue
                parts = candidates[0].get("content", {}).get("parts", [])
                text = ""
                for part in parts:
                    if part.get("thought"):
                        continue
                    text = part.get("text", "").strip()
                    if text:
                        break
                text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.MULTILINE)
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    match = re.search(r"\{[\s\S]*\}", text)
                    parsed = json.loads(match.group(0)) if match else {}
                if email_id:
                    results[str(email_id)] = parsed
            except Exception as ex:
                print(f"  Warning: failed to parse result line: {ex}")
    print(f"  Parsed {len(results)} results.")
    return results


def merge_results(emails: list[dict], results: dict[str, dict]) -> list[dict]:
    for e in emails:
        r = results.get(str(e["id"]), {})
        if not r or "error" in r:
            e["llm_scores"], e["llm_total"], e["category"], e["rationale"] = {}, 0, "error", str(r)
        else:
            e["llm_scores"] = r.get("scores", {})
            e["llm_total"]  = r.get("total_score", 0)
            e["category"]   = r.get("primary_wrongdoing", "none")
            e["rationale"]  = r.get("rationale", "")
    return emails



def run_pipeline(emails: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]

    print(f"\n{'='*60}")
    print("  ENRON FRAUD DETECTION PIPELINE (Vertex AI Batch)")
    print(f"{'='*60}")
    print(f"  Corpus size : {len(emails)} emails  |  run_id={run_id}")

    print(f"\n[Stage 1] Date filter ({DATE_FROM} → {DATE_TO}) + keyword filter…")
    date_filtered = []
    for e in emails:
        d = e.get("date", "")[:10]  
        if DATE_FROM <= d <= DATE_TO:
            date_filtered.append(e)
    print(f"  → {len(date_filtered)}/{len(emails)} emails in date window")

    filtered = []
    for i, e in enumerate(date_filtered, 1):
        kw = keyword_score(e["subject"] + " " + e["body"])
        e["_kw"] = kw
        if kw["cats_hit"] >= 1 and kw["total"] >= 1:
            filtered.append(e)
        if i % 10000 == 0 or i == len(date_filtered):
            print(f"  … scanned {i}/{len(date_filtered)}")
    filtered.sort(key=lambda x: (x["_kw"]["cats_hit"], x["_kw"]["total"]), reverse=True)
    print(f"  → {len(filtered)}/{len(date_filtered)} emails passed keyword filter")

    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)

    print(f"\n[Stage 2] Building batch for {len(filtered)} emails (model={MODEL})…")
    jsonl = build_batch_jsonl(filtered)
    input_uri  = upload_to_gcs(jsonl, GCS_BUCKET, f"enron/{run_id}/input.jsonl")
    output_uri = f"gs://{GCS_BUCKET}/enron/{run_id}/output/"
    job = run_batch_job(input_uri, output_uri, MODEL)
    results = download_batch_results(output_uri)
    final = merge_results(filtered, results)
    final.sort(key=lambda x: x.get("llm_total", 0), reverse=True)

    rows = []
    for e in final:
        kw = e["_kw"]
        row = {
            "id":              e.get("id", ""),
            "date":            e.get("date", ""),
            "from":            e.get("from", ""),
            "subject":         e.get("subject", ""),
            "kw_total":        kw["total"],
            "kw_cats":         kw["cats_hit"],
            "kw_deception":    kw["hits"].get("deception", 0),
            "kw_coercion":     kw["hits"].get("coercion", 0),
            "kw_self_dealing": kw["hits"].get("self_dealing", 0),
            "kw_manipulation": kw["hits"].get("manipulation", 0),
            "llm_total":       e.get("llm_total", 0),
            "category":        e.get("category", ""),
            "flag":            "HIGH" if e.get("llm_total", 0) >= HIGH_RISK_THRESHOLD else
                               "SUSPICIOUS" if e.get("llm_total", 0) >= SUSPICIOUS_THRESHOLD else "NORMAL",
            "rationale":       e.get("rationale", ""),
        }
        for q in RUBRIC:
            row[q] = e.get("llm_scores", {}).get(q, "")
        rows.append(row)
    df = pd.DataFrame(rows)

    flagged = [e for e in final if e.get("llm_total", 0) >= SUSPICIOUS_THRESHOLD]

    print(f"\n{'='*60}")
    print(f"  {len(flagged)} FLAGGED EMAILS — WRONGDOING INDEX")
    print(f"{'='*60}")
    for i, e in enumerate(flagged, 1):
        print(f"\n  #{i}  [{e.get('id')}]  Wn={e.get('llm_total')}/50  [{e.get('category','').upper()}]")
        print(f"       Subject : {e['subject']}")
        print(f"       From    : {e['from']}")
        print(f"       Evidence: {e.get('rationale','')}")

    csv_path  = SAVE_PATH / "enron_scored_emails.csv"
    json_path = SAVE_PATH / "enron_flagged_emails.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        out = [{k: v for k, v in e.items() if not k.startswith("_")} for e in flagged]
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {csv_path}")
    print(f"  Saved: {json_path} ({len(flagged)} flagged emails)\n")

    return df, flagged



if __name__ == "__main__":
    print(f"Loading corpus from {CORPUS_PATH}…")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        emails = json.load(f)
    print(f"Loaded {len(emails)} emails.")

    df, flagged = run_pipeline(emails)

    print("\nTop results preview:")
    if flagged:
        preview = pd.DataFrame(flagged[:10])[["id", "subject", "llm_total", "category"]]
        print(preview.to_string(index=False))
    else:
        print("No flagged emails found.")

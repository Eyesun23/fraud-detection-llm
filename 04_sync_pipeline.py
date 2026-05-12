"""
Enron Fraud Detection Pipeline
Focus: Earnings Manipulation / Concealment of Financial Losses

Requirements:
    pip install requests pandas

Usage:
    Set GEMINI_API_KEY in your environment, then:
    python enron.py

    Optional env: ENRON_ES_HOST, ENRON_ES_INDEX, ENRON_MAX_EMAILS, ENRON_SCROLL_SIZE (batch size when using --full-corpus).
    With no maildir argument, the corpus is loaded only from Elasticsearch (no embedded fallback).

    Use --full-corpus to walk the entire index via the Scroll API (all ~251k emails); memory-heavy.
"""

import json
import re
import os
import requests
import pandas as pd



API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini-3.1-pro-preview"
HIGH_RISK_THRESHOLD = 25      
SUSPICIOUS_THRESHOLD = 12     
MAX_EMAILS_FOR_LLM = 100

DEFAULT_MAX_EMAILS_ES = 10000
ES_MAX_RESULT_WINDOW = 10000
DEFAULT_SCROLL_SIZE = 2500
DEFAULT_SCROLL_KEEPALIVE = "5m"



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
    hits = {}
    matched = {}
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


def llm_score(email: dict) -> dict:
    if not API_KEY:
        raise EnvironmentError("GEMINI_API_KEY not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": build_prompt(email)}]}],
            "generationConfig": {
                "maxOutputTokens": 8192,
                "temperature": 0.2,
            },
        },
        timeout=90,
    )
    if resp.status_code != 200:
        print(f"\n  API Error {resp.status_code}: {resp.text[:300]}")
    resp.raise_for_status()
    parts = resp.json()["candidates"][0]["content"]["parts"]
    text = ""
    for part in parts:
        if part.get("thought"):
            continue
        text = part.get("text", "").strip()
        if text:
            break
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))
        raise



def run_pipeline(emails: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    print(f"\n{'='*60}")
    print("  ENRON FRAUD DETECTION PIPELINE")
    print(f"{'='*60}")
    print(f"  Corpus size : {len(emails)} emails")

    print("\n[Stage 1] Keyword high-pass filter…")
    filtered = []
    stage1_verbose = len(emails) <= 200
    for i, e in enumerate(emails, 1):
        text = e["subject"] + " " + e["body"]
        kw = keyword_score(text)
        e["_kw"] = kw
        if kw["cats_hit"] >= 1 and kw["total"] >= 1:
            filtered.append(e)
            status = "PASS"
        else:
            status = "skip"
        if stage1_verbose:
            print(f"  {e.get('id','?'):5s} cats={kw['cats_hit']} hits={kw['total']:2d}  {status}  {e['subject'][:50]}")
        elif i % 25000 == 0 or i == len(emails):
            print(f"  … scanned {i}/{len(emails)} emails")

    filtered.sort(key=lambda x: (x["_kw"]["cats_hit"], x["_kw"]["total"]), reverse=True)
    print(f"\n  → {len(filtered)}/{len(emails)} emails passed keyword filter")

    to_score = filtered[:MAX_EMAILS_FOR_LLM]
    print(f"\n[Stage 2] LLM scoring {len(to_score)} emails…")

    results = []
    for i, e in enumerate(to_score, 1):
        print(f"  [{i}/{len(to_score)}] Scoring {e.get('id','?')} — {e['subject'][:45]}…", end=" ", flush=True)
        try:
            r = llm_score(e)
            e["llm_scores"] = r["scores"]
            e["llm_total"]  = r["total_score"]
            e["category"]   = r["primary_wrongdoing"]
            e["rationale"]  = r["rationale"]
            flag = "🔴 HIGH" if r["total_score"] >= HIGH_RISK_THRESHOLD else \
                   "🟡 SUSPICIOUS" if r["total_score"] >= SUSPICIOUS_THRESHOLD else "🟢 NORMAL"
            print(f"Wn={r['total_score']}/50  {flag}")
        except Exception as ex:
            e["llm_scores"] = {}
            e["llm_total"]  = 0
            e["category"]   = "error"
            e["rationale"]  = str(ex)
            print(f"ERROR: {ex}")
        results.append(e)

    results.sort(key=lambda x: x.get("llm_total", 0), reverse=True)

    rows = []
    for e in results:
        kw = e["_kw"]
        row = {
            "id":            e.get("id", ""),
            "date":          e.get("date", ""),
            "from":          e.get("from", ""),
            "subject":       e.get("subject", ""),
            "kw_total":      kw["total"],
            "kw_cats":       kw["cats_hit"],
            "kw_deception":  kw["hits"].get("deception", 0),
            "kw_coercion":   kw["hits"].get("coercion", 0),
            "kw_self_dealing": kw["hits"].get("self_dealing", 0),
            "kw_manipulation": kw["hits"].get("manipulation", 0),
            "llm_total":     e.get("llm_total", 0),
            "category":      e.get("category", ""),
            "flag":          "HIGH" if e.get("llm_total",0) >= HIGH_RISK_THRESHOLD else
                             "SUSPICIOUS" if e.get("llm_total",0) >= SUSPICIOUS_THRESHOLD else "NORMAL",
            "rationale":     e.get("rationale", ""),
        }
        for q in RUBRIC:
            row[q] = e.get("llm_scores", {}).get(q, "")
        rows.append(row)

    df = pd.DataFrame(rows)

    flagged = [e for e in results if e.get("llm_total", 0) >= SUSPICIOUS_THRESHOLD]

    print(f"\n{'='*60}")
    print(f"  {len(flagged)} FLAGGED EMAILS — WRONGDOING INDEX")
    print(f"{'='*60}")
    for i, e in enumerate(flagged, 1):
        print(f"\n  #{i}  [{e.get('id')}]  Wn={e.get('llm_total')}/50  [{e.get('category','').upper()}]")
        print(f"       Subject : {e['subject']}")
        print(f"       From    : {e['from']}")
        print(f"       Evidence: {e.get('rationale','')}")

    df.to_csv("enron_scored_emails.csv", index=False)
    with open("enron_flagged_emails.json", "w") as f:
        out = [{k: v for k, v in e.items() if not k.startswith("_")} for e in flagged]
        json.dump(out, f, indent=2)

    print(f"\n  Saved: enron_scored_emails.csv")
    print(f"  Saved: enron_flagged_emails.json ({len(flagged)} flagged emails)\n")

    return df, flagged




def _es_base_url() -> str:
    base = os.environ.get("ENRON_ES_HOST", "http://18.188.56.207:9200").rstrip("/")
    return base + "/"


def _resolve_max_emails_es(explicit: int | None) -> int:
    if explicit is not None:
        n = explicit
    else:
        raw = os.environ.get("ENRON_MAX_EMAILS", str(DEFAULT_MAX_EMAILS_ES))
        try:
            n = int(raw.strip())
        except ValueError:
            n = DEFAULT_MAX_EMAILS_ES
    return max(1, min(n, ES_MAX_RESULT_WINDOW))


def _resolve_scroll_batch_size() -> int:
    raw = os.environ.get("ENRON_SCROLL_SIZE", str(DEFAULT_SCROLL_SIZE))
    try:
        n = int(raw.strip())
    except ValueError:
        n = DEFAULT_SCROLL_SIZE
    return max(100, min(n, 5000))


def _es_hit_to_email(hit: dict) -> dict:
    src = hit["_source"]
    return {
        "id":      hit["_id"],
        "from":    src.get("sender", ""),
        "to":      src.get("recipients", ""),
        "date":    src.get("date", ""),
        "subject": src.get("subject", "(no subject)"),
        "body":    src.get("text", ""),
    }


def load_corpus_elasticsearch_scroll(
    query: dict | None = None,
    scroll_keepalive: str | None = None,
    batch_size: int | None = None,
) -> list[dict]:
    query = query or {"match_all": {}}
    batch_size = batch_size if batch_size is not None else _resolve_scroll_batch_size()
    scroll_keepalive = scroll_keepalive or os.environ.get("ENRON_SCROLL_KEEPALIVE", DEFAULT_SCROLL_KEEPALIVE)

    base = _es_base_url()
    index = os.environ.get("ENRON_ES_INDEX", "enron")
    search_url = f"{base}{index}/_search"
    scroll_url = f"{base}_search/scroll"
    clear_url = f"{base}_search/scroll"

    params = {"scroll": scroll_keepalive}
    body = {
        "query": query,
        "sort": ["_doc"],
        "size": batch_size,
    }

    emails: list[dict] = []
    scroll_id = None

    def _clear_scroll(sid: str | None) -> None:
        if not sid:
            return
        try:
            requests.delete(
                clear_url,
                json={"scroll_id": sid},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
        except Exception:
            pass

    try:
        print(
            f"Elasticsearch scroll at {base} (index={index}, batch={batch_size}, keepalive={scroll_keepalive})…"
        )
        r = requests.post(
            search_url,
            params=params,
            json=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=180,
        )
        r.raise_for_status()
        payload = r.json()
        scroll_id = payload.get("_scroll_id")
        total_reported = None

        while True:
            hits = payload["hits"]["hits"]
            if not hits:
                break
            for h in hits:
                emails.append(_es_hit_to_email(h))
            if total_reported is None:
                t = payload["hits"].get("total", {})
                total_reported = t.get("value", t) if isinstance(t, dict) else t
            if len(emails) <= batch_size or len(emails) % 10000 == 0:
                print(f"  … {len(emails)} documents retrieved (index total ≈ {total_reported})")

            if not scroll_id:
                break
            r = requests.post(
                scroll_url,
                json={"scroll": scroll_keepalive, "scroll_id": scroll_id},
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=180,
            )
            r.raise_for_status()
            payload = r.json()
            scroll_id = payload.get("_scroll_id", scroll_id)

        print(f"  Scroll complete: {len(emails)} documents.")
        if not emails:
            raise RuntimeError("Elasticsearch scroll returned no documents.")
        return emails

    except Exception as ex:
        if isinstance(ex, RuntimeError):
            raise
        raise RuntimeError(f"Elasticsearch scroll failed: {ex}") from ex
    finally:
        _clear_scroll(scroll_id)


def load_corpus_elasticsearch(max_emails: int | None = None) -> list[dict]:
    """
    Load emails from the Enron Elasticsearch corpus.
    Queries for fraud-related keywords to find the most relevant emails.
    Raises RuntimeError if Elasticsearch is unreachable or returns no hits.

    max_emails: page size for one _search (default from ENRON_MAX_EMAILS or DEFAULT_MAX_EMAILS_ES).
    """
    max_emails = _resolve_max_emails_es(max_emails)

    keyword_clause = {"match_all": {}}

    pr_senders = ["sarah.palmer@enron.com", "karen.denne@enron.com"]
    digest_subjects = [
        "Enron Mentions", "Press Review", "major papers only",
        "e-Journal", "IEP Clips", "IEP News", "Energy Issues",
        "Daily Power Report", "PowerMarketers", "News Headlines",
        "Newsletter", "Daily Update", "Press Clippings",
        "Media Review", "News Summary", "Daily Digest",
    ]

    query = {
        "bool": {
            "must": keyword_clause,
            "filter": {
                "range": {
                    "date": {
                        "lt": "2001-11-01"
                    }
                }
            },
            "must_not": [
                {"terms": {"sender.keyword": pr_senders}},
                *[{"match_phrase": {"subject": subj}} for subj in digest_subjects],
            ],
        }
    }

    doc = {
        "query": query,
        "from": 0,
        "size": max_emails,
        "track_total_hits": True,
    }

    base = _es_base_url()
    index = os.environ.get("ENRON_ES_INDEX", "enron")
    url = f"{base}{index}/_search"

    try:
        print(f"Connecting to Enron Elasticsearch at {base} (index={index})…")
        r = requests.post(
            url,
            json=doc,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=max(60, min(300, 30 + max_emails // 20)),
        )
        r.raise_for_status()
        payload = r.json()
        hits = payload["hits"]["hits"]
        total_available = payload["hits"]["total"]
        if isinstance(total_available, dict):
            total_available = total_available.get("value", "?")
        print(f"  Found {total_available} matching emails, retrieved {len(hits)}")

        if not hits:
            raise RuntimeError(
                "Elasticsearch returned 0 hits for the keyword query. "
                "Check ENRON_ES_INDEX and that the index contains a `text` field."
            )

        emails = [_es_hit_to_email(h) for h in hits]
        return emails

    except RuntimeError:
        raise
    except Exception as ex:
        raise RuntimeError(
            f"Failed to load corpus from Elasticsearch ({url}): {ex}"
        ) from ex


def load_corpus(
    email_dir: str | None = None,
    max_emails: int | None = None,
    full_corpus: bool = False,
) -> list[dict]:
    if full_corpus and email_dir:
        raise ValueError("full_corpus cannot be used together with a local maildir path.")

    if email_dir and os.path.isdir(email_dir):
        emails = []
        for root, _, files in os.walk(email_dir):
            for fname in files:
                path = os.path.join(root, fname)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        raw = f.read()
                    parts = raw.split("\n\n", 1)
                    headers_raw = parts[0] if len(parts) > 1 else ""
                    body = parts[1].strip() if len(parts) > 1 else raw.strip()
                    headers = {}
                    for line in headers_raw.splitlines():
                        if ": " in line:
                            k, v = line.split(": ", 1)
                            headers[k.strip().lower()] = v.strip()
                    emails.append({
                        "id":      path,
                        "from":    headers.get("from", ""),
                        "to":      headers.get("to", ""),
                        "date":    headers.get("date", ""),
                        "subject": headers.get("subject", "(no subject)"),
                        "body":    body,
                    })
                except Exception:
                    continue
        print(f"Loaded {len(emails)} emails from {email_dir}")
        return emails

    if full_corpus:
        return load_corpus_elasticsearch_scroll()
    return load_corpus_elasticsearch(max_emails=max_emails)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Enron fraud detection pipeline")
    p.add_argument(
        "maildir",
        nargs="?",
        default=None,
        help="Optional path to local maildir / .txt corpus (skips Elasticsearch)",
    )
    p.add_argument(
        "--max-emails",
        type=int,
        default=None,
        metavar="N",
        help=f"Elasticsearch keyword search: max documents in one request (default {DEFAULT_MAX_EMAILS_ES} or ENRON_MAX_EMAILS; cap {ES_MAX_RESULT_WINDOW}). Ignored with --full-corpus.",
    )
    p.add_argument(
        "--full-corpus",
        action="store_true",
        help="Elasticsearch: load every document in the index via Scroll (≈251k). High memory. Ignores --max-emails.",
    )
    p.add_argument(
        "--max-llm",
        type=int,
        default=None,
        metavar="N",
        help=f"Number of keyword-top emails to send to the LLM (default {MAX_EMAILS_FOR_LLM}).",
    )
    args = p.parse_args()
    if args.maildir and args.full_corpus:
        p.error("Cannot combine a maildir path with --full-corpus.")
    if args.max_llm is not None:
        MAX_EMAILS_FOR_LLM = max(1, args.max_llm)
    emails = load_corpus(
        args.maildir,
        max_emails=args.max_emails,
        full_corpus=args.full_corpus,
    )

    df, flagged = run_pipeline(emails)

    print(f"\nScored DataFrame preview ({len(flagged)} flagged):")
    print(df[["id", "subject", "kw_total", "llm_total", "flag", "category"]].to_string(index=False))
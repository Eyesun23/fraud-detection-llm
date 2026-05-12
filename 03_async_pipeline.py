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

import os
import requests
import httpx
import json
import re
import pandas as pd
import asyncio
from pathlib import Path




API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_FAST = "gemini-3.1-flash-lite-preview"  
MODEL_PRO  = "gemini-3.1-pro-preview"        

DATE_FROM = "2001-06-01"   
DATE_TO   = "2001-12-31"
HIGH_RISK_THRESHOLD = 25      
SUSPICIOUS_THRESHOLD = 12     
STAGE2B_TOP_N = 500            

DEFAULT_MAX_EMAILS_ES = 5000
ES_MAX_RESULT_WINDOW = 10000
DEFAULT_SCROLL_SIZE = 2500
DEFAULT_SCROLL_KEEPALIVE = "5m"

MAX_EMAILS_FOR_LLM = 10000     
CONCURRENCY_LIMIT = 5   
BURST_SIZE = 5         
STAGGER_DELAY = 1.0     
LLM_BATCH_SIZE = 20




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
    """Return per-category hit counts and matched terms."""
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
    """
    Load every matching document using the Scroll API (no 10k cap).
    Uses match_all if query is None.
    """
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

    # Build a query using our lexicon keywords to find relevant emails
    fraud_keywords = []
    for terms in LEXICONS.values():
        fraud_keywords.extend(terms[:5])  # top 5 from each category
    query_string = " OR ".join(f'"{t}"' for t in fraud_keywords if " " in t or len(t) > 3)

    if query_string.strip():
        query = {
            "query_string": {
                "default_field": "text",
                "query": query_string,
            }
        }
    else:
        query = {"match_all": {}}

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
        # POST is required for reliable _search with a body; GET+body often gets 403 from proxies.
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
    save_corpus_path: str = '.',
) -> list[dict]:
    """
    Load emails from a directory of .txt files (Enron corpus format),
    or from Elasticsearch if no directory is given.
    max_emails: single-page keyword search only (ignored when full_corpus=True).
    full_corpus: scroll the entire index (match_all), all documents.
    """
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
        emails = get_or_fetch_enron_corpus(save_path=save_corpus_path)
    else:
        emails = load_corpus_elasticsearch(max_emails=max_emails)

    filtered = [e for e in emails if DATE_FROM <= e.get("date", "")[:10] <= DATE_TO]
    print(f"Date filter ({DATE_FROM} → {DATE_TO}): {len(filtered)}/{len(emails)} emails")
    return filtered

def get_and_prepare_path(folder_name):
    full_path = Path.cwd() / folder_name
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path

async def llm_score_async(client: httpx.AsyncClient, email: dict, sem: asyncio.Semaphore, model: str = MODEL_FAST) -> dict:
    """Asynchronous call to Gemini API for a single email, with retry on 429."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": build_prompt(email)}]}],
        "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.2},
    }

    async with sem:
        backoff = 5.0
        for attempt in range(6):
            try:
                resp = await client.post(url, json=payload, timeout=90.0)
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", backoff))
                    await asyncio.sleep(retry_after)
                    backoff = min(backoff * 2, 60.0)
                    continue
                resp.raise_for_status()

                data = resp.json()
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                text = ""
                for part in parts:
                    if part.get("thought"):
                        continue
                    text = part.get("text", "").strip()
                    if text:
                        break

                text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.MULTILINE)
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    match = re.search(r"\{[\s\S]*\}", text)
                    if match:
                        return json.loads(match.group(0))
                    raise
            except Exception as e:
                if attempt == 5:
                    print(f"error:{str(e)}")
                    return {"error": str(e)}
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
        return {"error": "max retries exceeded"}


async def process_batch(emails: list[dict], limits, model: str = MODEL_FAST) -> list[dict]:

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async with httpx.AsyncClient(limits=limits, timeout=60.0) as client:
        tasks = []
        print(f"  Launching {len(emails)} requests in bursts of {BURST_SIZE} (model={model})...")

        for i in range(0, len(emails), BURST_SIZE):
            batch = emails[i: i + BURST_SIZE]
            for email in batch:
                task = asyncio.create_task(llm_score_async(client, email, semaphore, model))
                tasks.append(task)
            await asyncio.sleep(STAGGER_DELAY)
        results = await asyncio.gather(*tasks)
        return results

def key_word_filtering(emails: list[dict], save_path = '.'):
    # List to collect data for the CSV
    csv_data = []
    filtered = []
    stage1_verbose = len(emails) <= 200

    print("\n[Stage 1] Keyword high-pass filter…")
    for i, e in enumerate(emails, 1):
        text = e["subject"] + " " + e["body"]
        kw = keyword_score(text)
        e["_kw"] = kw

        if kw["cats_hit"] >= 1 and kw["total"] >= 1:
            filtered.append(e)
            status = "PASS"
        else:
            status = "skip"

        csv_data.append({
            "id": e.get("id", "?"),
            "total_hits": kw["total"],
            "cats_hit": kw["cats_hit"],
             **kw["hits"],
            "status": status,
            "status_numeric": 1 if status == "PASS" else 0,
            "subject_snippet": e.get("subject", "")[:50]
        })

        if stage1_verbose:
            print(
                f"  {e.get('id', '?'):5s} cats={kw['cats_hit']} hits={kw['total']:2d}  {status}  {e.get('subject', '')[:50]}")
        elif i % 25000 == 0 or i == len(emails):
            print(f"  … scanned {i}/{len(emails)} emails")

    df_results = pd.DataFrame(csv_data)
    df_results.to_csv(os.path.join(save_path, "full_keyword_filter_stat.csv"), index=False)

    print(f"\n[Data Export] Saved {len(df_results)} rows to 'keyword_filter.csv'")
    print("\n--- DataFrame Summary Stats ---")
    print(df_results.describe())
    print("-" * 30)
    df_results.describe().to_csv(os.path.join(save_path, "summary_stat_keyword_filters.csv"), index=False)


    filtered.sort(key=lambda x: (x["_kw"]["cats_hit"], x["_kw"]["total"]), reverse=True)
    print(f"\n  → {len(filtered)}/{len(emails)} emails passed keyword filter")

    return df_results, filtered

async def run_pipeline_async(emails: list[dict], save_path ="") -> tuple[pd.DataFrame, list[dict]]:
    """
    Full two-stage pipeline.
    Returns (scored_df, top5_emails).
    """
    print(f"\n{'=' * 60}")
    print("  ENRON FRAUD DETECTION PIPELINE")
    print(f"{'=' * 60}")
    print(f"  Corpus size : {len(emails)} emails")

    _, filtered = key_word_filtering(emails, save_path)

    limits = httpx.Limits(max_connections=CONCURRENCY_LIMIT, max_keepalive_connections=50)

    async def score_emails(emails, model):
        scored = []
        total = len(emails)
        for i in range(0, total, LLM_BATCH_SIZE):
            batch = emails[i: i + LLM_BATCH_SIZE]
            print(f"  → Batch {i // LLM_BATCH_SIZE + 1}/{(total + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE}: emails {i}–{min(i + LLM_BATCH_SIZE, total)}")
            responses = await process_batch(batch, limits, model)
            for e, r in zip(batch, responses):
                if "error" in r:
                    e["llm_scores"], e["llm_total"], e["category"], e["rationale"] = {}, 0, "error", r["error"]
                else:
                    e["llm_scores"] = r.get("scores", {})
                    e["llm_total"]  = r.get("total_score", 0)
                    e["category"]   = r.get("primary_wrongdoing", "none")
                    e["rationale"]  = r.get("rationale", "")
                scored.append(e)
        return scored

    print(f"\n[Stage 2a] Flash-lite screening {len(filtered)} emails…")
    results_2a = await score_emails(filtered, MODEL_FAST)
    results_2a.sort(key=lambda x: x.get("llm_total", 0), reverse=True)

    top_candidates = results_2a[:STAGE2B_TOP_N]
    print(f"\n[Stage 2b] Pro deep-scoring top {len(top_candidates)} candidates…")
    results_2b = await score_emails(top_candidates, MODEL_PRO)
    results_2b.sort(key=lambda x: x.get("llm_total", 0), reverse=True)

    pro_ids = {e["id"] for e in results_2b}
    results = results_2b + [e for e in results_2a if e["id"] not in pro_ids]
    results.sort(key=lambda x: x.get("llm_total", 0), reverse=True)

    rows = []
    for e in results:
        kw = e["_kw"]
        row = {
            "id": e.get("id", ""),
            "date": e.get("date", ""),
            "from": e.get("from", ""),
            "subject": e.get("subject", ""),
            "kw_total": kw["total"],
            "kw_cats": kw["cats_hit"],
            "kw_deception": kw["hits"].get("deception", 0),
            "kw_coercion": kw["hits"].get("coercion", 0),
            "kw_self_dealing": kw["hits"].get("self_dealing", 0),
            "kw_manipulation": kw["hits"].get("manipulation", 0),
            "llm_total": e.get("llm_total", 0),
            "category": e.get("category", ""),
            "flag": "HIGH" if e.get("llm_total", 0) >= HIGH_RISK_THRESHOLD else
            "SUSPICIOUS" if e.get("llm_total", 0) >= SUSPICIOUS_THRESHOLD else "NORMAL",
            "rationale": e.get("rationale", ""),
        }
        for q in RUBRIC:
            row[q] = e.get("llm_scores", {}).get(q, "")
        rows.append(row)

    df = pd.DataFrame(rows)

    top5 = []
    try:
        top5 = sorted(
            [e for e in results if e.get("llm_total", 0) >= SUSPICIOUS_THRESHOLD],
            key=lambda x: x.get("llm_total", 0),
            reverse=True
        )[:5]
    except TypeError as error:
        print(f"Sorting failed: {error}")


    print(f"\n{'=' * 60}")
    print(f"  TOP {len(top5)} EMAILS — WRONGDOING INDEX")
    print(f"{'=' * 60}")
    for i, e in enumerate(top5, 1):
        print(f"\n  #{i}  [{e.get('id')}]  Wn={e.get('llm_total')}/50  [{e.get('category', '').upper()}]")
        print(f"       Subject : {e['subject']}")
        print(f"       From    : {e['from']}")
        print(f"       Evidence: {e.get('rationale', '')}")

    # ── Save outputs ──
    df.to_csv(os.path.join(save_path, "enron_scored_emails.csv"), index=False)
    with open(os.path.join(save_path, "enron_top5.json"), "w") as f:
        out = [{k: v for k, v in e.items() if not k.startswith("_")} for e in top5]
        json.dump(out, f, indent=2)

    print(f"\n  Saved: enron_scored_emails.csv")
    print(f"  Saved: enron_top5.json\n")

    return df, top5

def get_or_fetch_enron_corpus(save_path: str) -> list[dict]:
    """
    Checks if the Enron corpus JSON exists locally.
    If not, fetches it from Elasticsearch and saves it.
    """
    file_name = "emros_curpose.json"  
    full_path = os.path.join(save_path, file_name)

    if os.path.exists(full_path):
        print(f"Loading existing corpus from: {full_path}")
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Corpus not found locally. Starting Elasticsearch scroll...")
    emails = load_corpus_elasticsearch_scroll()

    print(f"Saving {len(emails)} emails to {full_path}...")

    os.makedirs(save_path, exist_ok=True)

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(emails, f, indent=2)

    return emails



if __name__ == "__main__":

    SAVE_PATH = get_and_prepare_path("data")
    print(f"saving to {SAVE_PATH}")

    CONCURRENCY_LIMIT = 5
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    emails = load_corpus(
        "",
        max_emails=None,
        full_corpus=True,
        save_corpus_path=SAVE_PATH,
    )


    df, top5 = asyncio.run(run_pipeline_async(emails, save_path=SAVE_PATH))
    top5 = pd.DataFrame(top5)

    print("Scored DataFrame preview:")
    if len(top5) > 0:
        print(top5[["id", "subject",  "llm_total",  "category"]].to_string(index=False))
    else:
        print("Nothing Detected.")
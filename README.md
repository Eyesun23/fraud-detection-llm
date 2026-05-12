# Enron Fraud Detection Pipeline

This project implements an automated pipeline to detect financial wrongdoing, earnings manipulation, and coercion within the Enron email corpus. It uses Google's Gemini LLMs (via the standard API and Vertex AI Batch Prediction) to evaluate corporate communications against a rigorous forensic investigator rubric.

## Methodology: From Elasticsearch to Fraud Detection

This pipeline uses a multi-stage approach to securely process and evaluate hundreds of thousands of Enron emails. Here is the step-by-step process of how the data is acquired and analyzed:

### 1. Acquiring the Corpus from Elasticsearch
The raw Enron dataset is initially hosted in an Elasticsearch database. To get the data without overwhelming the server or missing records, `01_fetch_and_batch.py` uses the **Elasticsearch Scroll API**:
- **Scroll Batches**: The script connects to the `enron` Elasticsearch index and requests documents in scrolling batches (typically 2,500 to 5,000 emails per batch).
- **Match All**: It bypasses the standard 10,000 document search limit by keeping a scrolling cursor open, downloading every single email in the index.
- **Local Storage**: Once all emails are retrieved, the script saves the entire collection locally as a single JSON file (`data/emros_curpose.json`) for efficient downstream processing.

### 2. High-Pass Keyword Filtering
Before sending data to the LLMs, the pipeline performs a fast, local high-pass filter:
- Emails are checked against curated lexicons representing four categories: **deception**, **coercion**, **self-dealing**, and **manipulation**.
- Only emails containing at least one relevant keyword (e.g., "off-balance", "hide losses", "rank and yank") are passed to the next stage, significantly reducing API costs.

### 3. LLM Forensic Evaluation
The surviving emails are evaluated by Google's Gemini models using a 10-question forensic rubric:
- **Vertex AI Batch (Preferred)**: The emails are bundled into a JSONL format, uploaded to a Google Cloud Storage bucket, and evaluated in bulk by `gemini-2.5-flash` using a Vertex AI Batch Prediction job. 
- **Dual-Model Async Pipeline**: Alternatively, a two-stage asynchronous pipeline first uses `gemini-3.1-flash-lite` for bulk scoring, followed by a deeper dive using `gemini-3.1-pro` on the most suspicious candidates.

---

## Project Structure

The project is broken down into a multi-stage pipeline. The scripts are numbered in the order they should typically be executed:

*   **`01_fetch_and_batch.py`**: 
    Connects to an Elasticsearch index using the Scroll API, downloads the entire Enron corpus, performs initial keyword filtering, and saves the resulting dataset locally as `data/emros_curpose.json`.
*   **`02_vertex_ai_pipeline.py`**: 
    Reads the downloaded corpus and orchestrates a bulk evaluation using Google Cloud's **Vertex AI Batch Prediction**. It uploads the dataset to Google Cloud Storage (GCS), triggers the batch job, polls for completion, and downloads the fraud scores.
*   **`03_async_pipeline.py`**: 
    An alternative, high-performance local pipeline. It uses Python's `asyncio` and `httpx` to perform a two-stage evaluation: first screening the bulk of emails with `gemini-3.1-flash-lite`, and then doing a deep forensic re-evaluation of the top suspicious candidates using `gemini-3.1-pro`.
*   **`04_sync_pipeline.py`**: 
    The original, synchronous pipeline script. It processes emails sequentially using the `requests` library and `gemini-3.1-pro`.
*   **`exploratory_analysis.ipynb`**: 
    A Jupyter Notebook for Exploratory Data Analysis (EDA). Contains code to visualize the results, chart the most suspicious timeframes (e.g., late 2001), and analyze the dominant categories of fraud detected by the LLM.

## Prerequisites

To run these pipelines, you will need the following dependencies:
```bash
pip install requests httpx pandas google-cloud-aiplatform google-cloud-storage google-genai
```

### Environment Variables
Depending on which pipeline you run, you must export the following credentials:
*   `GEMINI_API_KEY`: Required for `03_async_pipeline.py` and `04_sync_pipeline.py`.
*   *Google Cloud Auth*: You must be authenticated with Google Cloud (e.g., via `gcloud auth application-default login`) and have your `GCP_PROJECT`, `GCP_LOCATION`, and `GCS_BUCKET` environment variables configured to use `02_vertex_ai_pipeline.py`.
*   *Elasticsearch Config*: (Optional) You can customize the ES connection in `01_fetch_and_batch.py` via `ENRON_ES_HOST` and `ENRON_ES_INDEX`.

## How to Run

If you are starting from scratch without the dataset, run the pipeline in this order:

1.  **Download the data**:
    ```bash
    python 01_fetch_and_batch.py
    ```
    *(This connects to Elasticsearch via the Scroll API and saves `data/emros_curpose.json`)*

2.  **Score the emails**:
    To score the emails at scale using Vertex AI, run:
    ```bash
    python 02_vertex_ai_pipeline.py
    ```
    Or, to score them locally using the async dual-model approach, run:
    ```bash
    python 03_async_pipeline.py
    ```

3.  **Analyze the results**:
    Open `exploratory_analysis.ipynb` in Jupyter or VS Code to visualize the generated CSV/JSON results.

import pandas as pd
import pathlib
import numpy as np
import csv
import time
from google import genai
from config_new import GEMINI_API_KEY, EMBED_MODEL, PROC_EMBED_DIR, BASE_DIR, RAW_SEC_DIR

client = genai.Client(api_key=GEMINI_API_KEY)
SUMMARY_DIR = BASE_DIR / "data/processed/sec_summaries"

def embed_text(text):
    for attempt in range(3):
        try:
            emb = client.models.embed_content(model=EMBED_MODEL, contents=text)
            return np.array(emb.embeddings[0].values, dtype=np.float32)
        except Exception as e:
            print(f"Embedding attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return None

def main():
    processed_keys = set()
    PROC_EMBED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PROC_EMBED_DIR / "manifest.csv"
    
    if manifest_path.exists():
        m_df = pd.read_csv(manifest_path, names=['ticker','date','path','dim'])
        processed_keys.update(set(zip(m_df['ticker'], m_df['date'].astype(str))))

    if not SUMMARY_DIR.exists():
        print("Summary directory not found. Run 02b_summarize_sec.py first.")
        return

    succeeded = 0
    failed = 0

    for summary_path in SUMMARY_DIR.glob("*.txt"):
        # Format: TICKER_ACCESSION.txt
        parts = summary_path.stem.split("_")
        ticker = parts[0]
        accession = parts[1]
        
        # We need the filing date from the raw files (logic preserved from original)
        file_date = None
        raw_filing_dir = RAW_SEC_DIR / "sec-edgar-filings" / ticker / "10-Q" / accession
        full_sub = raw_filing_dir / "full-submission.txt"
        
        if full_sub.exists():
            with open(full_sub, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if "FILED AS OF DATE:" in line:
                        raw_date = line.split(":")[1].strip()
                        if len(raw_date) == 8:
                            file_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
                        break
        
        if not file_date:
            print(f"Could not find date for {summary_path.name}")
            continue
            
        if (ticker, str(file_date)) in processed_keys:
            continue
            
        print(f"Embedding summary for {ticker} on {file_date}...")
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_text = f.read()
            
        embedding = embed_text(summary_text)
        if embedding is not None:
            out = PROC_EMBED_DIR / f"{ticker}_{accession}.npy"
            np.save(out, embedding)
            with open(manifest_path, 'a', newline='') as f:
                csv.writer(f).writerow([ticker, file_date, str(out), embedding.shape[0]])
            processed_keys.add((ticker, str(file_date)))
            succeeded += 1
            print(f"Saved embedding to {out.name}")
        else:
            failed += 1
            print(f"FAILED embedding for {ticker} {file_date}")

    print(f"\nDone: {succeeded} succeeded, {failed} failed")
    if failed > 0:
        raise RuntimeError(f"{failed} embeddings failed — check EMBED_MODEL name")

if __name__ == "__main__":
    main()

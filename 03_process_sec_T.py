import pandas as pd
import pathlib
import numpy as np
import csv
import time
import json
import bs4
from google import genai
from config import GEMINI_API_KEY, GEN_MODEL, EMBED_MODEL, RAW_SEC_DIR, PROC_EMBED_DIR

client = genai.Client(api_key=GEMINI_API_KEY)

def parse_html_to_text(html_content):
    soup = bs4.BeautifulSoup(html_content, 'lxml')
    for tag in soup.find_all(['script', 'style']): tag.decompose()
    return soup.get_text(separator=" ", strip=True)[:100000]

def analyze_and_embed(text, ticker):
    for attempt in range(3):
        try:
            res = client.models.generate_content(model=GEN_MODEL, contents=f"Analyze 10-Q momentum factors for {ticker}:\n{text}")
            emb = client.models.embed_content(model=EMBED_MODEL, contents=res.text)
            return np.array(emb.embeddings[0].values, dtype=np.float32)
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None

def main():
    processed_keys = set()
    PROC_EMBED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PROC_EMBED_DIR / "manifest.csv"
    
    if manifest_path.exists():
        m_df = pd.read_csv(manifest_path, names=['ticker','date','path','dim'])
        processed_keys.update(set(zip(m_df['ticker'], m_df['date'].astype(str))))

    sec_base = RAW_SEC_DIR / "sec-edgar-filings"
    if not sec_base.exists(): return

    for ticker_dir in sec_base.iterdir():
        if not ticker_dir.is_dir(): continue
        ticker = ticker_dir.name
        type_dir = ticker_dir / "10-Q"
        if not type_dir.exists(): continue
        
        for filing_dir in type_dir.iterdir():
            if not filing_dir.is_dir(): continue
            
            file_date = None
            
            target_file = next(filing_dir.glob("*.htm"), next(filing_dir.glob("full-submission.txt"), None))
            if not target_file: continue
            
            # Extract date from full-submission.txt
            full_sub = filing_dir / "full-submission.txt"
            if full_sub.exists():
                with open(full_sub, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if "FILED AS OF DATE:" in line:
                            raw_date = line.split(":")[1].strip()
                            if len(raw_date) == 8:
                                file_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
                            break
                            
            if not file_date: continue
            
            if (ticker, str(file_date)) in processed_keys: 
                print(f"Skipping already processed filing for {ticker} on {file_date}")
                continue
            
            print(f"Processing filing for {ticker} on {file_date}...")
            with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
                text = parse_html_to_text(f.read()) if target_file.suffix == '.htm' else f.read()[:100000]
                
            embedding = analyze_and_embed(text, ticker)
            if embedding is not None:
                out = PROC_EMBED_DIR / f"{ticker}_{filing_dir.name}.npy"
                np.save(out, embedding)
                with open(manifest_path, 'a', newline='') as f:
                    csv.writer(f).writerow([ticker, file_date, str(out), embedding.shape[0]])
                processed_keys.add((ticker, str(file_date)))

if __name__ == "__main__": main()

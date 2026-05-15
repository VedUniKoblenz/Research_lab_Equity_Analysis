import pandas as pd
import pathlib
import time
import bs4
from google import genai
from config_new import GEMINI_API_KEY, GEN_MODEL, RAW_SEC_DIR, BASE_DIR

client = genai.Client(api_key=GEMINI_API_KEY)
SUMMARY_DIR = BASE_DIR / "data/processed/sec_summaries"

def parse_html_to_text(html_content):
    soup = bs4.BeautifulSoup(html_content, 'lxml')
    for tag in soup.find_all(['script', 'style']): tag.decompose()
    return soup.get_text(separator=" ", strip=True)[:100000] 

def summarize_filing(text, ticker):
    prompt = (
    f"You are a Senior Equity Research Analyst specializing in quantitative alpha extraction for {ticker}. "
    "Your goal is to extract non-obvious, price-sensitive signals from this SEC Form 10-Q filing. "
    "Ignore all standard legal boilerplate. Focus exclusively on:\n\n"

    "1. Sentiment Divergence\n"
    "- Identify areas where management's tone (bullish/bearish) contradicts the raw financial data.\n"
    "- Flag over-optimism despite weakening fundamentals.\n\n"

    "2. Hidden Operational Shifts\n"
    "- Inventory levels (days outstanding, write-downs)\n"
    "- Supply chain dependencies or concentration risks\n"
    "- R&D spending pivot points (new vs. deprecated projects)\n"
    "- Headcount changes or restructuring signals\n"
    "- Accounts receivable/payable trends\n\n"

    "3. Forward-Looking Nuance\n"
    "- Extract specific guidance details\n"
    "- 'We expect' (high confidence) vs 'We anticipate' (moderate) vs 'We believe' (speculative)\n"
    "- Numerical ranges vs directional statements\n"
    "- Any withdrawn or revised prior guidance\n\n"

    "4. Atypical Risk Factors\n"
    "- NEW or materially MODIFIED risks only\n"
    "- Competitive landscape shifts\n"
    "- Regulatory exposure escalation\n"
    "- Macroeconomic vulnerability\n"
    "- Technology obsolescence threats\n\n"

    "5. Segment-Specific KPI Trajectory\n"
    "- Extract 1–2 most predictive metrics per segment:\n"
    "  • Tech/Software: ARR, RPO, NRR, DBNRR\n"
    "  • Retail: Same-store sales, GMV, AOV\n"
    "  • Hardware/Auto: Deliveries, ASP, backlog\n"
    "  • Subscriptions: Net adds, churn, ARPU\n"
    "  • Financials: NIM, efficiency ratio, loan growth\n\n"

    "6. Capital Allocation Signals\n"
    "- Share buybacks (pace vs prior quarters)\n"
    "- Dividend changes\n"
    "- M&A activity\n"
    "- Debt issuance or refinancing\n\n"

    "Priority Rules:\n"
    "- Signal strength = Novelty × Materiality × Surprise\n"
    "- Price-sensitive info > backward-looking info\n"
    "- Specific numbers > vague statements\n"
    "- Current quarter changes > YoY only\n\n"

    "Output Format:\n"
    "- Dense, technical summary (500–1500 characters)\n"
    "- Bullet points only\n"
    "- Include exact figures wherever possible\n"
    "- No explanations, no fluff\n\n"

    f"Ticker: {ticker}\n\n"
    f"Filing Content (truncated):\n{text}"
)
    for attempt in range(3): 
        try:
            res = client.models.generate_content(model=GEN_MODEL, contents=prompt)
            return res.text
        except Exception as e:
            if "429" in str(e):
                print(f"Rate limited. Waiting 15s...")
                time.sleep(15)
            else:
                print(f"Error on attempt {attempt+1}: {e}")
                time.sleep(5)
    return None

def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    sec_base = RAW_SEC_DIR / "sec-edgar-filings"
    
    if not sec_base.exists(): return

    all_filing_dirs = []
    for ticker_dir in sec_base.iterdir():
        if not ticker_dir.is_dir(): continue
        ticker = ticker_dir.name
        type_dir = ticker_dir / "10-Q"
        if not type_dir.exists(): continue
        for filing_dir in type_dir.iterdir():
            if filing_dir.is_dir():
                all_filing_dirs.append((ticker, filing_dir))

    succeeded = 0
    failed = 0
    print(f"Starting high-quality summarization of {len(all_filing_dirs)} filings...")
    for ticker, filing_dir in all_filing_dirs:
        summary_file = SUMMARY_DIR / f"{ticker}_{filing_dir.name}.txt"
        if summary_file.exists(): continue

        target_file = next(filing_dir.glob("*.htm"), next(filing_dir.glob("full-submission.txt"), None))
        if not target_file: continue
        
        print(f"Processing {ticker} {filing_dir.name}...")
        with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = parse_html_to_text(f.read()) if target_file.suffix == '.htm' else f.read()[:100000]
            
        summary = summarize_filing(text, ticker)
        if summary:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            succeeded += 1
            time.sleep(5)
        else:
            failed += 1
            print(f"Failed to process {ticker} {filing_dir.name}")

    print(f"\nDone: {succeeded} succeeded, {failed} failed")
    if failed > 0:
        raise RuntimeError(f"{failed} summaries failed — check GEN_MODEL name")

if __name__ == "__main__":
    main()

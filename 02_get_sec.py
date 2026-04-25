from sec_edgar_downloader import Downloader
from config import RAW_SEC_DIR, TICKERS

def main():
    print(f"Downloading 10-Q filings for {TICKERS}...")
    RAW_SEC_DIR.mkdir(parents=True, exist_ok=True)
    dl = Downloader("MyResearchOrg", "research@example.com", RAW_SEC_DIR)
    for t in TICKERS:
        print(f"Fetching 10-Q for {t}...")
        dl.get("10-Q", t, after="2022-01-01", download_details=True)

if __name__ == "__main__": main()

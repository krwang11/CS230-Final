import os
import re
import time
import json
import argparse
from typing import Optional, Iterable, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# -------- CONFIG --------
GOVINFO_API_KEY = os.getenv("GOVINFO_API_KEY")
CONGRESS_API_KEY = os.getenv("CONGRESS_API_KEY")

if not GOVINFO_API_KEY:
    raise RuntimeError("GOVINFO_API_KEY not set. Get one from https://api.data.gov/signup/")
if not CONGRESS_API_KEY:
    raise RuntimeError("CONGRESS_API_KEY not set. Get one from https://api.data.gov/signup/")

GOVINFO_BASE = "https://api.govinfo.gov/collections/BILLS"
CONGRESS_API_BASE = "https://api.congress.gov/v3"
MIN_SECONDS_BETWEEN_REQUESTS = 0.02 
MAX_WORKERS = 10

# new bill-key format requirement: "118-hres-643"
BILLS_PACKAGE_RE = re.compile(r"^BILLS-(\d+)([A-Za-z]+)(\d+)([A-Za-z]*)$", re.IGNORECASE)

CONGRESSIONAL_CONTROL = {
    118: ("R", "D", "D"),
    117: ("D", "D", "D"),
    116: ("D", "R", "R"),
    115: ("R", "R", "R"),
    114: ("R", "R", "D"),
    113: ("R", "D", "D"),
    112: ("R", "D", "D"),
    111: ("D", "D", "D"),
    110: ("D", "D", "R"),
    109: ("R", "R", "R"),
    108: ("R", "R", "R"),
}

_last_request_time = 0.0

def rate_limit():
    """Simple global rate limiter to space out requests across threads."""
    global _last_request_time
    if MIN_SECONDS_BETWEEN_REQUESTS <= 0:
        return
    now = time.time()
    delta = now - _last_request_time
    if delta < MIN_SECONDS_BETWEEN_REQUESTS:
        time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - delta)
    _last_request_time = time.time()

def make_session_with_retries(total_retries: int = 3, backoff: float = 1.0) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET", "POST"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def parse_package_id(package_id: str) -> Optional[Dict]:
    """
    Parse GovInfo package ids like:
      BILLS-118hres643ih
      BILLS-117hr3076ih

    Returns:
      {"congress": 118, "bill_type": "hres", "bill_number": "643"}
    or None on failure.
    """
    if not package_id:
        return None
    m = BILLS_PACKAGE_RE.match(package_id.strip())
    if not m:
        return None
    congress_s, bill_type_raw, bill_number_s, _version = m.groups()
    try:
        congress = int(congress_s)
    except ValueError:
        return None
    bill_type = bill_type_raw.lower()
    bill_number = "".join(ch for ch in bill_number_s if ch.isdigit())
    if not bill_number:
        return None
    return {"congress": congress, "bill_type": bill_type, "bill_number": bill_number}

def make_bill_key(parsed: Dict) -> str:
    """Return a key like: 118-hres-643"""
    return f"{parsed['congress']}-{parsed['bill_type']}-{parsed['bill_number']}"

def congress_start_end(congress: int):
    start_year = 1789 + 2 * (congress - 1)
    start = f"{start_year}-01-03T00:00:00Z"
    end_year = start_year + 2
    end = f"{end_year}-01-03T23:59:59Z"
    return start, end

def iter_bills_packages_for_congress(
    session: requests.Session,
    congress: int,
    start_date: str,
    end_date: str,
) -> Iterable[str]:
    """
    Yield all BILLS package IDs from GovInfo for a given congress and date range.
    This yields raw package ids (versions included). Deduplication into unique bills
    is handled later.
    """
    url = f"{GOVINFO_BASE}/{start_date}"
    params = {
        "pageSize": 1000,
        "offsetMark": "*",
        "dateIssuedEndDate": end_date,
        "congress": str(congress),
        "api_key": GOVINFO_API_KEY,
    }

    seen_ids = set()

    while True:
        rate_limit()
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        packages = data.get("packages", [])
        if not packages:
            break

        for pkg in packages:
            pid = pkg.get("packageId")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            yield pid

        next_page = data.get("nextPage")
        if not next_page:
            break

        url = next_page
        if "api_key=" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}api_key={GOVINFO_API_KEY}"
        params = {}

# -------- Minimal metadata fetch + record creation --------
def fetch_minimal_bill_record(session: requests.Session, parsed: Dict) -> Optional[Dict]:
    """
    Given a parsed package id (congress, bill_type, bill_number),
    fetch /bill/{congress}/{bill_type}/{bill_number} and extract minimal fields.
    """
    congress = parsed["congress"]
    bill_type = parsed["bill_type"]
    bill_number = parsed["bill_number"]

    url = f"{CONGRESS_API_BASE}/bill/{congress}/{bill_type}/{bill_number}"
    params = {"api_key": CONGRESS_API_KEY, "format": "json"}

    try:
        rate_limit()
        resp = session.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            # Non-200: return None to mark as failed
            return None
        bill_json = resp.json().get("bill", {})

        # Extract fields requested by user
        origin_chamber = bill_json.get("originChamber")
        policy_area = (bill_json.get("policyArea") or {}).get("name")
        sponsor = (bill_json.get("sponsors") or [{}])[0]
        sponsor_party = sponsor.get("party")
        num_cosponsors = int((bill_json.get("cosponsors") or {}).get("count") or 0)
        num_actions = int((bill_json.get("actions") or {}).get("count") or 0)
        latest_action_text = (bill_json.get("latestAction") or {}).get("text")
        num_amendments = int((bill_json.get("amendments") or {}).get("count") or 0)

        # Congressional control static info
        house_majority, senate_majority, president_party = CONGRESSIONAL_CONTROL.get(congress, (None, None, None))

        record = {
            "bill_key": make_bill_key(parsed),
            "bill_number": bill_number,
            "origin_chamber": origin_chamber,
            "policy_area": policy_area,
            "sponsor_party": sponsor_party,
            "num_cosponsors": num_cosponsors,
            "num_actions": num_actions,
            "latest_action": latest_action_text,
            "num_amendments": num_amendments,
            "house_majority": house_majority,
            "senate_majority": senate_majority,
            "president_party": president_party,
        }
        return record

    except Exception:
        return None

def process_bill_record_minimal(session: requests.Session, package_id: str) -> Optional[Dict]:
    """
    Process a single govinfo package id: parse, fetch minimal metadata, return record dict.
    Returns None on parse/fetch failure.
    """
    if not package_id:
        return None
    parsed = parse_package_id(package_id)
    if not parsed:
        return None
    return fetch_minimal_bill_record(session, parsed)

def scrape_congress(congress: int, output_jsonl: str, limit: int = 0, resume: bool = True, max_workers: int = MAX_WORKERS):
    start_date, end_date = congress_start_end(congress)
    print(f"Scraping Congress {congress}: {start_date} → {end_date}")
    print(f"Output file: {output_jsonl}")
    print(f"Concurrent workers: {max_workers}")

    processed_keys = set()
    if resume and os.path.exists(output_jsonl):
        print(f"Loading already processed bill keys from {output_jsonl} ...")
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    k = rec.get("bill_key")
                    if k:
                        processed_keys.add(k)
                except Exception:
                    pass
        print(f"Found {len(processed_keys)} already-processed bill keys\n")

    with make_session_with_retries() as session:

        print(f"Fetching package ids from GovInfo...")
        package_ids = list(iter_bills_packages_for_congress(session, congress, start_date, end_date))
        print(f"Found {len(package_ids)} raw packages from GovInfo\n")

        seen_bill_keys = set()
        unique_package_ids = []
        for pid in package_ids:
            parsed = parse_package_id(pid)
            if not parsed:
                continue
            bill_key = make_bill_key(parsed)
            if bill_key in seen_bill_keys:
                continue
            seen_bill_keys.add(bill_key)
            if bill_key in processed_keys:
                continue
            unique_package_ids.append(pid)

        total_unique = len(seen_bill_keys)
        remaining = len(unique_package_ids)
        print(f"Unique bills (after dedup): {total_unique}")
        print(f"Already processed: {len(processed_keys)}")
        print(f"Remaining to process: {remaining}\n")

        if remaining == 0:
            print("Nothing left to do for this Congress.")
            return

        if limit and limit > 0:
            unique_package_ids = unique_package_ids[:limit]
            print(f"Limiting to first {len(unique_package_ids)} bills for testing\n")

        n_written = 0
        n_failed = 0
        mode = "a" if (resume and os.path.exists(output_jsonl)) else "w"
        with open(output_jsonl, mode, encoding="utf-8") as out_f:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_bill_record_minimal, session, pid): pid for pid in unique_package_ids}

                # Iterate as futures complete for progress
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Congress {congress}"):
                    pid = futures[future]
                    try:
                        rec = future.result()
                        if not rec:
                            n_failed += 1
                            continue
                        out_f.write(json.dumps(rec) + "\n")
                        out_f.flush()
                        n_written += 1
                    except Exception as e:
                        n_failed += 1
                        print(f"\n[ERROR] Failed to process {pid}: {e}")

    print(f"Wrote {n_written} new bills to {output_jsonl}")
    print(f"Failed: {n_failed}")
    total_in_file = len(processed_keys) + n_written
    print(f"Total bills recorded (approx): {total_in_file}")

def main():
    p = argparse.ArgumentParser(description="Scrape minimal bill metadata (bill_key format: congress-billtype-billnum)")
    p.add_argument("--congress", type=int, nargs="+", required=True, help="Congress number(s), e.g. 117")
    p.add_argument("--output-dir", type=str, default=".", help="Output directory")
    p.add_argument("--limit", type=int, default=0, help="Limit number of bills processed (0 = all)")
    p.add_argument("--no-resume", action="store_true", help="Start fresh (don't resume)")
    p.add_argument("--workers", type=int, default=MAX_WORKERS, help="Number of concurrent workers")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume

    for congress in args.congress:
        output_file = output_dir / f"{congress}_bills_metadata.jsonl"
        try:
            scrape_congress(congress, str(output_file), limit=args.limit, resume=resume, max_workers=args.workers)
        except KeyboardInterrupt:
            print(f"\nStopping scrape for Congress {congress}. Progress saved to {output_file}")
            break
        except Exception as e:
            print(f"\nFailed to scrape Congress {congress}: {e}")
            continue

if __name__ == "__main__":
    main()

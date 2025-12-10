"""Full bill text and bill IDs from API

Original file is located at
    https://colab.research.google.com/drive/1AApxR732h0qqIqENGvEKNxKrJ6QhMdSH
"""
import os
os.environ["GOVINFO_API_KEY"] = " "

import os
import re
import time
import json
from typing import Dict, Iterable, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


GOVINFO_API_KEY = os.getenv("GOVINFO_API_KEY")
if not GOVINFO_API_KEY:
    raise RuntimeError(
        "Please set GOVINFO_API_KEY environment variable "
        "with your api.data.gov key for api.govinfo.gov."
    )

COLLECTION = "BILLS"

COLLECTIONS_BASE = f"https://api.govinfo.gov/collections/{COLLECTION}"
PACKAGES_BASE = "https://api.govinfo.gov/packages"

MAX_WORKERS = 8
MIN_SECONDS_BETWEEN_REQUESTS = 0.0

# Range of sessions
START_CONGRESS = 118
END_CONGRESS = 108

BILLS_PACKAGE_RE = re.compile(r"^BILLS-(\d+)([a-z]+)(\d+)([a-z]+)$")

_last_request_time = 0.0

def congress_start_end(congress: int) -> Tuple[str, str]:
    """
    Compute start and end ISO datetimes for a given Congress.
    1st Congress: 1789–1791, 2nd: 1791–1793, etc.
    We'll use Jan 3 of the start year to Jan 3 two years later.
    """
    start_year = 1789 + 2 * (congress - 1)
    start = f"{start_year}-01-03T00:00:00Z"
    end_year = start_year + 2
    end = f"{end_year}-01-03T23:59:59Z"
    return start, end


def parse_package_id(package_id: str) -> Optional[Dict]:
    """
    Parse a BILLS packageId like 'BILLS-108hr5804ih' into parts.
    """
    m = BILLS_PACKAGE_RE.match(package_id)
    if not m:
        return None
    congress_str, bill_type, bill_number_str, version_code = m.groups()
    return {
        "congress": int(congress_str),
        "bill_type": bill_type,         # e.g., 'hr', 's', 'hjres', ...
        "bill_number": bill_number_str,
        "version_code": version_code,   # e.g., 'ih', 'is', 'enr', ...
    }


def make_bill_key(congress: int, bill_type: str, bill_number: str) -> str:
    """
    Canonical join key you and your partner both use.
    Example: '108-hr-5804'
    """
    return f"{congress}-{bill_type.lower()}-{bill_number}"


def make_text_id(bill_key: str, version_code: str) -> str:
    """
    Identifier for a specific text version of a bill.
    Example: '108-hr-5804-ih'
    """
    return f"{bill_key}-{version_code.lower()}"


# Collect bill info from API
def iter_bills_packages_for_congress(
    session: requests.Session,
    congress: int,
    start_date: str,
    end_date: str,
) -> Iterable[Dict]:
    """
    Yield all BILLS packages for a given Congress using offsetMark.
    """
    url = f"{COLLECTIONS_BASE}/{start_date}"
    params = {
        "pageSize": 1000,
        "offsetMark": "*",
        "dateIssuedEndDate": end_date,
        "congress": str(congress),
        "api_key": GOVINFO_API_KEY,
    }

    seen_ids = set()

    while True:
        resp = session.get(url, params=params)
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
            yield {
                "package_id": pid,
                "last_modified": pkg.get("lastModified"),
                "doc_class": pkg.get("docClass"),
                "title": pkg.get("title"),
            }

        next_page = data.get("nextPage")
        if not next_page:
            break

        url = next_page
        if "api_key=" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}api_key={GOVINFO_API_KEY}"

        params = {}


def fetch_bill_xml(session: requests.Session, package_id: str) -> Optional[str]:
    """
    Fetch the bill's XML text for a given packageId using the
    GovInfo 'packages' service.
    """
    global _last_request_time
    if MIN_SECONDS_BETWEEN_REQUESTS > 0:
        now = time.time()
        delta = now - _last_request_time
        if delta < MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - delta)
        _last_request_time = time.time()

    url = f"{PACKAGES_BASE}/{package_id}/xml"
    params = {"api_key": GOVINFO_API_KEY}
    resp = session.get(url, params=params)
    if resp.status_code != 200:
        return None
    return resp.text


def xml_to_plain_text(xml_str: str) -> str:
    """
    Convert bill XML into roughly plain text using BeautifulSoup.
    This is a simple flattening, good enough for embeddings.
    """
    soup = BeautifulSoup(xml_str, "lxml-xml")
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_and_build_record(
    session: requests.Session,
    pkg: Dict,
    congress: int,
) -> Optional[Dict]:
    """
    Worker function for concurrent execution.
    Takes one package dict, fetches XML, converts to text, returns record dict.
    Returns None on failure.
    """
    package_id = pkg["package_id"]
    parsed = parse_package_id(package_id)
    if not parsed:
        return None
    if parsed["congress"] != congress:
        return None

    bill_type = parsed["bill_type"]
    bill_number = parsed["bill_number"]
    version_code = parsed["version_code"]

    xml_str = fetch_bill_xml(session, package_id)
    if not xml_str:
        return None

    text = xml_to_plain_text(xml_str)

    bill_key = make_bill_key(congress, bill_type, bill_number)
    text_id = make_text_id(bill_key, version_code)

    record = {
        "bill_key": bill_key,
        "text_id": text_id,
        "package_id": package_id,
        "congress": congress,
        "bill_type": bill_type,
        "bill_number": bill_number,
        "version_code": version_code,
        "doc_class": pkg.get("doc_class"),
        "title": pkg.get("title"),
        "last_modified": pkg.get("last_modified"),
        "text": text,
    }
    return record


# Scrape one session of Congress

def scrape_congress(congress: int) -> None:
    global _last_request_time
    _last_request_time = time.time()

    start_date, end_date = congress_start_end(congress)
    output_jsonl = f"{congress}_bills_text_govinfo_fast.jsonl"

    print(f"\n[INFO] Scraping Congress {congress}: {start_date} → {end_date}")
    print(f"[INFO] Output file: {output_jsonl}")

    with requests.Session() as session:
        packages = list(
            iter_bills_packages_for_congress(session, congress, start_date, end_date)
        )
        print(f"[INFO] Total unique BILLS packages collected for {congress}th: {len(packages)}")

        n_written = 0
        seen_packages = set()

        with open(output_jsonl, "w", encoding="utf-8") as out_f:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []
                for pkg in packages:
                    pid = pkg["package_id"]
                    if not pid or pid in seen_packages:
                        continue
                    seen_packages.add(pid)
                    futures.append(
                        executor.submit(fetch_and_build_record, session, pkg, congress)
                    )

                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Downloading {congress}th Congress bill texts",
                ):
                    rec = fut.result()
                    if rec is None:
                        continue
                    out_f.write(json.dumps(rec) + "\n")
                    n_written += 1

        print(f"[INFO] Done. Wrote {n_written} bill text records to {output_jsonl}")


# Loop over sessions of Congress

def main():
    for congress in range(START_CONGRESS, END_CONGRESS - 1, -1):
        scrape_congress(congress)


if __name__ == "__main__":
    main()
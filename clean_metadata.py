"""
clean_metadata.py

Reads: JSONL with bills
Writes: cleaned JSONL with origin_committee and stage softmax label

Requirements:
    pip install requests tqdm

Usage:
    export CONGRESS_API_KEY="your_key_here"
    python clean_metadata.py --input ./data/118_bills_metadata.jsonl --output ./data/118_bills_clean_metadata.jsonl --limit 100
"""
import os
import json
import time
import sys
import re
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET

import requests
from tqdm import tqdm

DEFAULT_INPUT = "118_bills_metadata.jsonl"
DEFAULT_OUTPUT = "118_bills_clean_metadata.jsonl"
CACHE_PATH = "committee_cache.json"
PROGRESS_PATH = ".clean_metadata.progress"
CONGRESS_API_BASE = "https://api.congress.gov/v3/bill"
MAX_WORKERS = 8        
SLEEP_SECONDS = 0.1 
MAX_RETRIES = 5

def safe_read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except json.JSONDecodeError:
                print(f"warning: skipping malformed json line {i} in {path}", file=sys.stderr)
    return records

STAGE_ORDER = [
    "INTRODUCED/COMMITTEE", "SUBCOMMITTEE", "FLOOR/CALENDAR", "OTHER_CHAMBER", "ENACTED_PUBLIC_LAW"
]
STAGE_COLS = [
    "stage_committee", "stage_subcommittee", "stage_floor", "stage_other_chamber", "stage_enacted"
]

def derive_stage_labels(latest_action: str):
    flags = {c: 0 for c in STAGE_COLS}
    if not isinstance(latest_action, str):
        return flags
    s = latest_action.lower().strip()

    if "became public law" in s or re.search(r"public law no\.?\s*\d+", s) or "became law" in s or "signed by the president" in s:
        flags["stage_enacted"] = 1
    if "received in the senate" in s or "received in the house" in s or "received in the other chamber" in s or re.search(r"received in the \w+", s):
        flags["stage_other_chamber"] = 1
    floor_keywords = [
        "placed on the union calendar","placed on the house calendar","placed on senate legislative calendar",
        "considered under suspension","on motion to suspend the rules","motion to reconsider laid on the table",
        "agreed to without objection","agreed to in the house","agreed to in the senate","passed the house",
        "passed the senate","ordered to be reported","ordered to be reported by","reported to the"
    ]
    if any(k in s for k in floor_keywords):
        flags["stage_floor"] = 1
    if "subcommittee on" in s or "referred to subcommittee" in s or "subcommittee," in s:
        flags["stage_subcommittee"] = 1
    if (("committee on" in s or "referred to the committee" in s or "read twice and referred to the committee" in s)
        and "subcommittee on" not in s):
        flags["stage_committee"] = 1
    if flags["stage_subcommittee"]:
        flags["stage_committee"] = 1
    if flags["stage_floor"] == 1:
        flags["stage_committee"] = 1
    if flags["stage_other_chamber"] == 1:
        flags["stage_floor"] = 1
        flags["stage_committee"] = 1
    if flags["stage_enacted"] == 1:
        flags["stage_other_chamber"] = 1
        flags["stage_floor"] = 1
        flags["stage_committee"] = 1
    return flags

def stage_flags_to_softmax_label(flags: Dict[str,int]):
    if flags.get("stage_enacted"):
        return 4, STAGE_ORDER[4]
    if flags.get("stage_other_chamber"):
        return 3, STAGE_ORDER[3]
    if flags.get("stage_floor"):
        return 2, STAGE_ORDER[2]
    if flags.get("stage_subcommittee"):
        return 1, STAGE_ORDER[1]
    if flags.get("stage_committee"):
        return 0, STAGE_ORDER[0]
    return 0, STAGE_ORDER[0]

def parse_bill_key(bill_key: str):
    if not isinstance(bill_key, str):
        return None, None, None
    parts = bill_key.split("-")
    if len(parts) < 3:
        return None, None, None
    congress = parts[0]
    bill_type = parts[1]
    bill_number = "-".join(parts[2:])
    return congress, bill_type, bill_number

class CongressAPIClient:
    def __init__(self, api_key: Optional[str] = None, cache_path: str = CACHE_PATH):
        self.api_key = api_key or os.getenv("CONGRESS_API_KEY")
        self.cache_path = Path(cache_path)
        self.session = requests.Session()
        self._load_cache()

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}
        else:
            self.cache = {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"warning: failed to save cache: {e}", file=sys.stderr)

    def _parse_committee_from_json(self, data: Dict[str,Any]) -> Optional[str]:
        if not data:
            return None
        if "committees" in data:
            c = data["committees"]
            if isinstance(c, dict):
                items = c.get("item") or c.get("items") or []
                if isinstance(items, list) and items:
                    first = items[0]
                    if isinstance(first, dict) and "name" in first:
                        return first.get("name")
            elif isinstance(c, list) and c:
                first = c[0]
                if isinstance(first, dict) and "name" in first:
                    return first.get("name")
        if "results" in data and isinstance(data["results"], list) and data["results"]:
            first = data["results"][0]
            if isinstance(first, dict):
                committees = first.get("committees") or first.get("committee") or first.get("committeesList")
                if isinstance(committees, list) and committees:
                    firstc = committees[0]
                    if isinstance(firstc, dict) and "name" in firstc:
                        return firstc.get("name")
        def rec(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if "committee" in k.lower():
                        if isinstance(v, str) and v.strip():
                            return v.strip()
                        if isinstance(v, dict):
                            for k2 in ("name","committee","committee_name","display_name"):
                                if k2 in v and isinstance(v[k2], str) and v[k2].strip():
                                    return v[k2].strip()
                        if isinstance(v, list) and v:
                            for it in v:
                                res = rec(it)
                                if res:
                                    return res
                    res = rec(v)
                    if res:
                        return res
            elif isinstance(obj, list):
                for item in obj:
                    res = rec(item)
                    if res:
                        return res
            return None
        return rec(data)

    def _parse_committee_from_xml(self, text: str) -> Optional[str]:
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return None
        for name in root.findall(".//committees//item//name"):
            if name.text and name.text.strip():
                return name.text.strip()
        for name in root.findall(".//item//name"):
            if name.text and name.text.strip():
                return name.text.strip()
        for name in root.findall(".//name"):
            if name.text and "committee" in name.text.lower():
                return name.text.strip()
        return None

    def get_committee_for_bill(self, congress: str, bill_type: str, bill_number: str) -> Optional[str]:
        if not (congress and bill_type and bill_number):
            return None
        key = f"{congress}-{bill_type}-{bill_number}"
        if key in self.cache:
            return self.cache[key]

        url = f"{CONGRESS_API_BASE}/{congress}/{bill_type}/{bill_number}/committees"
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        backoff = 1.0
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, headers=headers, timeout=25)
            except requests.RequestException as e:
                print(f"request exception for {key}: {e} (attempt {attempt})", file=sys.stderr)
                time.sleep(backoff)
                backoff *= 2
                continue

            if resp.status_code in (429, 503):
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if (retry_after and retry_after.isdigit()) else backoff
                print(f"rate limited for {key}: status {resp.status_code}, waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
                backoff = min(backoff * 2, 60)
                continue

            if resp.status_code == 404:
                self.cache[key] = None
                self._save_cache()
                return None

            if resp.status_code != 200:
                print(f"unexpected status {resp.status_code} for {key} (attempt {attempt}). body: {resp.text[:200]}", file=sys.stderr)
                time.sleep(backoff)
                backoff *= 2
                continue

            committee = None
            ctype = resp.headers.get("Content-Type", "").lower()
            if "application/json" in ctype or resp.text.strip().startswith("{"):
                try:
                    data = resp.json()
                except Exception:
                    data = None
                if data:
                    committee = self._parse_committee_from_json(data)
            if not committee:
                try:
                    committee = self._parse_committee_from_xml(resp.text)
                except Exception:
                    committee = committee  
            self.cache[key] = committee
            self._save_cache()
            time.sleep(SLEEP_SECONDS)
            return committee
        self.cache[key] = None
        self._save_cache()
        return None

def clean_metadata(input_path, output_path, bill_limit=None):
    records = safe_read_jsonl(input_path)
    total_in_file = len(records)

    # Resume index reading
    resume_index = 0
    if Path(PROGRESS_PATH).exists():
        try:
            with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
                resume_index = int(f.read().strip())
        except Exception:
            resume_index = 0

    # Compute start/end bounds
    start = resume_index
    if bill_limit:
        end = min(total_in_file, start + bill_limit)
    else:
        end = total_in_file

    if start >= end:
        print(f"Nothing to do: start={start} end={end}", file=sys.stderr)
        return

    to_process = records[start:end]
    print(f"Starting. total_in_file={total_in_file}, processing_index_range=[{start}:{end}] ({len(to_process)} records), resuming_from_index={resume_index}")

    # Open output file in append
    mode = "a" if resume_index > 0 and Path(output_path).exists() else "w"
    client = CongressAPIClient()

    # Use threadpool to fetch committees in parallel 
    with open(output_path, mode, encoding="utf-8") as outf:
        progress_bar = tqdm(total=len(to_process), desc="Bills", unit="bill")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {} 
            for local_idx, rec in enumerate(to_process):
                abs_idx = start + local_idx
                bill_key = rec.get("bill_key")
                congress, bill_type, bill_number = parse_bill_key(bill_key) if bill_key else (None, None, None)
                if not (congress and bill_type and bill_number):
                    future = executor.submit(lambda: None)
                    future_map[future] = (abs_idx, rec)
                else:
                    future = executor.submit(client.get_committee_for_bill, congress, bill_type, bill_number)
                    future_map[future] = (abs_idx, rec)

            # iterate results as they complete
            for fut in as_completed(future_map):
                abs_idx, rec = future_map[fut]
                try:
                    origin_committee = fut.result()
                except Exception as e:
                    print(f"error fetching committee for index {abs_idx} bill_key={rec.get('bill_key')}: {e}", file=sys.stderr)
                    origin_committee = None

                latest_action = rec.get("latest_action") or ""
                flags = derive_stage_labels(latest_action)
                stage_id, stage_name = stage_flags_to_softmax_label(flags)

                rec_out = dict(rec)
                rec_out.pop("bill_number", None)  # remove bill number if present
                rec_out["origin_committee"] = origin_committee
                rec_out["stage_label"] = int(stage_id)
                rec_out["stage_name"] = stage_name

                outf.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                outf.flush()

                # update progress file: next index to resume from
                next_index = abs_idx + 1
                try:
                    with open(PROGRESS_PATH, "w", encoding="utf-8") as pf:
                        pf.write(str(next_index))
                except Exception as e:
                    print(f"warning: failed to write progress file: {e}", file=sys.stderr)

                progress_bar.update(1)

        progress_bar.close()
    print(f"Finished. wrote records up to index {end} to {output_path}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    clean_metadata(args.input, args.output, bill_limit=args.limit)

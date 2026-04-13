"""
1_download.py
-------------
Crawls the NMSU PDS InSight directories and downloads all calibrated CSVs
for TWINS and PS instruments, preserving the server's chunk folder structure.

Output layout:
    data/raw/twins/<chunk_folder>/*.csv
    data/raw/ps/<chunk_folder>/*.csv

Re-run safe: skips files that already exist.
"""

import os
import time
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

TWINS_URL = "https://atmos.nmsu.edu/PDS/data/PDS4/InSight/twins_bundle/data_calibrated/"
PS_URL    = "https://atmos.nmsu.edu/PDS/data/PDS4/InSight/ps_bundle/data_calibrated/"

OUTPUT_ROOT = Path("data/raw")

# Number of parallel download threads (tune down if you hit rate limits)
MAX_WORKERS = 8

# Seconds to wait between retries on failure
RETRY_DELAY = 3
MAX_RETRIES = 3

# ── Helpers ───────────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "InSight-Data-Pipeline/1.0 (research)"})


def get_links(url: str, suffix: str = "/") -> list[str]:
    """
    Fetch an HTML directory listing and return all hrefs ending with `suffix`.
    Uses suffix='/' to get subdirectory links, suffix='.csv' to get CSV links.
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = SESSION.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            links = [
                a["href"] for a in soup.find_all("a", href=True)
                if a["href"].endswith(suffix)
                and not a["href"].startswith("?")   # ignore sort query strings
                and not a["href"].startswith("/PDS") # ignore parent nav links
            ]
            return links
        except Exception as e:
            print(f"  [warn] Failed to list {url} (attempt {attempt+1}): {e}")
            time.sleep(RETRY_DELAY)
    return []


def download_file(url: str, dest: Path) -> tuple[str, bool]:
    """
    Download a single file to dest. Returns (url, success).
    Skips if dest already exists.
    """
    if dest.exists():
        return url, True  # already downloaded

    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(MAX_RETRIES):
        try:
            resp = SESSION.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            return url, True
        except Exception as e:
            print(f"  [warn] Failed to download {url} (attempt {attempt+1}): {e}")
            if dest.exists():
                dest.unlink()  # remove partial file
            time.sleep(RETRY_DELAY)

    return url, False


def collect_download_tasks(base_url: str, local_root: Path) -> list[tuple[str, Path]]:
    """
    Crawl base_url for chunk subdirectories, then collect all .csv links
    within each subdirectory. Returns list of (file_url, local_path) tuples.
    """
    tasks = []
    chunk_dirs = get_links(base_url, suffix="/")

    if not chunk_dirs:
        print(f"[error] No subdirectories found at {base_url}")
        print("        Check that the URL is accessible and returns an HTML listing.")
        return tasks

    print(f"  Found {len(chunk_dirs)} chunk directories at {base_url}")

    for chunk_dir in chunk_dirs:
        chunk_url = base_url + chunk_dir
        csv_links = get_links(chunk_url, suffix=".csv")

        # Strip trailing slash from dir name for local folder
        chunk_name = chunk_dir.rstrip("/")
        chunk_local = local_root / chunk_name

        for csv_link in csv_links:
            file_url = chunk_url + csv_link
            dest = chunk_local / csv_link
            tasks.append((file_url, dest))

    return tasks


def run_downloads(tasks: list[tuple[str, Path]], label: str):
    """Run a list of (url, dest) download tasks in parallel with progress reporting."""
    total = len(tasks)
    if total == 0:
        print(f"[{label}] No files to download.")
        return

    # Count already-done
    already_done = sum(1 for _, dest in tasks if dest.exists())
    print(f"[{label}] {total} files total, {already_done} already downloaded, "
          f"{total - already_done} to fetch.")

    completed = already_done
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_file, url, dest): url
            for url, dest in tasks
            if not dest.exists()
        }

        for future in as_completed(futures):
            url, success = future.result()
            completed += 1
            if success:
                if completed % 50 == 0 or completed == total:
                    print(f"  [{label}] {completed}/{total} done...")
            else:
                failed.append(url)
                print(f"  [{label}] FAILED: {url}")

    print(f"[{label}] Download complete. {len(failed)} failures.")
    if failed:
        fail_log = OUTPUT_ROOT / f"{label}_failed_downloads.txt"
        fail_log.write_text("\n".join(failed))
        print(f"  Failed URLs saved to {fail_log}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== InSight Data Downloader ===\n")

    for label, base_url in [("TWINS", TWINS_URL), ("PS", PS_URL)]:
        local_root = OUTPUT_ROOT / label.lower()
        print(f"[{label}] Scanning {base_url} ...")
        tasks = collect_download_tasks(base_url, local_root)
        print(f"[{label}] Collected {len(tasks)} CSV files to download.")
        run_downloads(tasks, label)
        print()

    print("All downloads finished.")
    print(f"Data saved under: {OUTPUT_ROOT.resolve()}")

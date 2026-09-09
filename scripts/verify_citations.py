"""Batch-verify .bib references against CrossRef, Semantic Scholar, and OpenAlex."""

import re
import json
import time
import urllib.request
import urllib.parse
import urllib.error
import ssl
from pathlib import Path
from typing import Optional

# Skip SSL verification for API calls
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

BIB_PATH = Path(__file__).resolve().parent.parent.parent / "paper" / "references" / "references.bib"
REPORT_PATH = Path(__file__).resolve().parent.parent / "results" / "citation_verification_report.json"


def parse_bib(path: Path) -> list[dict]:
    """Parse .bib file into list of entries."""
    text = path.read_text(encoding="utf-8")
    entries = []
    # Match @type{key, ... }
    pattern = re.compile(r"@(\w+)\{([^,]+),\s*(.*?)\n\}", re.DOTALL)
    for m in pattern.finditer(text):
        entry_type, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields: dict[str, str] = {"_type": entry_type, "_key": key}
        # Parse field = {value} or field = value
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}(?:\s*,|\s*$)", body, re.DOTALL):
            fields[fm.group(1).lower()] = fm.group(2).strip()
        for fm in re.finditer(r"(\w+)\s*=\s*(\d+)", body):
            fields[fm.group(1).lower()] = fm.group(2).strip()
        entries.append(fields)
    return entries


def extract_authors_lastnames(author_str: str) -> list[str]:
    """Extract last names from BibTeX author string."""
    authors = re.split(r"\s+and\s+", author_str)
    lastnames = []
    for a in authors:
        a = a.strip()
        if a.lower() == "others":
            continue
        if "," in a:
            lastnames.append(a.split(",")[0].strip())
        else:
            parts = a.split()
            if parts:
                lastnames.append(parts[-1].strip())
    # Clean LaTeX accents
    cleaned = []
    for n in lastnames:
        n = re.sub(r"\{.*?\}", "", n)
        n = re.sub(r"[\\{}'`\"~^]", "", n)
        cleaned.append(n.strip())
    return [n for n in cleaned if n]


def api_get(url: str, headers: Optional[dict] = None) -> Optional[dict]:
    """Make GET request and return JSON."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "SecHarness-CiteVerify/1.0 (mailto:research@example.com)")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as e:
        return None


def verify_crossref(title: str, authors: list[str], year: str, doi: Optional[str] = None) -> Optional[dict]:
    """Verify via CrossRef API."""
    if doi:
        data = api_get(f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}")
        if data and data.get("status") == "ok":
            item = data["message"]
            return {
                "source": "crossref",
                "doi": item.get("DOI", ""),
                "title": item.get("title", [""])[0],
                "year": str(item.get("published-print", item.get("published-online", item.get("created", {})))
                           .get("date-parts", [[None]])[0][0] or ""),
                "journal": item.get("container-title", [""])[0],
                "authors": [f"{a.get('family', '')} {a.get('given', '')}".strip()
                           for a in item.get("author", [])],
            }

    # Search by title
    query = urllib.parse.quote(title[:200])
    data = api_get(f"https://api.crossref.org/works?query.title={query}&rows=3")
    if not data or data.get("status") != "ok":
        return None

    for item in data.get("message", {}).get("items", []):
        item_title = (item.get("title") or [""])[0].lower()
        if _title_similarity(title.lower(), item_title) > 0.7:
            return {
                "source": "crossref",
                "doi": item.get("DOI", ""),
                "title": (item.get("title") or [""])[0],
                "year": str(item.get("published-print", item.get("published-online", item.get("created", {})))
                           .get("date-parts", [[None]])[0][0] or ""),
                "journal": (item.get("container-title") or [""])[0],
                "authors": [f"{a.get('family', '')} {a.get('given', '')}".strip()
                           for a in item.get("author", [])],
            }
    return None


def verify_semanticscholar(title: str, authors: list[str]) -> Optional[dict]:
    """Verify via Semantic Scholar API."""
    query = urllib.parse.quote(title[:200])
    data = api_get(
        f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=3"
        f"&fields=title,authors,year,externalIds,venue,journal"
    )
    if not data:
        return None

    for paper in data.get("data", []):
        if _title_similarity(title.lower(), (paper.get("title") or "").lower()) > 0.7:
            journal = ""
            if paper.get("journal"):
                journal = paper["journal"].get("name", "")
            elif paper.get("venue"):
                journal = paper["venue"]
            return {
                "source": "semanticscholar",
                "doi": (paper.get("externalIds") or {}).get("DOI", ""),
                "title": paper.get("title", ""),
                "year": str(paper.get("year", "")),
                "journal": journal,
                "authors": [a.get("name", "") for a in paper.get("authors", [])],
            }
    return None


def verify_openalex(title: str, authors: list[str]) -> Optional[dict]:
    """Verify via OpenAlex API."""
    query = urllib.parse.quote(title[:200])
    data = api_get(f"https://api.openalex.org/works?search={query}&per_page=3")
    if not data:
        return None

    for work in data.get("results", []):
        work_title = (work.get("title") or "").lower()
        if _title_similarity(title.lower(), work_title) > 0.7:
            return {
                "source": "openalex",
                "doi": (work.get("doi") or "").replace("https://doi.org/", ""),
                "title": work.get("title", ""),
                "year": str(work.get("publication_year", "")),
                "journal": ((work.get("primary_location") or {}).get("source") or {}).get("display_name", ""),
                "authors": [
                    (a.get("author", {}) or {}).get("display_name", "")
                    for a in work.get("authorships", [])
                ],
            }
    return None


def _title_similarity(a: str, b: str) -> float:
    """Simple word-overlap Jaccard similarity."""
    a_clean = re.sub(r"[^a-z0-9\s]", "", a.lower())
    b_clean = re.sub(r"[^a-z0-9\s]", "", b.lower())
    wa = set(a_clean.split())
    wb = set(b_clean.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def check_metadata_match(bib_entry: dict, verified: dict) -> list[dict]:
    """Check if verified metadata matches bib entry. Return list of mismatches."""
    mismatches = []
    # Year check
    bib_year = bib_entry.get("year", "")
    v_year = verified.get("year", "")
    if bib_year and v_year and bib_year != v_year:
        mismatches.append({"field": "year", "bib_value": bib_year, "verified_value": v_year})

    # Journal check
    bib_journal = bib_entry.get("journal", bib_entry.get("booktitle", ""))
    v_journal = verified.get("journal", "")
    if bib_journal and v_journal:
        if _title_similarity(bib_journal.lower(), v_journal.lower()) < 0.4:
            mismatches.append({"field": "journal", "bib_value": bib_journal, "verified_value": v_journal})

    return mismatches


def main() -> None:
    entries = parse_bib(BIB_PATH)
    print(f"Parsed {len(entries)} entries from {BIB_PATH.name}\n")

    results: list[dict] = []
    counts = {"verified": 0, "mismatch": 0, "not_found": 0, "doi_missing": 0}

    for i, entry in enumerate(entries):
        key = entry["_key"]
        title = entry.get("title", "").replace("{", "").replace("}", "")
        authors = extract_authors_lastnames(entry.get("author", ""))
        year = entry.get("year", "")
        doi = entry.get("doi", None)
        is_arxiv = "arxiv" in entry.get("journal", "").lower() or "arxiv" in entry.get("note", "").lower() or "arxiv" in entry.get("howpublished", "").lower()
        is_web = entry["_type"] == "misc" and entry.get("howpublished", "")

        print(f"[{i+1}/{len(entries)}] {key}: ", end="", flush=True)

        # Try CrossRef first, then Semantic Scholar, then OpenAlex
        verified = None
        for verify_fn in [
            lambda: verify_crossref(title, authors, year, doi),
            lambda: verify_semanticscholar(title, authors),
            lambda: verify_openalex(title, authors),
        ]:
            verified = verify_fn()
            if verified:
                break
            time.sleep(0.3)

        record: dict = {
            "key": key,
            "title": title,
            "bib_year": year,
            "bib_doi": doi or "",
            "is_arxiv": is_arxiv,
            "is_web": is_web,
        }

        if verified:
            mismatches = check_metadata_match(entry, verified)
            if mismatches:
                status = "MISMATCH"
                counts["mismatch"] += 1
                print(f"MISMATCH ({verified['source']})")
                for mm in mismatches:
                    print(f"    {mm['field']}: '{mm['bib_value']}' -> '{mm['verified_value']}'")
            else:
                status = "VERIFIED"
                counts["verified"] += 1
                print(f"VERIFIED ({verified['source']})")

            record["status"] = status
            record["verified"] = verified
            record["mismatches"] = mismatches

            if not verified.get("doi") and not is_arxiv and not is_web:
                counts["doi_missing"] += 1
                record["doi_missing"] = True
                print(f"    WARNING: No DOI found")
        else:
            if is_web:
                status = "VERIFIED_WEB"
                counts["verified"] += 1
                print("VERIFIED_WEB (web resource, skip API check)")
            else:
                status = "NOT_FOUND"
                counts["not_found"] += 1
                print("NOT_FOUND")
            record["status"] = status
            record["verified"] = None

        results.append(record)
        time.sleep(0.5)  # Rate limiting

    # Summary
    print("\n" + "=" * 60)
    print("CITATION VERIFICATION REPORT")
    print("=" * 60)
    print(f"Total entries:            {len(entries)}")
    print(f"Verified (exact match):   {counts['verified']}")
    print(f"Metadata mismatch:        {counts['mismatch']}")
    print(f"Not found:                {counts['not_found']}")
    print(f"DOI missing (non-arXiv):  {counts['doi_missing']}")

    if counts["mismatch"] > 0:
        print("\n--- MISMATCHES ---")
        for r in results:
            if r["status"] == "MISMATCH":
                print(f"  {r['key']}:")
                for mm in r.get("mismatches", []):
                    print(f"    {mm['field']}: bib='{mm['bib_value']}' -> db='{mm['verified_value']}'")
                v = r["verified"]
                if v and v.get("doi"):
                    print(f"    correct DOI: {v['doi']}")

    if counts["not_found"] > 0:
        print("\n--- NOT FOUND (manual review required) ---")
        for r in results:
            if r["status"] == "NOT_FOUND":
                print(f"  {r['key']}: {r['title'][:80]}")

    # Save full report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()

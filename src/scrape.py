from __future__ import annotations

import csv
import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from time import sleep

import pandas as pd
import requests

from utils import (
    RAW_DIR,
    clean_html,
    ensure_directories,
    normalize_tags,
    normalize_whitespace,
    setup_logging,
    tags_to_string,
)


SITEMAP_INDEX_URL = "https://remoteok.com/sitemap.xml"
OUTPUT_FILE = RAW_DIR / "remoteok_jobs_raw.csv"
SITEMAP_URLS_FILE = RAW_DIR / "remoteok_sitemap_job_urls.csv"

HEADERS = {
    "User-Agent": "remoteok-ml-academic-project/1.0",
    "Accept": "text/html,application/xml",
}

COLUMNS = [
    "id",
    "position",
    "company",
    "location",
    "tags",
    "salary_min",
    "salary_max",
    "description",
    "url",
    "date",
]


def get_xml_links(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [loc.text for loc in root.findall(".//sm:loc", namespace) if loc.text]


def is_job_url(url: str) -> bool:
    slug = url.rstrip("/").split("/")[-1]
    return "/remote-jobs/" in url and bool(re.search(r"\d", slug))


def get_job_urls(session: requests.Session, delay: float = 1.0) -> list[str]:
    response = session.get(SITEMAP_INDEX_URL, headers=HEADERS, timeout=45)
    response.raise_for_status()

    sitemap_urls = [
        url for url in get_xml_links(response.text)
        if "sitemap-jobs-" in url
    ]

    job_urls = []

    for index, sitemap_url in enumerate(sitemap_urls, start=1):
        response = session.get(sitemap_url, headers=HEADERS, timeout=45)
        response.raise_for_status()

        urls = get_xml_links(response.text)
        job_urls.extend(url for url in urls if is_job_url(url))

        if delay > 0 and index < len(sitemap_urls):
            sleep(delay)

    return list(dict.fromkeys(job_urls))


def save_job_urls(job_urls: list[str]) -> None:
    pd.DataFrame({"url": job_urls}).to_csv(
        SITEMAP_URLS_FILE,
        index=False,
        encoding="utf-8"
    )


def get_job_id(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    matches = re.findall(r"\d+", slug)
    return matches[-1] if matches else ""


def get_json_ld_blocks(html_text: str) -> list[dict]:
    blocks = re.findall(
        r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    result = []

    for block in blocks:
        try:
            data = json.loads(html.unescape(block.strip()))
        except json.JSONDecodeError:
            continue

        if isinstance(data, dict):
            result.append(data)
        elif isinstance(data, list):
            result.extend(item for item in data if isinstance(item, dict))

    return result


def find_job_posting(html_text: str) -> dict:
    for block in get_json_ld_blocks(html_text):
        block_type = block.get("@type")

        if block_type == "JobPosting":
            return block

        if isinstance(block_type, list) and "JobPosting" in block_type:
            return block

        graph = block.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                if isinstance(item, dict) and item.get("@type") == "JobPosting":
                    return item

    return {}


def parse_salary(job: dict) -> tuple[object, object]:
    salary = job.get("baseSalary") or {}

    if not isinstance(salary, dict):
        return None, None

    value = salary.get("value") or {}

    if isinstance(value, dict):
        min_salary = value.get("minValue") or value.get("value")
        max_salary = value.get("maxValue") or value.get("value")
        return min_salary, max_salary

    return value, value


def parse_company(job: dict) -> str:
    company = job.get("hiringOrganization") or {}

    if isinstance(company, dict):
        return normalize_whitespace(company.get("name"))

    return normalize_whitespace(company)


def parse_location(job: dict) -> str:
    values = []

    applicant_locations = job.get("applicantLocationRequirements") or []
    if isinstance(applicant_locations, dict):
        applicant_locations = [applicant_locations]

    for location in applicant_locations:
        if isinstance(location, dict):
            values.append(normalize_whitespace(location.get("name")))

    job_locations = job.get("jobLocation") or []
    if isinstance(job_locations, dict):
        job_locations = [job_locations]

    for location in job_locations:
        if not isinstance(location, dict):
            continue

        address = location.get("address") or {}

        if isinstance(address, dict):
            for field in ["addressLocality", "addressRegion", "addressCountry"]:
                values.append(normalize_whitespace(address.get(field)))

    values = [
        value for value in values
        if value and value.lower() not in {"nan", "none"}
    ]

    return "|".join(dict.fromkeys(values))


def parse_tags(html_text: str) -> str:
    tag_values = re.findall(
        r'class=["\'][^"\']*\btag\b[^"\']*["\'][^>]*>(.*?)</',
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    tags = [clean_html(value) for value in tag_values]

    if not tags:
        slugs = re.findall(
            r'href=["\']/remote-([a-z0-9-]+)-jobs["\']',
            html_text,
            flags=re.IGNORECASE,
        )
        tags = [slug.replace("-", " ") for slug in slugs]

    return tags_to_string(normalize_tags(tags))


def parse_job_page(url: str, html_text: str) -> dict:
    job = find_job_posting(html_text)
    salary_min, salary_max = parse_salary(job)

    return {
        "id": get_job_id(url),
        "position": normalize_whitespace(job.get("title")),
        "company": parse_company(job),
        "location": parse_location(job),
        "tags": parse_tags(html_text),
        "salary_min": salary_min,
        "salary_max": salary_max,
        "description": clean_html(job.get("description")),
        "url": url,
        "date": normalize_whitespace(job.get("datePosted")),
    }


def write_rows(path: Path, rows: list[dict], write_header: bool) -> None:
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)

        if write_header:
            writer.writeheader()

        for row in rows:
            writer.writerow({column: row.get(column) for column in COLUMNS})


def get_existing_urls(output_file: Path) -> set[str]:
    if not output_file.exists():
        return set()

    try:
        return set(
            pd.read_csv(output_file, usecols=["url"])["url"]
            .dropna()
            .astype(str)
        )
    except Exception:
        return set()


def clean_output(output_file: Path) -> pd.DataFrame:
    df = pd.read_csv(output_file)

    df = df.reindex(columns=COLUMNS)
    df = df[df["position"].astype(str).str.strip().ne("")]
    df = df.drop_duplicates(subset=["id"], keep="first")
    df = df.drop_duplicates(subset=["position", "company", "url"], keep="first")

    df.to_csv(output_file, index=False, encoding="utf-8")
    return df


def scrape_remoteok_sitemap(
    output_file: Path = OUTPUT_FILE,
    delay: float = 1.0,
    batch_size: int = 100,
    resume: bool = True,
) -> pd.DataFrame:
    logger = setup_logging("remoteok_sitemap_scraper")
    ensure_directories()

    if not resume and output_file.exists():
        output_file.unlink()

    with requests.Session() as session:
        logger.info("Sitemap okunuyor: %s", SITEMAP_INDEX_URL)

        job_urls = get_job_urls(session, delay=delay)
        save_job_urls(job_urls)

        existing_urls = get_existing_urls(output_file) if resume else set()
        remaining_urls = [url for url in job_urls if url not in existing_urls]

        logger.info("%s ilan bulundu.", len(job_urls))
        logger.info("%s ilan daha once cekilmis.", len(existing_urls))
        logger.info("%s ilan cekilecek.", len(remaining_urls))

        rows = []
        write_header = not output_file.exists()

        for index, url in enumerate(remaining_urls, start=1):
            try:
                response = session.get(url, headers=HEADERS, timeout=60)
                response.raise_for_status()

                row = parse_job_page(url, response.text)

                if row["position"] and row["description"]:
                    rows.append(row)

            except requests.RequestException as error:
                logger.warning("URL atlandi: %s | %s", url, error)

            if len(rows) >= batch_size:
                write_rows(output_file, rows, write_header)
                write_header = False
                rows = []
                logger.info("%s/%s URL islendi.", index, len(remaining_urls))

            if delay > 0 and index < len(remaining_urls):
                sleep(delay)

        if rows:
            write_rows(output_file, rows, write_header)

    df = clean_output(output_file)
    logger.info("%s benzersiz ilan kaydedildi: %s", len(df), output_file)

    return df


if __name__ == "__main__":
    scrape_remoteok_sitemap()
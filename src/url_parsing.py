import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import trafilatura
from bs4 import BeautifulSoup
from tabulate import tabulate

logger = logging.getLogger(__name__)


class URLParser:
    """Parse URLs with trafilatura and export reports in pipeline-compatible JSON.

    Export schema matches processed reports consumed by text splitter/ingestion:
    {
      "metainfo": {
        "sha1_name": "...",
        "source_type": "url",
        "url": "..."
      },
      "content": {
        "chunks": null,
        "pages": [{"page": 1, "text": "..."}]
      }
    }
    """

    def __init__(
        self,
        user_agent: Optional[str] = None,
        crawl_delay: float = 0.5,
        include_tables: bool = True,
        output_format: str = "markdown",
        output_dir: Optional[Path] = None,
    ):
        self.user_agent = user_agent or "rag-challenge-url-parser/1.0"
        self.crawl_delay = crawl_delay
        self.include_tables = include_tables
        self.output_format = output_format if output_format in ("plain", "markdown") else "markdown"
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _extract_with_trafilatura(self, html: str) -> Optional[str]:
        if not html:
            return None
        return trafilatura.extract(
            html,
            include_comments=False,
            include_tables=self.include_tables,
            output_format=self.output_format,
        )

    def _fetch_with_trafilatura(self, url: str) -> Optional[str]:
        try:
            return trafilatura.fetch_url(url)
        except Exception as error:
            logger.debug("trafilatura.fetch_url error for %s: %s", url, error)
            return None

    def _safe_filename(self, url: str, suffix: str = ".json") -> str:
        parsed = urlparse(url)
        base = parsed.netloc + parsed.path
        if parsed.query:
            base += "?" + parsed.query
        name = re.sub(r"[^0-9A-Za-z]+", "_", base).strip("_")
        if not name:
            name = hashlib.sha1(url.encode("utf-8")).hexdigest()
        if len(name) > 200:
            name = name[:200]
        short_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
        return f"{name}_{short_hash}{suffix}"

    @staticmethod
    def _get_sha1_name(url: str) -> str:
        return hashlib.sha1(url.encode("utf-8")).hexdigest()

    def _table_to_markdown(self, table_tag) -> str:
        rows = []
        for tr in table_tag.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            row = [" ".join(cell.stripped_strings) for cell in cells]
            if row:
                rows.append(row)

        if not rows:
            return ""

        max_cols = max(len(row) for row in rows)
        normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]

        headers = normalized_rows[0]
        body = normalized_rows[1:]
        if not body:
            body = [[""] * len(headers)]

        return tabulate(body, headers=headers, tablefmt="github")

    def _extract_tables(self, html: str) -> List[Dict]:
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        extracted_tables = []

        for table_id, table_tag in enumerate(soup.find_all("table")):
            table_html = str(table_tag)
            table_markdown = self._table_to_markdown(table_tag)
            if not table_markdown.strip():
                continue

            extracted_tables.append({
                "table_id": table_id,
                "page": 1,
                "bbox": [],
                "#-rows": len(table_tag.find_all("tr")),
                "#-cols": max((len(row.find_all(["th", "td"])) for row in table_tag.find_all("tr")), default=0),
                "markdown": table_markdown,
                "html": table_html,
                "json": {},
            })

        return extracted_tables

    def _build_output_payload(self, url: str, text: str, html: Optional[str] = None) -> Dict:
        sha1_name = self._get_sha1_name(url)
        tables = self._extract_tables(html or "")

        page_content = [{"type": "text", "text": text}]
        for table in tables:
            page_content.append({"type": "table", "table_id": table["table_id"]})

        return {
            "metainfo": {
                "sha1_name": sha1_name,
                "source_type": "url",
                "url": url,
                "tables_amount": len(tables),
            },
            "content": [{"page": 1, "content": page_content}],
            "tables": tables,
        }

    def parse_urls(self, urls: List[Union[str, Dict[str, str]]]) -> None:
        """Parse URL list and write one JSON report per URL.

        `urls` accepts either:
        - list[str]
        - list[{"url": "..."}]
        """
        for index, item in enumerate(urls):
            if isinstance(item, str):
                url = item
            else:
                url = item.get("url", "")

            if not url:
                logger.warning("Skipping malformed URL record at index %d", index)
                continue

            logger.info("Parsing %d/%d: %s", index + 1, len(urls), url)
            time.sleep(self.crawl_delay)

            downloaded = self._fetch_with_trafilatura(url)
            extracted_text = self._extract_with_trafilatura(downloaded) if downloaded else None

            if not extracted_text:
                logger.warning("Failed to extract text from %s (skipping)", url)
                continue

            json_obj = self._build_output_payload(url, extracted_text, downloaded)
            out_path = self.output_dir / self._safe_filename(url)

            with out_path.open("w", encoding="utf-8") as file:
                json.dump(json_obj, file, ensure_ascii=False, indent=2)
            logger.info("Saved parsed URL data to %s", out_path)
import json
import logging
import re
import hashlib
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import trafilatura

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class URLParser:
    """
    URLParser extracts text from web pages (using trafilatura) and writes per-URL JSON files
    to an output directory. Each JSON has the structure:
    {
      "metainfo": {"url": url},
      "content": {"chunks": None, "pages": [{"page": 1, "text": text}]}
    }
    """

    def __init__(
        self,
        user_agent: Optional[str] = None,
        crawl_delay: float = 0.5,
        include_tables: bool = True,
        output_format: str = "markdown",
        output_dir: Optional[Path] = None
    ):
        self.user_agent = user_agent or "my-scraper-bot/1.0 (+https://example.com)"
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
            output_format=self.output_format
        )

    def _fetch_with_trafilatura(self, url: str) -> Optional[str]:
        try:
            downloaded = trafilatura.fetch_url(url)
            return downloaded
        except Exception as e:
            logger.debug("trafilatura.fetch_url error for %s: %s", url, e)
            return None

    def _safe_filename(self, url: str, suffix: str = "_data.json") -> str:
        """
        Create a filesystem-safe filename from a URL.
        Keeps human-readable parts (netloc + path) but replaces non-alnum with underscores
        and appends a short SHA1 hash for uniqueness.
        """
        parsed = urlparse(url)
        base = parsed.netloc + parsed.path
        if parsed.query:
            base += "?" + parsed.query
        # replace non-alphanumeric characters with underscore
        name = re.sub(r'[^0-9A-Za-z]+', '_', base).strip('_')
        if not name:
            name = hashlib.sha1(url.encode('utf-8')).hexdigest()
        # limit length for safety
        max_base_len = 200
        if len(name) > max_base_len:
            name = name[:max_base_len]
        short_hash = hashlib.sha1(url.encode('utf-8')).hexdigest()[:8]
        filename = f"{name}_{short_hash}{suffix}"
        return filename

    def parse_urls(self, urls: List[str]) -> None:
        """
        Parse a list of URLs, extract text with trafilatura and write each result to a JSON file
        in self.output_dir. The function does not return anything; results are saved to disk.

        JSON format:
        {
          "metainfo": {"url": url},
          "content": {"chunks": None, "pages": [{"page": 1, "text": text}]}
        }
        """
        for i, url in enumerate(urls):
            logger.info("Parsing %d/%d: %s", i + 1, len(urls), url)
            time.sleep(self.crawl_delay)  # polite delay

            text = ""

            # 1) Attempt: trafilatura.fetch_url()
            downloaded = self._fetch_with_trafilatura(url)
            if downloaded:
                logger.debug("trafilatura.fetch_url returned HTML for %s", url)
                extracted = self._extract_with_trafilatura(downloaded)
                if extracted:
                    text = extracted

            if not text:
                logger.warning("Failed to extract text from %s (skipping url)", url)
                continue

            json_obj = {
                "metainfo": {"url": url},
                "content": {
                    "chunks": None,
                    "pages": [
                        {"page": 1, "text": text or ""}
                    ]
                }
            }

            filename = self._safe_filename(url)
            out_path = self.output_dir / filename

            try:
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(json_obj, f, ensure_ascii=False, indent=2)
                logger.info("Saved parsed data to %s", out_path)
            except Exception as e:
                logger.error("Failed to write JSON for %s to %s: %s", url, out_path, e)

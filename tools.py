"""Host-side research tools with bounded retries and compact results."""
import json
import os
import random
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

ARXIV_URL = "https://export.arxiv.org/api/query"
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"
_CLIENT = httpx.Client(timeout=30.0, follow_redirects=True)
_ARXIV_LOCK = threading.Lock()
_ARXIV_LAST_CALL = None


class RetryableError(Exception):
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            delay = exc.retry_after
            if delay is None:
                delay = base * 2**attempt + random.uniform(0, base)
            delay = max(0.0, min(cap, delay))
            if __name__ == "__main__":
                print(f"  Retry {attempt + 2}/{attempts} in {delay:.1f}s: {_error(exc)}", file=sys.stderr, flush=True)
            time.sleep(delay)


def _request(method, url, **kwargs):
    try:
        response = _CLIENT.request(method, url, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(str(exc)) from exc
    if response.status_code in {429, 500, 502, 503, 504}:
        try:
            retry_after = float(response.headers.get("Retry-After", ""))
        except ValueError:
            retry_after = None
        raise RetryableError(f"HTTP {response.status_code}", retry_after)
    response.raise_for_status()
    return response


def _error(exc):
    message = f"{type(exc).__name__}: {exc}"
    key = os.getenv("EXA_API_KEY", "")
    if key:
        for secret in (key, quote(key, safe="")):
            message = message.replace(secret, "[REDACTED]")
    return "ERROR: " + message


def _compact(value):
    return " ".join(str(value or "").split())


def _records(records):
    return json.dumps(records, ensure_ascii=False, separators=(",", ":")) if records else "NO RESULTS"


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv by keywords, newest first. Return JSON records with id, url, published, title and summary; NO RESULTS or ERROR on failure."""
    try:
        terms = [term for term in re.findall(r"[^\W_]+(?:-[^\W_]+)*", query) if term.upper() not in {"AND", "OR", "NOT"}]
        if not terms:
            return "NO RESULTS"

        def fetch():
            global _ARXIV_LAST_CALL
            with _ARXIV_LOCK:
                if _ARXIV_LAST_CALL is not None:
                    delay = 3.0 - (time.monotonic() - _ARXIV_LAST_CALL)
                    if delay > 0:
                        time.sleep(delay)
                _ARXIV_LAST_CALL = time.monotonic()
                return _request("GET", ARXIV_URL, params={
                    "search_query": " AND ".join(f"all:{term}" for term in terms),
                    "sortBy": "submittedDate", "sortOrder": "descending",
                    "max_results": max(1, min(30, max_results)), "start": 0,
                })

        root = ET.fromstring(with_retry(fetch, attempts=7, cap=60).text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("a:entry", ns):
            paper_id = re.sub(r"v\d+$", "", entry.findtext("a:id", "", ns).split("/abs/")[-1])
            records.append({
                "id": paper_id, "url": f"https://arxiv.org/abs/{paper_id}",
                "published": entry.findtext("a:published", "", ns)[:10],
                "title": _compact(entry.findtext("a:title", "", ns)),
                "summary": _compact(entry.findtext("a:summary", "", ns))[:600],
            })
        return _records(records)
    except Exception as exc:
        return _error(exc)


def _hf_records(items, *, prefer_ai=False, keyword=""):
    records = []
    for item in items:
        paper = item.get("paper") or {}
        paper_id = paper.get("id")
        if not paper_id:
            continue
        title = _compact(paper.get("title") or item.get("title"))
        summary = _compact((paper.get("ai_summary") or item.get("ai_summary") if prefer_ai else None)
                           or paper.get("summary") or item.get("summary"))
        if keyword and keyword.casefold() not in (title + " " + summary).casefold():
            continue
        records.append({
            "id": paper_id, "url": f"https://huggingface.co/papers/{paper_id}",
            "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
            "title": title, "summary": summary[:600],
            "upvotes": paper.get("upvotes") or item.get("upvotes") or 0,
            "github": paper.get("githubRepo") or item.get("githubRepo") or "",
            "stars": paper.get("githubStars") or item.get("githubStars") or 0,
        })
    return records


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Get trending Hugging Face papers sorted by upvotes. Optional YYYY-MM-DD date and title/summary keyword filter. Return JSON id, url, published, title, summary, upvotes, github, stars; NO RESULTS or ERROR."""
    try:
        params = {"limit": max(1, min(100, limit))}
        if date:
            params["date"] = date
        items = with_retry(lambda: _request("GET", HF_DAILY_URL, params=params)).json()
        records = _hf_records(items, keyword=keyword)
        records.sort(key=lambda record: record["upvotes"], reverse=True)
        return _records(records)
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic, preferring AI summaries. Return JSON id, url, published, title, summary, upvotes, github, stars; NO RESULTS or ERROR."""
    try:
        if not query.strip():
            return "NO RESULTS"
        items = with_retry(lambda: _request("GET", HF_SEARCH_URL, params={
            "q": query, "limit": max(1, min(50, limit)),
        })).json()
        return _records(_hf_records(items, prefer_ai=True))
    except Exception as exc:
        return _error(exc)


def _exa_call(name, arguments):
    key = os.getenv("EXA_API_KEY", "")

    def fetch():
        response = _request("POST", EXA_URL,
            params={"exaApiKey": key} if key else None,
            headers={"Accept": "application/json, text/event-stream", "Content-Type": "application/json"},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": name, "arguments": arguments}})
        if response.text.lstrip().startswith("{"):
            payload = response.json()
        else:
            payload = None
            for event in re.split(r"\r?\n\r?\n", response.text):
                data = "\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith("data:"))
                if data:
                    candidate = json.loads(data)
                    if "result" in candidate or "error" in candidate:
                        payload = candidate
                        break
            if payload is None:
                raise ValueError("Exa returned no JSON-RPC result")
        result = payload.get("result") or {}
        content = "\n".join(part.get("text", "") for part in result.get("content", []) if part.get("type") == "text")
        meta = result.get("_meta") or {}
        status = json.dumps({"meta": meta, "error": payload.get("error")}).casefold()
        flagged = any(
            re.sub(r"[^a-z]", "", str(key).casefold()) in {"ratelimited", "isratelimited", "ratelimitexceeded"}
            and value not in (False, None, 0, "false")
            for key, value in meta.items()
        )
        signal = (status if "error" in payload else "") + " " + content.casefold()
        if flagged or any(marker in signal for marker in ("rate limit exceeded", "rate limit reached", "rate limited", "rate_limit_exceeded", "too many requests")):
            raise RetryableError("Exa rate limit reached")
        if "error" in payload:
            raise ValueError(str(payload["error"]))
        if result.get("isError"):
            raise ValueError(content or "Exa tool failed")
        return content.strip() or "NO RESULTS"

    return with_retry(fetch, attempts=7, base=2.0, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web with Exa. Describe the ideal page in objective. Return result text with URLs, NO RESULTS or ERROR."""
    try:
        if not query.strip():
            return "NO RESULTS"
        return _exa_call("web_search_exa", {"query": query,
            "objective": objective or f"Find reliable sources about {query}",
            "numResults": max(1, min(20, num_results))})
    except Exception as exc:
        return _error(exc)


@tool
def web_fetch(url: str) -> str:
    """Fetch one web page as markdown using Exa, truncated to 12000 characters. Return NO RESULTS or ERROR when unavailable."""
    try:
        return _exa_call("web_fetch_exa", {"urls": [url]})[:12000]
    except Exception as exc:
        return _error(exc)


SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        print(f"== {name}: running...", flush=True)
        started = time.monotonic()
        print(f"{fn.invoke(args)[:400]}\nFinished in {time.monotonic() - started:.1f}s\n", flush=True)

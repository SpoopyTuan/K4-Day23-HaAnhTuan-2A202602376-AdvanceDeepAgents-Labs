"""Run a research survey in a sandbox and save its report and metadata."""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"


def slugify(topic):
    return re.sub(r"[^\w]+", "-", topic.lower()).strip("-")[:60].rstrip("-") or "topic"


def build_prompt(topic):
    return (
        f"Create a complete research survey on this topic: {topic.strip()}\n"
        "Follow your research workflow: plan, delegate at least three independent "
        "research tasks in parallel, verify notes, and synthesize an English report "
        "with at least three source labels and supported citations. "
        f"Write the report to {REPORT_PATH} and sources to {SOURCES_PATH}. "
        "Run the finalizer and validator in the sandbox, spot-check claims, and "
        "confirm the final validator prints OK. Report any blocker honestly."
    )


def summarize(messages, elapsed, model_name):
    calls = Counter()
    tokens = {"input": 0, "output": 0}
    for message in messages:
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls") or []
            usage = message.get("usage_metadata") or {}
        else:
            tool_calls = getattr(message, "tool_calls", None) or []
            usage = getattr(message, "usage_metadata", None) or {}
        for call in tool_calls:
            calls[call["name"]] += 1
        tokens["input"] += usage.get("input_tokens", 0) or 0
        tokens["output"] += usage.get("output_tokens", 0) or 0
    return {
        "model": model_name,
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": calls["task"],
        "tool_calls": dict(calls),
        "tokens": tokens,
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report = files.get(REPORT_PATH)
    source_bytes = files.get(SOURCES_PATH)
    if not report or not report.strip():
        raise RuntimeError("sandbox report is missing or empty")
    if not source_bytes:
        raise RuntimeError("sandbox sources.json is missing or empty")
    try:
        report.decode("utf-8")
        sources = json.loads(source_bytes)
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("sandbox report or sources.json is invalid") from exc
    families = {"arxiv", "hf-daily", "hf-search", "web"}
    if not isinstance(sources, list) or not sources or any(
        not isinstance(source, dict)
        or type(source.get("n")) is not int
        or not isinstance(source.get("url"), str)
        or source.get("source") not in families
        for source in sources
    ):
        raise RuntimeError("sandbox sources.json must contain valid source records")

    meta = {
        "topic": topic,
        **summarize(messages, elapsed, model_name),
        "n_sources": len(sources),
        "source_families": sorted({source["source"] for source in sources}),
    }
    slug = slugify(topic)
    outputs = {
        f"{slug}.sources.json": source_bytes,
        f"{slug}.meta.json": (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        f"{slug}.md": report,
    }
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".research-", dir=reports_dir) as staging:
        staging = Path(staging)
        for name, content in outputs.items():
            (staging / name).write_bytes(content)
        for name in outputs:
            os.replace(staging / name, reports_dir / name)
    return reports_dir / f"{slug}.md"


def main(topic):
    topic = topic.strip()
    if not topic:
        print('Usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    try:
        model = make_model()
        model_name = (
            os.getenv("LAB_MODEL") or os.getenv("OPENAI_DEPLOYMENT_MODEL")
            or getattr(model, "model_name", None) or getattr(model, "model", None)
            or type(model).__name__
        )
        started = time.monotonic()
        print("Opening sandbox...", flush=True)
        with open_sandbox() as backend:
            setup = backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            if setup.exit_code != 0:
                raise RuntimeError(f"cannot prepare sandbox: {setup.output}")
            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            })
            agent = build_lead_agent(backend, model)
            print(f"Researching: {topic}", flush=True)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                config={"recursion_limit": 1000},
            )
            validation = backend.execute(f"python3 {VALIDATOR_PATH} {REPORT_PATH} {SOURCES_PATH}")
            if validation.exit_code != 0 or not validation.output.strip().startswith("OK:"):
                raise RuntimeError(f"sandbox citation validation failed: {validation.output}")
            print("Validation OK; downloading outputs...", flush=True)
            report_path = save_outputs(
                backend, topic, result["messages"], time.monotonic() - started, model_name,
            )
        print(f"Saved report: {report_path}", flush=True)
        return 0
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))

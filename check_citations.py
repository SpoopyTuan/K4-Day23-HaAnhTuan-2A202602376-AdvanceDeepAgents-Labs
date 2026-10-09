"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    if not sources:
        return ["no sources in sources.json"]
    if not isinstance(sources, list):
        return ["sources.json must contain a list"]

    problems = []
    by_number = {}
    seen_urls = set()
    for index, source in enumerate(sources, 1):
        if not isinstance(source, dict):
            problems.append(f"source entry {index} must be an object")
            continue
        n, url = source.get("n"), source.get("url")
        if type(n) is not int:
            problems.append(f"source entry {index}: n must be an integer")
        elif n in by_number:
            problems.append(f"duplicate source number [{n}]")
        else:
            by_number[n] = url
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source entry {index}: invalid URL")
        elif url in seen_urls:
            problems.append(f"duplicate source URL: {url}")
        else:
            seen_urls.add(url)

    text = re.sub(r"(?ms)^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?^ {0,3}\1[^\n]*(?:\n|$)", "", report_text)
    text = re.sub(r"`+[^`\n]*`+", "", text)
    heading = re.search(r"(?m)^## References[ \t]*\r?$", text)
    if heading is None:
        problems.append("missing ## References heading")
        body, references = text, ""
    else:
        body, references = text[:heading.start()], text[heading.end():]

    cited = set()
    for match in re.finditer(r"\[(\d+(?:\s*(?:,|-)\s*\d+)*)\](?!\s*\()", body):
        for part in match.group(1).split(","):
            bounds = part.split("-")
            start = int(bounds[0])
            if len(bounds) == 1:
                cited.add(start)
            else:
                end = int(bounds[1])
                if end < start:
                    problems.append(f"invalid citation range [{part}]")
                else:
                    cited.update(n for n in by_number if start <= n <= end)
                    if sum(start <= n <= end for n in by_number) != end - start + 1:
                        problems.append(f"citation range [{part}] includes numbers missing from sources.json")

    for n in sorted(cited - by_number.keys()):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(by_number.keys() - cited):
        problems.append(f"source [{n}] never cited")

    reference_numbers = set()
    for match in re.finditer(r"(?m)^\[(\d+)\][ \t]*(.*)$", references):
        n = int(match.group(1))
        if n in reference_numbers:
            problems.append(f"duplicate reference [{n}]")
        reference_numbers.add(n)
        if n not in by_number:
            problems.append(f"reference [{n}] missing from sources.json")
        urls = re.findall(r"https?://[^\s<>\[\]\"']+", match.group(2))
        if len(urls) != 1:
            problems.append(f"reference [{n}] must contain exactly one URL")
        elif n in by_number and urls[0] != by_number[n]:
            problems.append(f"reference [{n}] URL does not match sources.json")
    for n in sorted(by_number.keys() - reference_numbers):
        problems.append(f"source [{n}] missing from References")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

"""Join pseudo web observations back to every aligned HumanTrajs memory row."""
from __future__ import annotations
import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def linearize(row: dict, pseudo: dict) -> str:
    lines = [f"URL: {(row.get('observation') or {}).get('url', '')}"]
    if pseudo.get("page_title"):
        lines.append(f"Title: {pseudo['page_title']}")
    if pseudo.get("page_summary"):
        lines.append(f"Page summary: {pseudo['page_summary']}")
    if pseudo.get("visible_text"):
        lines.append("Visible text:")
        lines.extend(f"- {item}" for item in pseudo["visible_text"])
    if pseudo.get("interactive_elements"):
        lines.append("Visible interactive elements:")
        for element in pseudo["interactive_elements"]:
            suffix = " ".join(f"{key}={element[key]!r}" for key in ("value", "state") if element.get(key))
            lines.append(f"- {element.get('role', 'other')} {element.get('name', '')!r}" +
                         (f" [{suffix}]" if suffix else ""))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    cache = {}
    for path in args.observations:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    cache[item["image_sha256"]] = item
    rows, missing = [], 0
    hash_splits = defaultdict(set)
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            hash_splits[row["image_sha256"]].add(row["split"])
            cached = cache.get(row["image_sha256"])
            if not cached:
                missing += 1
                continue
            row["pseudo_observation"] = cached["pseudo_observation"]
            row["text_observation"] = linearize(row, cached["pseudo_observation"])
            row["observation_protocol"] = "vl-pseudo-web-observation-v1"
            rows.append(row)
    overlaps = {key for key, splits in hash_splits.items() if len(splits) > 1}
    for row in rows:
        row["cross_split_image_overlap"] = row["image_sha256"] in overlaps
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    pseudos = [item["pseudo_observation"] for item in cache.values()]
    text_lengths = sorted(len(row["text_observation"]) for row in rows)
    percentile = lambda fraction: text_lengths[min(int((len(text_lengths) - 1) * fraction),
                                                   len(text_lengths) - 1)] if text_lengths else 0
    report = {"cache_entries": len(cache), "source_rows": len(rows) + missing,
              "materialized_rows": len(rows), "missing_rows": missing,
              "cross_split_image_hashes": len(overlaps),
              "rows_by_split": dict(Counter(row["split"] for row in rows)),
              "observation_quality": {
                  "empty_page_title": sum(not item.get("page_title") for item in pseudos),
                  "empty_visible_text": sum(not item.get("visible_text") for item in pseudos),
                  "empty_interactive_elements": sum(not item.get("interactive_elements")
                                                     for item in pseudos),
                  "visible_text_items_median": statistics.median(
                      [len(item.get("visible_text", [])) for item in pseudos]) if pseudos else 0,
                  "interactive_elements_median": statistics.median(
                      [len(item.get("interactive_elements", [])) for item in pseudos]) if pseudos else 0,
                  "text_observation_chars_p50": percentile(0.50),
                  "text_observation_chars_p95": percentile(0.95),
                  "text_observation_chars_max": text_lengths[-1] if text_lengths else 0,
              },
              "protocol": "vl-pseudo-web-observation-v1"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Batch evaluation -> results/eval_results.csv (use these numbers in the report).

    python scripts/ingest.py data/sample_docs
    python scripts/evaluate.py                       # uses data/eval/questions.json

Metrics per question:
  * retrieval_hit  - did the expected page appear among the retrieved sources?
  * keyword_recall - fraction of expected keywords found in the answer text
  * latency_s      - end-to-end time
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.factory import build_pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "data/eval/questions.json"))
    ap.add_argument("--out", default=str(ROOT / "results/eval_results.csv"))
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    pipeline = build_pipeline()

    rows = []
    for q in questions:
        ans = pipeline.ask(q["question"])
        pages = {(s.doc_name, s.page) for s in ans.sources}
        want = q.get("expected_page")
        hit = ""
        if want is not None:
            hit = int((q.get("doc"), want) in pages) if q.get("doc") else int(any(p == want for _, p in pages))
        kws = [k.lower() for k in q.get("expected_keywords", [])]
        recall = sum(k in ans.text.lower() for k in kws) / len(kws) if kws else ""
        rows.append(
            {
                "question": q["question"],
                "answer": ans.text.replace("\n", " "),
                "sources": "; ".join(f"{s.doc_name} p.{s.page}" for s in ans.sources),
                "retrieval_hit": hit,
                "keyword_recall": round(recall, 2) if recall != "" else "",
                "latency_s": round(ans.latency_s, 2),
                "mode": ans.mode,
            }
        )
        print(f"- {q['question'][:60]:60s} hit={hit} recall={rows[-1]['keyword_recall']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    hits = [r["retrieval_hit"] for r in rows if r["retrieval_hit"] != ""]
    recs = [r["keyword_recall"] for r in rows if r["keyword_recall"] != ""]
    if hits:
        print(f"\nRetrieval hit-rate : {sum(hits) / len(hits):.0%}")
    if recs:
        print(f"Mean keyword recall: {sum(recs) / len(recs):.0%}")
    print(f"Mean latency       : {sum(r['latency_s'] for r in rows) / len(rows):.2f}s")
    print(f"Saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Ask a question from the terminal.

    python scripts/ask.py "What reduces hallucination in RAG?"
    python scripts/ask.py --top-k 3 "What is chunk overlap?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.factory import build_pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    pipeline = build_pipeline()
    ans = pipeline.ask(args.question, top_k=args.top_k)
    print(f"\n{ans.text}\n")
    for s in ans.sources:
        mark = "*" if s.cited else " "
        print(f" {mark}[{s.index}] {s.doc_name} - page {s.page} (score {s.score})")
    print(f"\n({ans.mode} mode, {ans.latency_s:.2f}s; * = cited in the answer)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

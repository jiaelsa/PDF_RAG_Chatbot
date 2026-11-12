"""RAGPipeline: question -> retrieve -> prompt -> LLM -> answer with sources."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .models import RetrievedChunk
from .prompts import NOT_FOUND_MESSAGE, build_messages
from .retriever import Retriever


class ChatModel(Protocol):
    configured: bool

    def generate(self, messages: list[dict]) -> str: ...


@dataclass(frozen=True)
class Source:
    index: int  # the [n] used in the answer
    doc_name: str
    page: int
    score: float
    snippet: str
    cited: bool  # did the model actually cite this passage?


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    grounded: bool = True  # False when nothing relevant was found
    latency_s: float = 0.0
    mode: str = "llm"  # "llm" | "extractive" (no API key configured)


def _cited_numbers(text: str) -> set[int]:
    nums: set[int] = set()
    for group in re.findall(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]", text):
        nums.update(int(n) for n in re.findall(r"\d+", group))
    return nums


def _snippet(text: str, limit: int = 350) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: ChatModel):
        self.retriever = retriever
        self.llm = llm

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> Answer:
        start = time.perf_counter()
        hits = self.retriever.retrieve(question, top_k=top_k, doc_ids=doc_ids)

        if not hits:
            return Answer(NOT_FOUND_MESSAGE, [], grounded=False, latency_s=time.perf_counter() - start)

        if self.llm.configured:
            text = self.llm.generate(build_messages(question, hits))
            mode = "llm"
        else:
            text = self._extractive_answer(hits)
            mode = "extractive"

        cited = _cited_numbers(text)
        grounded = NOT_FOUND_MESSAGE.lower() not in text.lower()
        sources = [
            Source(
                index=i,
                doc_name=h.chunk.doc_name,
                page=h.chunk.page,
                score=round(h.score, 3),
                snippet=_snippet(h.chunk.text),
                cited=(i in cited) if cited else False,
            )
            for i, h in enumerate(hits, start=1)
        ]
        return Answer(text, sources, grounded, time.perf_counter() - start, mode)

    @staticmethod
    def _extractive_answer(hits: list[RetrievedChunk]) -> str:
        """Fallback when no LLM key is set: show the best passages verbatim."""
        top = hits[0]
        return (
            "*(No LLM configured: showing the most relevant passage instead of a generated answer.)*\n\n"
            f"{_snippet(top.chunk.text, 600)} [1]"
        )

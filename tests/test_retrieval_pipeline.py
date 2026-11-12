from rag.ingestion import ingest_pdf_bytes
from rag.llm import LLMClient, LLMSettings
from rag.pipeline import RAGPipeline, _cited_numbers
from rag.prompts import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_messages, format_context
from rag.retriever import Retriever
from tests.helpers import SAMPLE_PAGES, make_pdf


class FakeLLM:
    configured = True

    def __init__(self, reply="RAG was introduced by Lewis et al. in 2020 [1]."):
        self.reply, self.seen = reply, None

    def generate(self, messages):
        self.seen = messages
        return self.reply


def _setup(store, embedder, llm=None, min_score=0.1):
    ingest_pdf_bytes(make_pdf(SAMPLE_PAGES), "rag_intro.pdf", store, embedder, chunk_size=400, chunk_overlap=80)
    retriever = Retriever(store, embedder, top_k=3, min_score=min_score)
    return RAGPipeline(retriever, llm or FakeLLM()), retriever


def test_retriever_ranks_and_filters(store, embedder):
    _, retriever = _setup(store, embedder)
    hits = retriever.retrieve("Which vector databases use HNSW approximate nearest neighbour search?")
    assert hits[0].chunk.page == 2
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert retriever.retrieve("   ") == []
    strict = Retriever(store, embedder, min_score=0.99)
    assert strict.retrieve("zebra quantum xylophone") == []


def test_prompt_contains_sources_and_guardrails(store, embedder):
    _, retriever = _setup(store, embedder)
    hits = retriever.retrieve("hallucination")
    ctx = format_context(hits)
    assert "[1] (source: rag_intro.pdf, page" in ctx
    msgs = build_messages("What is hallucination?", hits)
    assert msgs[0]["content"] == SYSTEM_PROMPT and "ONLY" in SYSTEM_PROMPT
    assert "What is hallucination?" in msgs[1]["content"]


def test_answer_with_sources_and_citation_flags(store, embedder):
    llm = FakeLLM("RAG was introduced by Lewis et al. in 2020 [1].")
    pipe, _ = _setup(store, embedder, llm)
    ans = pipe.ask("Who introduced RAG?")
    assert ans.grounded and ans.mode == "llm"
    assert ans.sources[0].doc_name == "rag_intro.pdf" and ans.sources[0].page >= 1
    assert ans.sources[0].cited and not any(s.cited for s in ans.sources[1:])
    assert llm.seen is not None


def test_no_relevant_context_skips_llm(store, embedder):
    llm = FakeLLM()
    pipe, _ = _setup(store, embedder, llm, min_score=0.99)
    ans = pipe.ask("What is the airspeed velocity of an unladen swallow?")
    assert ans.text == NOT_FOUND_MESSAGE and not ans.grounded and ans.sources == []
    assert llm.seen is None  # the LLM was never called


def test_doc_filter(store, embedder):
    pipe, _ = _setup(store, embedder)
    r = ingest_pdf_bytes(make_pdf(["Cooking pasta requires boiling water and salt."]), "cook.pdf", store, embedder)
    ans = pipe.ask("boiling water pasta", doc_ids=[r.doc_id])
    assert {s.doc_name for s in ans.sources} == {"cook.pdf"}


def test_extractive_fallback_without_llm(store, embedder):
    class NoLLM:
        configured = False

        def generate(self, m):
            raise AssertionError("must not be called")

    pipe, _ = _setup(store, embedder, NoLLM())
    ans = pipe.ask("What is cosine similarity?")
    assert ans.mode == "extractive" and ans.sources


def test_citation_parser():
    assert _cited_numbers("see [1], [2][3] and [4, 5] but not [x]") == {1, 2, 3, 4, 5}


def test_llm_settings_configured_logic():
    assert not LLMSettings("", "https://api.groq.com/openai/v1", "m").configured
    assert LLMSettings("key", "https://api.groq.com/openai/v1", "m").configured
    assert LLMSettings("", "http://localhost:11434/v1", "llama3").configured  # Ollama needs no key
    assert not LLMClient(LLMSettings("", "https://x", "m")).configured

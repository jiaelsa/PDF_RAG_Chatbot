"""Test helpers shared by the whole test-suite (no network, no heavy models)."""
from __future__ import annotations

import hashlib
import re
import textwrap


def make_pdf(pages: list[str], wrap: int = 90, header: str | None = None) -> bytes:
    """Build a small but valid text PDF. ``pages`` is a list of page texts;
    blank lines separate paragraphs. ``header`` is printed on every page."""

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects: list[bytes] = []  # object i+1
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for i, text in enumerate(pages):
        lines: list[str] = []
        if header:
            lines += [header, ""]
        for para in text.split("\n"):
            lines += textwrap.wrap(para, wrap) or [""]
        ops = ["BT", "/F1 11 Tf", "14 TL", "50 780 Td"]
        for line in lines:
            ops.append(f"({esc(line)}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1", "replace")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_ids[i] + 1} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for idx, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{idx} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder. Texts that share words get a
    high cosine similarity, which is enough to test retrieval logic offline."""

    DIM = 256

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


SAMPLE_PAGES = [
    "Retrieval-Augmented Generation (RAG) is a technique that combines a retriever with a "
    "language model. Instead of relying only on knowledge stored in model weights, the system "
    "looks up relevant passages in an external knowledge base and gives them to the model as "
    "context. RAG was introduced by Lewis et al. in 2020.\n"
    "The main advantage of RAG is that knowledge can be updated without retraining. "
    "Adding a new document only requires embedding it and inserting it in the index.",
    "Vector embeddings map text to points in a high-dimensional space so that semantically "
    "similar passages are close together. Sentence-Transformers models such as all-MiniLM-L6-v2 "
    "produce 384-dimensional embeddings. Similarity is usually measured with cosine similarity.\n"
    "Vector databases like ChromaDB, FAISS and Pinecone use approximate nearest neighbour "
    "search (for example HNSW graphs) to find the closest chunks in milliseconds, even with "
    "hundreds of thousands of vectors.",
    "Hallucination happens when a language model produces fluent but unsupported statements. "
    "RAG reduces hallucination by restricting the answer to retrieved evidence and by returning "
    "the document name and page number of every source, so that users can verify the claim.\n"
    "Chunk size matters: chunks that are too small lose context, chunks that are too large "
    "dilute the embedding. A typical choice is 800 to 1000 characters with 10 to 20 percent overlap.",
]

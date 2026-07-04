"""
rag_retriever.py — embedding-based retrieval over care_intelligence_faq.md

Separated from the flow/orchestration code so it can be swapped or reused
(e.g. by the standalone care_intelligence_rag.py script) without touching
agent logic.
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import List

from sentence_transformers import SentenceTransformer
import faiss

from faq_loader import load_faq, FAQItem

FAQ_PATH = os.path.join(os.path.dirname(__file__), "care_intelligence_faq.md")


@dataclass
class RetrievedChunk:
    id: int
    question: str
    answer: str
    score: float


class FAQRetriever:
    def __init__(self, faq_path: str = FAQ_PATH, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.items: List[FAQItem] = load_faq(faq_path)
        texts = [f"Q: {i.q}\nA: {i.a}" for i in self.items]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(np.array(embeddings, dtype="float32"))

    def search(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        query_vec = self.model.encode([query], normalize_embeddings=True)
        # inner product on normalized vectors == cosine similarity, range [-1, 1]
        scores, indices = self.index.search(np.array(query_vec, dtype="float32"), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = self.items[idx]
            # rescale cosine [-1,1] to a 0-2 "relevance score" so thresholds
            # in agent_core.py (RELEVANCE_THRESHOLD) are easy to reason about
            rescaled = float(score) + 1.0
            results.append(RetrievedChunk(id=item.id, question=item.q, answer=item.a, score=rescaled))
        return results

    def format_context(self, chunks: List[RetrievedChunk]) -> str:
        return "\n\n".join(f"[FAQ #{c.id}] Q: {c.question}\nA: {c.answer}" for c in chunks)

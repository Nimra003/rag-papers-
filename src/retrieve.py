import json
import re
import sys
import numpy as np
import faiss
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/chunks.json")
INDEX_FILE = Path("data/index.faiss")
MODEL_NAME = "all-MiniLM-L6-v2"

_model = SentenceTransformer(MODEL_NAME)
_index = faiss.read_index(str(INDEX_FILE))
_chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:[._][a-z0-9]+)*", text.lower())


_bm25 = BM25Okapi([tokenize(c["text"]) for c in _chunks])


def _embed(question):
    q = _model.encode([question], normalize_embeddings=True)
    return np.asarray(q, dtype="float32")


def retrieve(question, k=4):
    q = _embed(question)
    scores, ids = _index.search(q, k)
    return [{**_chunks[int(i)], "score": float(s)}
            for s, i in zip(scores[0], ids[0])]


def retrieve_hybrid(question, k=4, pool=20, rrf_k=60):
    """Merge embedding search and BM25 keyword search (reciprocal rank fusion)."""
    q = _embed(question)
    _, dense_ids = _index.search(q, pool)
    bm25_scores = _bm25.get_scores(tokenize(question))
    bm25_ids = np.argsort(bm25_scores)[::-1][:pool]
    fused = {}
    for ranking in (dense_ids[0], bm25_ids):
        for rank, i in enumerate(ranking):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (rrf_k + rank + 1)
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    return [{**_chunks[i],
             "score": float(np.dot(q[0], _index.reconstruct(i))),  # cosine similarity
             "rrf": fused[i]} for i in top]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Which features predict heart disease?"
    for r in retrieve(question):
        print(f"[{r['score']:.3f}] {r['paper']} p.{r['page']}")
        print(r["text"][:300].replace("\n", " "), "...\n")
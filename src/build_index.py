import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/chunks.json")
INDEX_FILE = Path("data/index.faiss")
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in chunks]
    vecs = model.encode(texts, normalize_embeddings=True,
                        show_progress_bar=True, batch_size=32)
    vecs = np.asarray(vecs, dtype="float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(INDEX_FILE))
    print(f"Indexed {index.ntotal} chunks")


if __name__ == "__main__":
    main()
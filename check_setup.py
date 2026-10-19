import faiss, numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # downloads ~90 MB on first run
texts = ["heart disease", "cardiac illness", "pizza"]
vecs = model.encode(texts, normalize_embeddings=True)
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(np.asarray(vecs, dtype="float32"))
scores, ids = index.search(vecs[:1], 3)
print(ids, scores)
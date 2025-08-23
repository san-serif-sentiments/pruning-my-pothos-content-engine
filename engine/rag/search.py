import os
import chromadb
from sentence_transformers import SentenceTransformer

EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-small-en-v1.5")
DB_DIR = os.getenv("DB_DIR", "engine/.chroma")

_client = chromadb.PersistentClient(path=DB_DIR)
_col = _client.get_or_create_collection("content")
_embedder = SentenceTransformer(EMB_MODEL)

def retrieve(query: str, k: int = 6):
    qv = _embedder.encode([query]).tolist()[0]
    res = _col.query(query_embeddings=[qv], n_results=k)
    hits = []
    if not res.get("ids") or not res["ids"][0]:
        return "", []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "source": res["metadatas"][0][i]["source"],
        })
    ctx = "\n\n".join([h["text"] for h in hits])
    sources = list(dict.fromkeys([h["source"] for h in hits]))
    return ctx, sources

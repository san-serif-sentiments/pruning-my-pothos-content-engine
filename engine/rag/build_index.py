import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from markdown import markdown
from bs4 import BeautifulSoup

EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-small-en-v1.5")
DB_DIR = os.getenv("DB_DIR", "engine/.chroma")
CONTENT_DIR = os.getenv("CONTENT_DIR", "content")

def md_to_text(md: str) -> str:
    html = markdown(md)
    return BeautifulSoup(html, "html.parser").get_text(" ")

def chunks(s: str, n: int = 900):
    return [s[i:i+n] for i in range(0, len(s), n)]

def main():
    os.makedirs(DB_DIR, exist_ok=True)
    embedder = SentenceTransformer(EMB_MODEL)
    client = chromadb.PersistentClient(path=DB_DIR)
    col = client.get_or_create_collection("content")

    ids, docs, metas = [], [], []
    for fp in glob.glob(f"{CONTENT_DIR}/**/*.md", recursive=True):
        with open(fp, "r", encoding="utf-8") as f:
            text = md_to_text(f.read())
        for i, ch in enumerate(chunks(text)):
            ids.append(f"{fp}:{i}")
            docs.append(ch)
            metas.append({"source": fp})

    if not docs:
        print("No documents found in content/. Add Markdown and re-run.")
        return

    embeds = embedder.encode(docs).tolist()
    col.upsert(ids=ids, documents=docs, embeddings=embeds, metadatas=metas)
    print(f"Indexed {len(docs)} chunks from {len(set([m['source'] for m in metas]))} files.")

if __name__ == "__main__":
    main()

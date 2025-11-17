#!/usr/bin/env python3
"""
semantic_search_tfidf.py
Lightweight TF-IDF based semantic search over a docs repo.

Usage:
  python semantic_search.py --query "How do I fetch tweets with expansions?" --topk 5
"""
import os, argparse, json
from pathlib import Path
from typing import List, Dict
from git import Repo
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_URL = "https://github.com/xdevplatform/postman-twitter-api"
REPO_DIR = Path("postman-twitter-api")
INDEX_DIR = Path("index_data")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 200

def ensure_repo_cloned():
    if REPO_DIR.exists():
        print(f"[info] repo present at {REPO_DIR}")
        return
    print("[info] cloning repo...")
    Repo.clone_from(REPO_URL, str(REPO_DIR))
    print("[info] clone complete.")

def read_text_files_from_repo(repo_dir: Path) -> Dict[str,str]:
    docs = {}
    for root, _, files in os.walk(repo_dir):
        for fname in files:
            path = Path(root) / fname
            if fname.lower().endswith((".md", ".txt", ".html", ".htm")):
                try:
                    docs[str(path)] = path.read_text(encoding="utf-8", errors="ignore")
                except:
                    pass
            elif fname.lower().endswith(".json"):
                try:
                    raw = path.read_text(encoding="utf-8", errors="ignore")
                    # try parse json and extract text
                    import json as _json
                    j = _json.loads(raw)
                    parts = []
                    def walk(x):
                        if x is None: return
                        if isinstance(x, str):
                            if len(x.strip())>2: parts.append(x)
                        elif isinstance(x, dict):
                            for v in x.values(): walk(v)
                        elif isinstance(x, list):
                            for it in x: walk(it)
                    walk(j)
                    if parts:
                        docs[str(path)] = "\n\n".join(parts)
                    else:
                        docs[str(path)] = raw
                except:
                    docs[str(path)] = path.read_text(encoding="utf-8", errors="ignore")
    return docs

def smart_chunk(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks=[]
    start=0
    L=len(text)
    while start < L:
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        if end >= L: break
        start = end - overlap
    return chunks

def build_index(docs: Dict[str,str]):
    INDEX_DIR.mkdir(exist_ok=True)
    metadata=[]
    texts=[]
    for src, txt in docs.items():
        chs = smart_chunk(txt)
        for c in chs:
            metadata.append({"source": src, "text": c})
            texts.append(c)
    if not texts:
        raise ValueError("No text found in repo.")
    print(f"[info] creating TF-IDF matrix for {len(texts)} chunks...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    X = vectorizer.fit_transform(texts)
    # save vectorizer & metadata
    import pickle
    with open(INDEX_DIR/"vectorizer.pkl","wb") as f:
        pickle.dump(vectorizer,f)
    with open(INDEX_DIR/"metadata.pkl","wb") as f:
        pickle.dump(metadata,f)
    # store dense matrix for fast dot (optional); we will keep sparse X
    from scipy import sparse
    sparse.save_npz(INDEX_DIR/"tfidf.npz", X)
    return vectorizer, X, metadata

def load_index():
    import pickle
    from scipy import sparse
    if not (INDEX_DIR/"vectorizer.pkl").exists():
        return None, None, None
    with open(INDEX_DIR/"vectorizer.pkl","rb") as f:
        vectorizer = pickle.load(f)
    with open(INDEX_DIR/"metadata.pkl","rb") as f:
        metadata = pickle.load(f)
    X = sparse.load_npz(INDEX_DIR/"tfidf.npz")
    return vectorizer, X, metadata

def query_index(query: str, vectorizer, X, metadata, topk=5):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(X, qv).reshape(-1)
    idx = np.argpartition(-sims, range(min(topk, len(sims))))[:topk]
    idx_sorted = idx[np.argsort(-sims[idx])]
    results=[]
    for rank,i in enumerate(idx_sorted, start=1):
        results.append({
            "rank": rank,
            "score": float(sims[i]),
            "source": metadata[i]["source"],
            "text": metadata[i]["text"]
        })
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()

    ensure_repo_cloned()
    docs = read_text_files_from_repo(REPO_DIR)
    vec, X, meta = load_index()
    if vec is None or args.rebuild:
        vec, X, meta = build_index(docs)
    res = query_index(args.query, vec, X, meta, topk=args.topk)
    print(json.dumps({"query": args.query, "results": res}, indent=2, ensure_ascii=False))

if __name__=="__main__":
    main()

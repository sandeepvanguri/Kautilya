#!/usr/bin/env python3
"""
narrative_builder.py

Robust TF-IDF based narrative builder that handles a variety of JSON shapes
(list-of-dicts, list-of-lists, dict-of-lists, NDJSON converted to array, etc.)

Usage:
  python narrative_builder.py --topic "AI regulation" --json news.json
"""
import argparse
import json
from pathlib import Path
from dateutil import parser as dateparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
from tqdm import tqdm

# ---------- Utilities ----------
def load_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    txt = p.read_text(encoding="utf-8")
    txt_strip = txt.lstrip()
    # quick check: NDJSON (each line a JSON object) -> convert to list
    if "\n" in txt_strip and txt_strip.strip().startswith("{") and not txt_strip.startswith("["):
        # try to parse line-by-line into list
        lines = [line.strip() for line in txt.splitlines() if line.strip()]
        try:
            objs = [json.loads(line) for line in lines]
            return objs
        except Exception:
            # fall through to normal json.load attempt
            pass
    data = json.loads(txt)
    # normalize: if dict with "articles" key, return that list
    if isinstance(data, dict) and "articles" in data and isinstance(data["articles"], list):
        return data["articles"]
    # if dict mapping ids -> objects, return values
    if isinstance(data, dict) and all(not isinstance(v, (list, dict)) for v in data.values()):
        # probably a simple mapping; return list of values
        return list(data.values())
    return data

def detect_fields(item):
    # item is expected to be a dict
    # create map lowercase->original key
    if not isinstance(item, dict):
        return (None, None, None, None, None)
    keys = {k.lower(): k for k in item.keys()}
    date_key = keys.get("date") or keys.get("published") or keys.get("published_at") or keys.get("pubdate") or keys.get("publishedat")
    headline_key = keys.get("title") or keys.get("headline") or keys.get("heading")
    url_key = keys.get("url") or keys.get("link")
    content_key = keys.get("content") or keys.get("body") or keys.get("text") or keys.get("description")
    rating_key = keys.get("source_rating") or keys.get("rating") or keys.get("source_rating_score")
    return date_key, headline_key, url_key, content_key, rating_key

def normalize(item, date_k, head_k, url_k, cont_k, rate_k):
    # item is dict
    # parse date
    dt = None
    try:
        if date_k and item.get(date_k):
            dt = dateparser.parse(str(item.get(date_k)))
    except Exception:
        dt = None
    # headline and content
    headline = item.get(head_k) if head_k else item.get("title") or item.get("headline")
    url = item.get(url_k) if url_k else item.get("url") or item.get("link")
    content = item.get(cont_k) if cont_k else item.get("content") or item.get("body") or item.get("text")
    text = ""
    if isinstance(content, str) and content.strip():
        text = content.strip()
    elif isinstance(headline, str):
        text = headline.strip()
    # rating
    rating = None
    try:
        if rate_k and item.get(rate_k) is not None:
            rating = float(item.get(rate_k))
    except Exception:
        rating = None
    return {
        "date": dt,
        "headline": headline or "",
        "url": url or "",
        "text": text or "",
        "rating": rating,
        "raw": item
    }

# ---------- TF-IDF relevance + selection ----------
def build_tfidf(articles):
    texts = [a["text"] for a in articles]
    if len(texts) == 0:
        return None, None
    vectorizer = TfidfVectorizer(stop_words="english", max_features=30000)
    X = vectorizer.fit_transform(texts)
    return vectorizer, X

def select_relevant(topic, vectorizer, X, articles, topk=300):
    if vectorizer is None or X is None:
        return []
    q = vectorizer.transform([topic])
    sims = cosine_similarity(X, q).reshape(-1)
    scored = [(float(sims[i]), articles[i]) for i in range(len(articles))]
    scored.sort(key=lambda x: -x[0])
    # filter out near-zero scores; keep topk
    filtered = [s for s in scored if s[0] > 0.0]
    return filtered[:topk]

# ---------- Timeline ----------
def build_timeline(selected):
    timeline = []
    for score, art in selected:
        timeline.append({
            "date": art["date"].isoformat() if art["date"] else None,
            "headline": art["headline"],
            "url": art["url"],
            "score": float(score),
            "why_it_matters": explain_why(art, score)
        })
    # sort by date where available, fallback to original order
    timeline.sort(key=lambda x: x["date"] or "")
    return timeline

def explain_why(art, score):
    text = (art.get("text") or "").lower()
    markers = []
    if any(w in text for w in ("election", "vote", "ballot")):
        markers.append("electoral implications")
    if any(w in text for w in ("conflict", "attack", "ceasefire", "strike")):
        markers.append("security/conflict developments")
    if any(w in text for w in ("policy", "regulation", "law", "bill")):
        markers.append("policy/regulatory implications")
    if any(w in text for w in ("market", "stock", "econom")):
        markers.append("economic/market impact")
    if markers:
        return f"Relevant (score={score:.3f}). Focus: {', '.join(markers)}"
    # fallback: short excerpt
    excerpt = (art.get("text") or art.get("headline") or "")[:180]
    return f"Relevant (score={score:.3f}). Excerpt: {excerpt}"

# ---------- Clustering ----------
def cluster(selected, n_clusters=None):
    if not selected:
        return []
    texts = [a["text"] for _, a in selected]
    if len(texts) == 0:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    X = vectorizer.fit_transform(texts)
    N = len(texts)
    if n_clusters is None:
        n_clusters = max(2, int(round(np.sqrt(N))))
    if N <= n_clusters:
        labels = np.arange(N)
    else:
        try:
            km = KMeans(n_clusters=n_clusters, random_state=42)
            labels = km.fit_predict(X)
        except Exception:
            # fallback: each item its own cluster
            labels = np.arange(N)
    clusters = {}
    for idx, lbl in enumerate(labels):
        clusters.setdefault(int(lbl), []).append({
            "index": idx,
            "score": float(selected[idx][0]),
            "headline": selected[idx][1]["headline"],
            "url": selected[idx][1]["url"],
            "snippet": (selected[idx][1]["text"] or "")[:300]
        })
    cluster_list = []
    for k, members in clusters.items():
        members_sorted = sorted(members, key=lambda x: -x["score"])
        cluster_list.append({
            "cluster_id": int(k),
            "size": len(members_sorted),
            "representative": members_sorted[0],
            "members": members_sorted
        })
    cluster_list.sort(key=lambda x: -x["size"])
    return cluster_list

# ---------- Graph ----------
def build_graph(selected):
    n = len(selected)
    if n == 0:
        return {"nodes": [], "edges": []}
    texts = [a["text"] for _, a in selected]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    X = vectorizer.fit_transform(texts).toarray()
    sims = cosine_similarity(X)
    G = nx.DiGraph()
    for i, (_, art) in enumerate(selected):
        G.add_node(i, headline=art["headline"], url=art["url"], date=(art["date"].isoformat() if art["date"] else None))
    # edge threshold
    EDGE_THRESH = 0.30
    for i in range(n):
        for j in range(n):
            if i == j: continue
            s = float(sims[i, j])
            if s >= EDGE_THRESH:
                # simple temporal heuristic
                di = selected[i][1].get("date")
                dj = selected[j][1].get("date")
                relation = "related"
                if di and dj:
                    if di < dj:
                        relation = "builds_on"
                    elif di > dj:
                        relation = "adds_context"
                G.add_edge(i, j, score=s, relation=relation)
    nodes = [{"id": int(n), "headline": G.nodes[n].get("headline"), "url": G.nodes[n].get("url"), "date": G.nodes[n].get("date")} for n in G.nodes()]
    edges = [{"source": int(u), "target": int(v), "score": float(d.get("score")), "relation": d.get("relation")} for u, v, d in G.edges(data=True)]
    return {"nodes": nodes, "edges": edges}

# ---------- Summary synthesis ----------
def synthesize_summary(selected, clusters):
    if not selected:
        return "No relevant articles found for this topic after filtering."
    sents = []
    top_clusters = clusters[:min(4, len(clusters))]
    for c in top_clusters:
        rep = c["representative"]
        head = rep.get("headline") or rep.get("snippet", "")
        sents.append(f"One major thread concerns: \"{head}\" (representative of {c['size']} article(s)).")
    sents.append("Chronologically, coverage moves from initial reports to follow-ups that add context, statements, and policy reactions.")
    sents.append("Together these pieces highlight the interplay between events, official responses, and broader implications.")
    if len(sents) < 5:
        sents.append("The coverage suggests sustained attention and notable implications for stakeholders.")
    return " ".join(sents[:10])

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="Topic string to build narrative about")
    parser.add_argument("--json", required=True, help="Path to JSON news file")
    parser.add_argument("--min_rating", type=float, default=8.0, help="Minimum source_rating to include (if present)")
    parser.add_argument("--topk", type=int, default=300, help="Max number of relevant articles to keep")
    args = parser.parse_args()

    try:
        raw = load_json(args.json)
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e)}))
        return
    except Exception as e:
        print(json.dumps({"error": f"Failed to parse JSON file: {e}"}))
        return

    # find a dict-like sample to detect fields
    first_item = None
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                first_item = entry
                break
            if isinstance(entry, list):
                for sub in entry:
                    if isinstance(sub, dict):
                        first_item = sub
                        break
            if first_item:
                break
    elif isinstance(raw, dict):
        # try to find a dict value or a list-of-dicts
        for v in raw.values():
            if isinstance(v, dict):
                first_item = v
                break
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                first_item = v[0]
                break
    if first_item is None:
        print(json.dumps({"error": "Could not find any article objects (dicts) in the JSON file. Please provide a sample record."}))
        return

    date_k, head_k, url_k, cont_k, rate_k = detect_fields(first_item)

    # iterate records robustly and normalize
    def iter_records(collection):
        if isinstance(collection, dict):
            # try values that are lists
            for v in collection.values():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            yield item
            # as fallback yield the dict itself
            yield collection
            return
        for entry in collection:
            if isinstance(entry, dict):
                yield entry
            elif isinstance(entry, list):
                for sub in entry:
                    if isinstance(sub, dict):
                        yield sub
            # ignore other types

    normed = []
    for item in iter_records(raw):
        try:
            d = normalize(item, date_k, head_k, url_k, cont_k, rate_k)
        except Exception:
            continue
        # filter by rating if present
        if d["rating"] is not None:
            if d["rating"] <= args.min_rating:
                continue
        # discard items with no text
        if not d["text"]:
            continue
        normed.append(d)

    if len(normed) == 0:
        print(json.dumps({"error": "No articles left after filtering by source_rating and content presence"}))
        return

    # Build TF-IDF and select relevant
    vectorizer, X = build_tfidf(normed)
    selected = select_relevant(args.topic, vectorizer, X, normed, topk=args.topk)

    # Build narrative pieces
    timeline = build_timeline(selected)
    clusters = cluster(selected)
    graph = build_graph(selected)
    summary = synthesize_summary(selected, clusters)

    out = {
        "narrative_summary": summary,
        "timeline": timeline,
        "clusters": clusters,
        "graph": graph,
        "metadata": {
            "topic": args.topic,
            "source_count": len(normed),
            "selected_count": len(selected)
        }
    }
    print(json.dumps(out, indent=2, default=str, ensure_ascii=False))

if __name__ == "__main__":
    main()

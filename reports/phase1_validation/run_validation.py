#!/usr/bin/env python3
"""Echo × QuLab technical validation harness (Phase 1)."""

from __future__ import annotations

import json
import statistics
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = [
    "quantum materials",
    "oncology",
    "genomics",
    "climate modeling",
    "signal processing",
    "drug discovery",
    "electromagnetism",
    "metabolomics",
    "nanotechnology",
    "environmental engineering",
]

QUERIES = [
    "Top 2020-2025 publications on superconducting quantum materials with measured critical temperature.",
    "Clinical studies on PD-1 and CTLA-4 combination therapies in solid tumors.",
    "Methods papers comparing single-cell RNA-seq normalization strategies.",
    "Ensemble downscaling approaches for regional precipitation forecasting.",
    "Denoising autoencoders for non-stationary biomedical signal reconstruction.",
    "Structure-based virtual screening workflows validated by wet-lab IC50 assays.",
    "Finite-element approaches for high-frequency electromagnetic shielding design.",
    "Metabolite identification pipelines using tandem mass spectrometry libraries.",
    "Synthesis routes for 2D nanomaterials with tunable bandgaps.",
    "Life-cycle assessment studies of wastewater nutrient recovery systems.",
]


@dataclass
class Doc:
    id: str
    title: str
    abstract: str
    domain: str


def http_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "QuLabValidation/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def invert_inverted_index(inv: dict[str, list[int]]) -> str:
    if not inv:
        return ""
    length = max(max(pos) for pos in inv.values()) + 1
    words = [""] * length
    for word, positions in inv.items():
        for idx in positions:
            words[idx] = word
    return " ".join(words)


def fetch_openalex_snapshot(per_domain: int = 25) -> tuple[list[Doc], list[dict[str, Any]]]:
    docs: list[Doc] = []
    metadata: list[dict[str, Any]] = []

    for domain in DOMAINS:
        filt = urllib.parse.quote(f"title_and_abstract.search:{domain},from_publication_date:2020-01-01")
        url = (
            "https://api.openalex.org/works?"
            f"filter={filt}&per-page={per_domain}&sort=cited_by_count:desc"
        )
        payload = http_json(url)

        metadata.append(
            {
                "domain": domain,
                "openalex_meta_count": payload.get("meta", {}).get("count", 0),
                "sample_size": len(payload.get("results", [])),
                "query_url": url,
            }
        )

        for row in payload.get("results", []):
            abstract = invert_inverted_index(row.get("abstract_inverted_index") or {})
            docs.append(
                Doc(
                    id=row.get("id", ""),
                    title=row.get("title", ""),
                    abstract=abstract,
                    domain=domain,
                )
            )
    return docs, metadata


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in "".join(ch if ch.isalnum() else " " for ch in text).split() if tok]


def build_tfidf(docs: list[Doc]) -> tuple[list[dict[str, float]], dict[str, float]]:
    doc_tokens = [tokenize(f"{d.title} {d.abstract}") for d in docs]
    df: dict[str, int] = {}
    for tokens in doc_tokens:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    n = len(docs)
    idf = {term: (1.0 + (n / freq)) for term, freq in df.items()}

    vectors: list[dict[str, float]] = []
    for tokens in doc_tokens:
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        norm = sum(v * v for v in tf.values()) ** 0.5 or 1.0
        vec = {t: (c / norm) * idf[t] for t, c in tf.items()}
        vectors.append(vec)
    return vectors, idf


def vectorize_query(query: str, idf: dict[str, float]) -> dict[str, float]:
    tf: dict[str, int] = {}
    for t in tokenize(query):
        if t in idf:
            tf[t] = tf.get(t, 0) + 1
    norm = sum(v * v for v in tf.values()) ** 0.5 or 1.0
    return {t: (c / norm) * idf[t] for t, c in tf.items()}


def dot(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def retrieve(query: str, vectors: list[dict[str, float]], docs: list[Doc], idf: dict[str, float], k: int = 5) -> list[dict[str, Any]]:
    q = vectorize_query(query, idf)
    scored: list[tuple[float, int]] = []
    for i, vec in enumerate(vectors):
        scored.append((dot(q, vec), i))
    top = sorted(scored, reverse=True)[:k]
    return [
        {
            "score": score,
            "id": docs[i].id,
            "title": docs[i].title,
            "domain": docs[i].domain,
        }
        for score, i in top
    ]


def expected_domain_for_query(query: str) -> str:
    mapping = {
        0: "quantum materials",
        1: "oncology",
        2: "genomics",
        3: "climate modeling",
        4: "signal processing",
        5: "drug discovery",
        6: "electromagnetism",
        7: "metabolomics",
        8: "nanotechnology",
        9: "environmental engineering",
    }
    return mapping[QUERIES.index(query)]


def main() -> None:
    t0 = time.perf_counter()
    docs, metadata = fetch_openalex_snapshot(per_domain=25)
    t1 = time.perf_counter()

    vectors, idf = build_tfidf(docs)
    t2 = time.perf_counter()

    latencies_ms: list[float] = []
    runs: list[dict[str, Any]] = []
    hit_at_5 = 0

    for q in QUERIES:
        rs = []
        for _ in range(20):
            s = time.perf_counter()
            out = retrieve(q, vectors, docs, idf, k=5)
            e = time.perf_counter()
            latencies_ms.append((e - s) * 1000)
            rs = out

        exp = expected_domain_for_query(q)
        top_domains = [r["domain"] for r in rs]
        if exp in top_domains:
            hit_at_5 += 1

        runs.append(
            {
                "query": q,
                "expected_domain": exp,
                "top5": rs,
                "domain_hit@5": exp in top_domains,
            }
        )

    summary = {
        "corpus_documents": len(docs),
        "openalex_metadata": metadata,
        "timing": {
            "ingest_seconds": t1 - t0,
            "embed_seconds": t2 - t1,
            "retrieval_latency_ms": {
                "p50": statistics.median(latencies_ms),
                "p95": sorted(latencies_ms)[int(len(latencies_ms) * 0.95) - 1],
                "mean": statistics.mean(latencies_ms),
                "max": max(latencies_ms),
                "min": min(latencies_ms),
            },
        },
        "evaluation": {
            "queries": len(QUERIES),
            "hit_at_5": hit_at_5 / len(QUERIES),
            "runs": runs,
        },
    }

    out_file = OUT_DIR / "validation_results.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

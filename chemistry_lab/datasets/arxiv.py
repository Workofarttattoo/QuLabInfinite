"""
Specialized loader for fetching material property data from arXiv.
"""

from __future__ import annotations

import re
from typing import Dict, List
from urllib.parse import quote, quote_plus

import feedparser
import pandas as pd

PROPERTY_KEYWORDS = [
    "tensile", "yield", "strength", "modulus", "elastic", "elongation",
    "fracture", "hardness", "thermal conductivity", "specific heat",
    "heat capacity", "thermal expansion", "resistivity", "conductivity",
    "fatigue", "stress", "strain", "temperature",
]

NUMBER_SENTENCE = re.compile(r"[^.]*\d[^.]*\.", re.IGNORECASE | re.DOTALL)


def build_query(material: str, keywords: List[str]) -> str:
    escaped = material.replace('"', '\\"')
    keyword_query = " OR ".join(f"all:{quote(k)}" for k in keywords if k)
    if keyword_query:
        return f'all:"{escaped}" AND ({keyword_query})'
    return f'all:"{escaped}"'


def fetch_entries(query: str, limit: int) -> feedparser.FeedParserDict:
    base = "https://export.arxiv.org/api/query"
    url = f"{base}?search_query={quote_plus(query)}&start=0&max_results={limit}"
    return feedparser.parse(url)


def extract_numeric_sentences(text: str) -> List[str]:
    sentences = NUMBER_SENTENCE.findall(text)
    cleaned = [re.sub(r"\s+", " ", sentence).strip() for sentence in sentences]
    return cleaned


def filter_by_keywords(sentences: List[str], keywords: List[str]) -> List[str]:
    lowered_keywords = [k.lower() for k in keywords]
    filtered = []
    for sentence in sentences:
        lower_sentence = sentence.lower()
        if any(k in lower_sentence for k in lowered_keywords):
            filtered.append(sentence)
    return filtered


def collect_snippets(material: str, keywords: List[str], limit: int) -> List[Dict[str, str]]:
    query = build_query(material, keywords)
    feed = fetch_entries(query, limit)
    results: List[Dict[str, str]] = []

    for entry in feed.entries:
        summary = getattr(entry, "summary", "")
        numeric_sentences = extract_numeric_sentences(summary)
        if not numeric_sentences:
            snippet = " ".join(summary.strip().split()[:40])
            if not snippet:
                continue
            results.append({
                "material": material,
                "title": entry.title,
                "published": entry.published,
                "summary_snippet": snippet + ("…" if len(summary.split()) > 40 else ""),
                "link": entry.link,
                "query": feed.href if hasattr(feed, "href") else query,
            })
            continue
        candidate_sentences = filter_by_keywords(numeric_sentences, keywords or PROPERTY_KEYWORDS)
        if not candidate_sentences:
            snippet = " ".join(summary.strip().split()[:40])
            if not snippet:
                continue
            summary_text = snippet + ("…" if len(summary.split()) > 40 else "")
        else:
            summary_text = candidate_sentences[0]

        results.append({
            "material": material,
            "title": entry.title,
            "published": entry.published,
            "summary_snippet": summary_text,
            "link": entry.link,
            "query": feed.href if hasattr(feed, "href") else query,
        })
    return results


def load_arxiv_materials(material: str, keywords: List[str], limit: int = 10) -> pd.DataFrame:
    """
    Fetch material property snippets from arXiv and return as a DataFrame.
    """
    snippets = collect_snippets(material, keywords, limit)
    if not snippets and keywords:
        snippets = collect_snippets(material, [], limit)
        if snippets:
            for entry in snippets:
                entry["note"] = "fallback_without_keywords"
    return pd.DataFrame(snippets)

#!/usr/bin/env python3
"""
Complete arXiv Platform Ingestion System
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light)
Ingests from: arXiv, bioRxiv, medRxiv, chemRxiv, PsyArXiv, SocArXiv, EarthArXiv
"""
import requests, json, time, feedparser
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

PLATFORMS = {
    'arxiv': 'http://export.arxiv.org/api/query',
    'biorxiv': 'https://api.biorxiv.org/details/biorxiv',
    'medrxiv': 'https://api.biorxiv.org/details/medrxiv',
    'chemrxiv': 'https://chemrxiv.org/engage/chemrxiv/public-api/v1/items',
    'psyarxiv': 'https://api.osf.io/v2/preprints',
    'socarxiv': 'https://api.osf.io/v2/preprints',
    'eartharxiv': 'https://api.osf.io/v2/preprints'
}

QUERIES = [
    'cancer metabolism', 'metformin cancer', 'dichloroacetate',
    'Warburg effect', 'metabolic targeting', 'AMPK mTOR',
    'quantum computing', 'machine learning', 'drug discovery'
]

def ingest_arxiv(query, days=30, max_results=1000):
    """Ingest from arXiv.org - properly URL encoded"""
    params = {
        'search_query': f'all:{query}',
        'start': 0,
        'max_results': max_results,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    url = f"{PLATFORMS['arxiv']}?{urlencode(params)}"
    feed = feedparser.parse(url)
    return [{'title': e.title, 'authors': [a.name for a in e.authors], 'summary': e.summary,
             'url': e.link, 'published': e.published, 'source': 'arXiv'} for e in feed.entries]

def ingest_biorxiv(days=30):
    """Ingest from bioRxiv/medRxiv"""
    results = []
    from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')
    for platform in ['biorxiv', 'medrxiv']:
        try:
            url = f"{PLATFORMS[platform]}/{from_date}/{to_date}/0/json"
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            if 'collection' in data:
                results.extend([{**p, 'source': platform} for p in data['collection']])
        except Exception as e:
            print(f"[warn] {platform} failed: {e}")
    return results

def ingest_osf_preprints(provider, query='cancer', days=30):
    """Ingest from OSF-based preprint servers (PsyArXiv, SocArXiv, EarthArXiv)"""
    results = []
    try:
        params = {'filter[provider]': provider, 'filter[subjects]': query}
        url = f"{PLATFORMS[provider]}?{urlencode(params)}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        if 'data' in data:
            for item in data['data'][:100]:
                attrs = item.get('attributes', {})
                results.append({
                    'title': attrs.get('title', ''),
                    'summary': attrs.get('description', ''),
                    'url': attrs.get('links', {}).get('html', ''),
                    'published': attrs.get('date_published', ''),
                    'source': provider
                })
    except Exception as e:
        print(f"[warn] {provider} failed: {e}")
    return results

def ingest_chemrxiv(query='cancer', max_results=100):
    """Ingest from ChemRxiv"""
    results = []
    try:
        params = {'term': query, 'limit': max_results}
        url = f"{PLATFORMS['chemrxiv']}?{urlencode(params)}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        if 'itemHits' in data:
            for item in data['itemHits']:
                results.append({
                    'title': item.get('title', ''),
                    'authors': [a.get('name', '') for a in item.get('authors', [])],
                    'summary': item.get('abstract', ''),
                    'url': item.get('doi', ''),
                    'published': item.get('publishedDate', ''),
                    'source': 'chemRxiv'
                })
    except Exception as e:
        print(f"[warn] chemRxiv failed: {e}")
    return results

def main(days=30, historical=False):
    """
    Main ingestion function
    days: Number of days to look back (default 30)
    historical: If True, ingest last 365 days for historical backfill
    """
    if historical:
        days = 365
        print(f"[info] HISTORICAL MODE: Ingesting last {days} days")

    output = Path('/Users/noone/QuLabInfinite/data/arxiv_ingestion')
    output.mkdir(parents=True, exist_ok=True)

    all_papers = []

    # arXiv (all queries)
    print(f"\n[arXiv] Starting ingestion...")
    for query in QUERIES:
        print(f"  - {query}")
        all_papers.extend(ingest_arxiv(query, days=days, max_results=1000))
        time.sleep(3)

    # bioRxiv + medRxiv
    print(f"\n[bioRxiv/medRxiv] Starting ingestion...")
    all_papers.extend(ingest_biorxiv(days=days))

    # chemRxiv
    print(f"\n[chemRxiv] Starting ingestion...")
    all_papers.extend(ingest_chemrxiv(query='cancer metabolism', max_results=100))

    # OSF-based preprints
    print(f"\n[PsyArXiv] Starting ingestion...")
    all_papers.extend(ingest_osf_preprints('psyarxiv', query='cancer'))

    print(f"\n[SocArXiv] Starting ingestion...")
    all_papers.extend(ingest_osf_preprints('socarxiv', query='health'))

    print(f"\n[EarthArXiv] Starting ingestion...")
    all_papers.extend(ingest_osf_preprints('eartharxiv', query='environmental health'))

    # Save results
    filename = f'papers_{datetime.now():%Y%m%d}_{"historical" if historical else "daily"}.json'
    with open(output / filename, 'w') as f:
        json.dump(all_papers, f, indent=2)

    print(f"\n✅ Ingested {len(all_papers)} papers from all platforms")
    print(f"✅ Saved to: {output / filename}")

    return len(all_papers)

if __name__ == '__main__':
    import sys
    historical = '--historical' in sys.argv or '--past' in sys.argv
    main(historical=historical)

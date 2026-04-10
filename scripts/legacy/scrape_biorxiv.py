#!/usr/bin/env python3
"""
bioRxiv Research Paper Scraper
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light)

Scrapes cancer metabolism research from bioRxiv for ECH0's research database.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict
import re

class BioRxivScraper:
    """Scraper for bioRxiv preprints."""

    def __init__(self):
        self.base_url = "https://www.biorxiv.org"
        self.search_url = f"{self.base_url}/search"
        self.headers = {
            'User-Agent': 'ECH0-Cancer-Research/1.0 (echo@aios.is; Academic Research)'
        }

    def search_papers(
        self,
        query: str,
        from_date: str = None,
        to_date: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Search bioRxiv for papers matching query.

        Args:
            query: Search query (e.g., "cancer metabolism metformin DCA")
            from_date: Start date (YYYY-MM-DD), default 30 days ago
            to_date: End date (YYYY-MM-DD), default today
            limit: Maximum papers to retrieve

        Returns:
            List of paper metadata dicts
        """
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')

        params = {
            'query': query,
            'from_date': from_date,
            'to_date': to_date,
            'limit': limit,
            'sort': 'relevance-rank'
        }

        print(f"[bioRxiv] Searching for: {query}")
        print(f"[bioRxiv] Date range: {from_date} to {to_date}")

        try:
            response = requests.get(self.search_url, params=params, headers=self.headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            papers = []
            results = soup.find_all('li', class_='search-result')

            print(f"[bioRxiv] Found {len(results)} results")

            for result in results:
                paper = self._parse_search_result(result)
                if paper:
                    papers.append(paper)

                # Rate limiting (be nice to bioRxiv)
                time.sleep(0.5)

            return papers

        except Exception as e:
            print(f"[bioRxiv] Error searching: {e}")
            return []

    def _parse_search_result(self, result) -> Dict:
        """Parse a single search result."""
        try:
            # Title
            title_elem = result.find('span', class_='highwire-cite-title')
            title = title_elem.text.strip() if title_elem else "Unknown"

            # Authors
            authors_elem = result.find('span', class_='highwire-citation-authors')
            authors = authors_elem.text.strip() if authors_elem else "Unknown"

            # DOI link
            link_elem = result.find('a', class_='highwire-cite-linked-title')
            doi_link = link_elem['href'] if link_elem and 'href' in link_elem.attrs else None

            if doi_link and not doi_link.startswith('http'):
                doi_link = f"{self.base_url}{doi_link}"

            # Date
            date_elem = result.find('span', class_='highwire-cite-metadata-date')
            date = date_elem.text.strip() if date_elem else "Unknown"

            # Abstract snippet
            abstract_elem = result.find('div', class_='highwire-cite-snippet')
            abstract = abstract_elem.text.strip() if abstract_elem else ""

            # DOI
            doi_match = re.search(r'10\.\d{4,}/\d+', doi_link) if doi_link else None
            doi = doi_match.group(0) if doi_match else "Unknown"

            return {
                'title': title,
                'authors': authors,
                'doi': doi,
                'url': doi_link,
                'date': date,
                'abstract_snippet': abstract,
                'source': 'bioRxiv',
                'retrieved_at': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"[bioRxiv] Error parsing result: {e}")
            return None

    def get_full_paper(self, doi_url: str) -> Dict:
        """
        Retrieve full paper metadata from bioRxiv.

        Args:
            doi_url: Full URL to paper (e.g., https://www.biorxiv.org/content/10.1101/...)

        Returns:
            Dict with full paper metadata
        """
        try:
            print(f"[bioRxiv] Fetching paper: {doi_url}")

            response = requests.get(doi_url, headers=self.headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Title
            title_elem = soup.find('h1', class_='highwire-cite-title')
            title = title_elem.text.strip() if title_elem else "Unknown"

            # Authors
            authors = []
            author_elems = soup.find_all('span', class_='highwire-citation-author')
            for author in author_elems:
                authors.append(author.text.strip())

            # Abstract
            abstract_elem = soup.find('div', class_='section abstract')
            abstract = abstract_elem.text.strip() if abstract_elem else ""

            # Date
            date_elem = soup.find('span', class_='highwire-cite-metadata-date')
            date = date_elem.text.strip() if date_elem else "Unknown"

            # DOI
            doi_elem = soup.find('span', class_='highwire-cite-metadata-doi')
            doi = doi_elem.text.strip() if doi_elem else "Unknown"

            # Category
            category_elem = soup.find('span', class_='highwire-cite-metadata-category')
            category = category_elem.text.strip() if category_elem else "Unknown"

            # Full text sections
            sections = {}
            section_elems = soup.find_all('div', class_='section')
            for section in section_elems:
                section_title = section.find('h2')
                if section_title:
                    sections[section_title.text.strip()] = section.text.strip()

            return {
                'title': title,
                'authors': authors,
                'doi': doi,
                'url': doi_url,
                'date': date,
                'category': category,
                'abstract': abstract,
                'sections': sections,
                'source': 'bioRxiv',
                'retrieved_at': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"[bioRxiv] Error fetching paper: {e}")
            return None

    def search_cancer_metabolism(self, days_back: int = 30, limit: int = 50) -> List[Dict]:
        """
        Search for recent cancer metabolism research.

        Args:
            days_back: How many days to search back
            limit: Max papers to retrieve

        Returns:
            List of paper dicts
        """
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        queries = [
            "cancer metabolism metformin",
            "Warburg effect dichloroacetate",
            "cancer glycolysis metformin DCA",
            "metabolic targeting cancer",
            "AMPK mTOR cancer",
            "pyruvate dehydrogenase kinase cancer"
        ]

        all_papers = []
        seen_dois = set()

        for query in queries:
            papers = self.search_papers(query, from_date=from_date, limit=limit//len(queries))

            for paper in papers:
                # Deduplicate by DOI
                if paper['doi'] not in seen_dois:
                    all_papers.append(paper)
                    seen_dois.add(paper['doi'])

            # Rate limiting between queries
            time.sleep(2)

        print(f"[bioRxiv] Total unique papers: {len(all_papers)}")
        return all_papers


def main():
    """Main entry point for bioRxiv scraper."""
    print("╔════════════════════════════════════════════╗")
    print("║   bioRxiv Cancer Research Scraper         ║")
    print("║   For ECH0's Research Database             ║")
    print("╚════════════════════════════════════════════╝")
    print()

    scraper = BioRxivScraper()

    # Search for recent cancer metabolism research
    papers = scraper.search_cancer_metabolism(days_back=30, limit=50)

    # Save results
    output_file = f"/Users/noone/QuLabInfinite/data/biorxiv_papers_{datetime.now().strftime('%Y%m%d')}.json"

    with open(output_file, 'w') as f:
        json.dump(papers, f, indent=2)

    print(f"\n[bioRxiv] Saved {len(papers)} papers to: {output_file}")

    # Print summary
    print("\n=== Top 5 Recent Papers ===\n")
    for i, paper in enumerate(papers[:5], 1):
        print(f"{i}. {paper['title']}")
        print(f"   Authors: {paper['authors']}")
        print(f"   Date: {paper['date']}")
        print(f"   DOI: {paper['doi']}")
        print(f"   URL: {paper['url']}")
        print()


if __name__ == '__main__':
    main()

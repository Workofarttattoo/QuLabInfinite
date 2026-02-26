# Echo × QuLab Technical Validation Whitepaper v1.0

## 1) Scope
This document defines the Phase 1 technical validation baseline for Echo × QuLab retrieval over a scientific literature corpus.

Validation objectives:
1. Provide a reproducible corpus size estimate using OpenAlex metadata.
2. Document the parse → embed → retrieve pipeline.
3. Measure retrieval latency and report distribution statistics.
4. Execute a fixed 10-query domain evaluation harness.
5. Preserve full run logs and machine-readable outputs.
6. List known limitations and concrete mitigation paths.

## 2) Corpus Size (OpenAlex-verifiable)
Data source: OpenAlex `/works` API.

Acquisition parameters:
- 10 domains.
- Filter: `from_publication_date:2020-01-01`.
- Search scope: `title_and_abstract.search:<domain>`.
- Sampling: top 25 documents/domain (`sort=cited_by_count:desc`).

Observed corpus snapshot:
- Local parsed corpus size: **250 documents** (25 × 10).
- Domain-level OpenAlex metadata counts (global search-space cardinality):

| Domain | OpenAlex `meta.count` |
|---|---:|
| quantum materials | 78,175 |
| oncology | 148,708 |
| genomics | 313,312 |
| climate modeling | 320,053 |
| signal processing | 439,046 |
| drug discovery | 92,130 |
| electromagnetism | 225,281 |
| metabolomics | 86,428 |
| nanotechnology | 65,593 |
| environmental engineering | 71,280 |

Total OpenAlex search-space across these independent queries (not deduplicated): **1,840,006**.

## 3) Parsing Pipeline Description
Pipeline stages implemented in `run_validation.py`:

1. **Metadata pull**
   - Calls OpenAlex API for each domain.
   - Stores query URL + `meta.count` for verifiability.

2. **Document extraction**
   - Fields: `id`, `title`, `abstract_inverted_index`.
   - Reconstructs plain-text abstract from inverted index.

3. **Text normalization**
   - Lowercase.
   - Keep alphanumeric characters.
   - Whitespace tokenization.

4. **Corpus assembly**
   - One local record per work with explicit `domain` label inherited from query bucket.

## 4) Embedding Architecture
Current validation embedding stack (deterministic baseline):

- **Representation**: sparse TF-IDF vectors.
- **IDF variant**: `idf(term) = 1 + N/df(term)`.
- **Document vector**: L2-normalized term frequency × IDF.
- **Query vector**: same transform over query text.
- **Similarity metric**: dot product in sparse space.
- **Top-k**: `k=5`.

Rationale: minimal external dependency surface; deterministic; sufficient for latency characterization and harness wiring.

## 5) Retrieval Latency Benchmarks
Benchmark protocol:
- 10 fixed queries.
- 20 repeated retrieval calls/query.
- Total retrieval calls: 200.

Measured retrieval latency (milliseconds):
- **p50**: 0.437 ms
- **p95**: 0.589 ms
- **mean**: 0.460 ms
- **min**: 0.330 ms
- **max**: 0.780 ms

Ingestion and embedding wall-clock:
- Ingest (network + parse): 9.663 s
- Embedding build: 0.037 s

## 6) Structured Domain Queries (n=10)
1. Top 2020-2025 publications on superconducting quantum materials with measured critical temperature.
2. Clinical studies on PD-1 and CTLA-4 combination therapies in solid tumors.
3. Methods papers comparing single-cell RNA-seq normalization strategies.
4. Ensemble downscaling approaches for regional precipitation forecasting.
5. Denoising autoencoders for non-stationary biomedical signal reconstruction.
6. Structure-based virtual screening workflows validated by wet-lab IC50 assays.
7. Finite-element approaches for high-frequency electromagnetic shielding design.
8. Metabolite identification pipelines using tandem mass spectrometry libraries.
9. Synthesis routes for 2D nanomaterials with tunable bandgaps.
10. Life-cycle assessment studies of wastewater nutrient recovery systems.

## 7) Evaluation Harness Results
Primary metric:
- **Domain Hit@5 = 0.90** (9/10 queries include expected domain in top-5).

Artifacts:
- Full machine-readable results: `reports/phase1_validation/artifacts/validation_results.json`
- Full console log (unabridged JSON output): `reports/phase1_validation/artifacts/evaluation_harness_full.log`

## 8) Known Limitations
1. **No cross-domain deduplication**
   - A paper can appear in multiple domain buckets.

2. **Weakly supervised labels**
   - “Expected domain” is query-assigned, not manually adjudicated relevance.

3. **Sparse lexical embedding baseline**
   - TF-IDF does not capture deep semantics as well as transformer embeddings.

4. **Small evaluation set**
   - 10 queries are sufficient for smoke validation, not for publication-grade significance.

5. **OpenAlex ranking bias**
   - `sort=cited_by_count:desc` favors highly cited works and may underrepresent novel papers.

6. **No reranker**
   - Single-stage retrieval only; no cross-encoder reranking.

## 9) Reproducibility
Execution command:

```bash
python3 reports/phase1_validation/run_validation.py
```

Expected outputs:
- `reports/phase1_validation/artifacts/validation_results.json`
- `reports/phase1_validation/artifacts/evaluation_harness_full.log`

This completes Phase 1 “Freeze & Formalize” baseline documentation for technical validation.

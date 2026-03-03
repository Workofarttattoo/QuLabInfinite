# What Would Change My Mind: Investor Evidence Pack

This document converts the investor ask into a concrete diligence package with measurable pass/fail thresholds.

## 1) Vector database stats (show corpus quality + scale)

### What to report
- Total documents and chunk count.
- Embedding model, vector dimension, and index type.
- Chunk length distribution (p50/p90 tokens).
- Metadata coverage (paper DOI, year, modality, assay type).
- Duplicate rate and stale index rate.

### Investor-ready table (target ranges)
| Metric | Why it matters | Target |
|---|---|---|
| Indexed papers | Coverage breadth | 10k+ domain papers |
| Chunk count | Retrieval granularity | 200k+ chunks |
| Metadata completeness | Filtering + traceability | >= 95% |
| Duplicate chunk rate | Noise reduction | <= 2% |
| Re-index freshness | Operational reliability | <= 24h lag |

### Evidence artifact
- `vector_stats.json` generated per ingest run and attached to diligence room.

## 2) Retrieval latency (show production readiness)

### What to report
- p50/p95/p99 end-to-end retrieval latency by query type.
- Throughput at fixed concurrency (e.g., 10/50/100 QPS).
- Cold vs warm cache behavior.
- Timeout/error budget (HTTP + vector backend).

### Investor-ready SLA targets
| Metric | Target |
|---|---|
| p50 latency | <= 120 ms |
| p95 latency | <= 300 ms |
| p99 latency | <= 600 ms |
| Error rate | < 0.5% |

### Evidence artifact
- `retrieval_latency_report.md` with load-test plots and raw logs.

## 3) Top-k relevance accuracy (show retrieval quality)

### What to report
- Recall@k, Precision@k, and nDCG@k on a held-out query set.
- Split by task type: mechanistic Q&A, protocol lookup, materials parameter lookup.
- Human adjudication agreement (Cohen's kappa) for relevance labels.

### Investor-ready quality bar
| Metric | k | Target |
|---|---|---|
| Recall@k | 5 | >= 0.85 |
| Precision@k | 5 | >= 0.70 |
| nDCG@k | 10 | >= 0.80 |
| Inter-rater agreement | - | >= 0.75 |

### Evidence artifact
- `topk_relevance_eval.json` + labeling rubric.

## 4) Cross-paper synthesis benchmark (show non-trivial reasoning)

### Benchmark definition
- Build tasks where answers require combining evidence from **2-5 papers**, not single-document lookup.
- Include contradictions, reagent substitutions, and condition trade-offs.
- Score on:
  1. Correctness of final recommendation.
  2. Citation support (all key claims grounded).
  3. Consistency handling (resolves conflicting papers explicitly).

### Investor-ready target
- >= 75% pass rate on curated cross-paper tasks.
- >= 90% citation-grounded claims in generated outputs.

### Evidence artifact
- `cross_paper_synthesis_benchmark.md` with blinded eval set and scorer notes.

## 5) “Now we’re talking real capital”: tie corpus to lab validation

To unlock higher valuation, connect retrieval outputs to wet/dry lab outcomes using an auditable chain:

1. **Query -> Retrieved evidence** (paper/chunk IDs + scores).
2. **Model recommendation** (conditions, target, rationale, confidence).
3. **Experiment execution record** (protocol, instrument, operator, timestamp).
4. **Measured result** (yield, purity, kinetics, toxicity, etc.).
5. **Back-linking** (which retrieved papers actually predicted outcome).

### Validation scorecard (minimum investor pack)
| Layer | Metric | Target |
|---|---|---|
| Recommendation fidelity | Predicted vs observed trend agreement | >= 80% |
| Experimental uplift | Success rate vs baseline protocol | +20% relative |
| Reproducibility | Replicate coefficient of variation | <= 15% |
| Evidence traceability | Claims linked to source papers | 100% |

## 6) Mapping to current QuLabInfinite components

- Data ingestion and source connectors are already organized under `ingest/` (pipeline, plugins, sources), which is the natural place to emit vector stats and freshness telemetry.
- Experimental results storage and retrieval are already represented in `ingest/results.py`; extend this path to store benchmark and lab-outcome joins.
- Chemistry validation foundations exist in `chemistry_lab/validation/kinetics_validation.py`; reuse this pattern to add corpus-conditioned validation tasks.

## 7) Suggested 30-day execution plan

### Week 1: Instrumentation
- Add vector index telemetry export (`vector_stats.json`).
- Add latency tracing around retrieval endpoint(s) with percentile rollups.

### Week 2: Gold set + relevance eval
- Curate 300-500 investor-facing retrieval queries.
- Label top-10 results with dual reviewers.
- Ship Recall@k/Precision@k/nDCG dashboard.

### Week 3: Cross-paper benchmark
- Build 100 synthesis tasks requiring multi-paper fusion.
- Run benchmark on current system and baseline retriever-only system.

### Week 4: Lab tie-in
- Select 10-20 high-value recommendations.
- Execute validation loop and publish uplift/reproducibility metrics.

## 8) Board-ready one-liner

If QuLabInfinite demonstrates (a) fast retrieval SLAs, (b) high top-k relevance, (c) robust multi-paper synthesis performance, and (d) measurable lab uplift traceable to corpus evidence, the platform transitions from “interesting AI tooling” to a defensible scientific operating system suitable for materially higher valuation.

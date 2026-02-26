# Investor Evidence Pack: "What Would Change My Mind"

This document gives a board-ready answer to the investor challenge:

> "If you show vector database stats, retrieval latency, top-k relevance accuracy, and a cross-paper synthesis benchmark, valuation goes up. If you show lab validation tied to that corpus, now we're talking real capital."

## 1) Vector Database Stats (show quality + scale)

Report these metrics in a one-page dashboard:

- **Corpus size**: papers, paragraphs/chunks, and tokens indexed.
- **Coverage quality**: percent of papers with full-text vs abstract-only ingestion.
- **Embedding integrity**:
  - mean chunk length
  - duplicate chunk rate
  - outlier/failed embedding rate
- **Freshness**:
  - median time from publication to indexed availability
  - re-index lag after parser/schema updates
- **Metadata completeness**:
  - % chunks with DOI/arXiv ID, publication year, domain tags

### Minimum investor-grade targets

- Failed embedding rate: **< 0.5%**
- Duplicate chunk rate: **< 2%**
- Full-text coverage in target domains: **> 80%**
- Index refresh SLA: **< 24h** for newly added corpus sources

## 2) Retrieval Latency (show production readiness)

Measure under realistic concurrent load.

- **p50 / p95 / p99 query latency** for end-to-end retrieval.
- Separate stages:
  - query embedding latency
  - vector search latency
  - metadata join/rerank latency
- **Concurrency profile**: latency at 1, 10, 50, and 100 parallel users.
- **Cold vs warm cache** behavior.

### Minimum investor-grade targets

- p95 retrieval latency at normal load: **< 500 ms**
- p99 retrieval latency at normal load: **< 900 ms**
- No timeout spikes at 10x expected demo traffic

## 3) Top-k Relevance Accuracy (show retrieval trustworthiness)

Use a labeled benchmark set of real scientific questions.

- Evaluate **Recall@k, Precision@k, MRR, and nDCG**.
- Use question sets by domain (materials, chemistry, biology, medical).
- Include **hard negatives** (semantically similar but wrong papers).
- Run blind annotation by at least two SMEs for gold labels.

### Minimum investor-grade targets

- Recall@10: **> 0.85** on core domains
- nDCG@10: **> 0.80**
- Inter-annotator agreement (Cohen's kappa): **> 0.7**

## 4) Cross-Paper Synthesis Benchmark (show "insight engine," not just search)

Design benchmark tasks where the answer requires combining facts from multiple papers.

Task types:

1. **Mechanism synthesis**: unify findings across contradictory studies.
2. **Protocol transfer**: adapt a method from domain A to domain B with rationale.
3. **Constraint-aware recommendation**: produce candidate experiments obeying safety, cost, and availability constraints.
4. **Evidence conflict resolution**: detect disagreements and surface confidence.

Scoring:

- Factual support rate (claims traceable to cited passages)
- Citation coverage (key claims with citations)
- Hallucination rate
- SME utility score (1-5)

### Minimum investor-grade targets

- Hallucination rate: **< 5%** on benchmark prompts
- Claim traceability: **> 90%**
- SME utility score: **>= 4.0/5**

## 5) Lab Validation Tied to the Corpus (the valuation inflection point)

This is the "real capital" bridge: show closed-loop evidence from literature -> hypothesis -> experiment -> observed result.

### Validation framework

For each selected use case:

1. **Question**: concrete experimental objective.
2. **Corpus evidence bundle**: top retrieved papers and extracted rationale.
3. **Model-generated hypothesis/protocol**.
4. **Lab execution** (internal or partner lab).
5. **Outcome** versus baseline protocol.
6. **Attribution**: which cited evidence changed protocol decisions.

### Minimum investor-grade targets

- At least **3 prospective experiments** (not only retrospective matching).
- **Statistically significant** improvement over baseline in at least **2/3** experiments.
- Full provenance trail from corpus chunk -> recommendation -> lab step -> outcome.

## 6) Suggested "Investor Upgrade" Scorecard

Use one summary scorecard for fundraising meetings:

- Data Infrastructure (vector DB quality): /25
- System Performance (latency/reliability): /20
- Retrieval Quality (top-k relevance): /20
- Scientific Reasoning (cross-paper synthesis): /20
- Real-World Validation (lab outcomes): /15

**Valuation trigger recommendation**: treat **>= 80/100** with lab-backed prospective wins as the threshold for a major step-up in narrative and pricing.

## 7) 30-Day Execution Plan to Produce This Evidence

Week 1:

- Freeze benchmark datasets and gold-label rubric.
- Instrument retrieval pipeline for full latency tracing.
- Add automatic quality checks for embedding/index integrity.

Week 2:

- Run top-k and synthesis benchmarks (v1).
- Perform error analysis and reranker tuning.
- Lock investor dashboard format.

Week 3:

- Launch prospective lab validation runs with partner protocols.
- Capture full provenance linking corpus evidence to experimental decisions.

Week 4:

- Final benchmark rerun + confidence intervals.
- Produce one-page KPI sheet + appendix with methods.
- Package diligence folder for investors.

## 8) Board-Ready One-Liner

"We don't just retrieve papers quickly; we produce traceable, cross-paper scientific recommendations that are prospectively validated in the lab with measurable lift over baseline workflows."

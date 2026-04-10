SIMTEST v1 — Standard for Simulation Testing (Materials/Chemistry and Mechanics/Thermal/CFD)

Version: 1.0.0
Status: Draft (Phase 1)

1. Purpose and scope

SIMTEST v1 defines a unified, machine-validated standard for evaluating simulation engines against curated benchmarks. Phase 1 targets materials/chemistry and mechanics/thermal/CFD, with an extensible core to support additional domains (e.g., quantum, frequency, environmental) without breaking changes.

2. Guiding principles

- Reproducibility: Complete provenance must accompany every result.
- Comparability: Common metrics and tolerances enable fair comparisons.
- Extensibility: Domain-specific fields are allowed via a controlled extension section.
- Determinism within tolerance: Numeric differences are expected; metrics define acceptable bounds.
- Automation-first: Validation via JSON Schemas; CLI and CI required for conformance.

3. Data model

3.1 Test case (authoritative schema: standards/schemas/test_case.schema.json)

Required fields:
- id: string (unique, stable)
- version: string (semver)
- domain: enum {materials, chemistry, mechanics, thermal, cfd}
- description: string (human-readable)
- problem: object (domain-independent shape with domain_config for specifics)
- discretization: object (mesh/order/timestep/etc.)
- tolerances: object mapping metric_name → tolerance spec
- references: object mapping reference_name → number or array (ground truth or accepted baseline)
- metrics: array of metric specs to compute and check

Optional fields:
- tags: array of strings
- artifacts: array of artifact descriptors (e.g., reference meshes)
- notes: string

Tolerance spec (per metric):
- type: enum {absolute, relative_pct}
- value: number (for absolute: units of metric; for relative_pct: percentage)

Metric spec:
- name: string
- reducer: enum {L2, Linf, MAE, RMSE, mean_pct_error, F1, AUROC, custom}
- target: string (the reference key to compare against, if applicable)
- pass_if: enum {<=, <, >=, >, between}
- range: [lo, hi] (required if pass_if == between)
- unit: string (optional)

Domain configuration:
- domain_config: object (free-form per domain, validated by domain extension logic)

3.2 Result record (authoritative schema: standards/schemas/result_record.schema.json)

Required fields:
- test_id: string (matches test case id)
- test_version: string
- engine: { name: string, version: string }
- run_config: object (parameters used by the engine: mesh order, dt, solver tolerances)
- metrics: object mapping metric_name → number
- status: enum {pass, fail, error}
- provenance: see 3.3

Optional fields:
- outputs: object (aggregated outputs or sample slices)
- checks: array of { metric: string, observed: number, threshold: number or [lo,hi], pass: boolean }
- duration_s: number
- artifacts: array of { name: string, uri: string, type: string }
- errors: array of strings

3.3 Provenance (authoritative schema: standards/schemas/provenance.schema.json)

Minimum required:
- python: string (e.g., 3.11.9)
- os_name: string; os_version: string; architecture: string
- hardware: { cpu_model: string, memory_gb: number, gpu_model?: string, gpu_driver?: string }
- container_image?: string; container_digest?: string
- repo: { commit: string, dirty: boolean }
- dependencies: array of { name: string, version: string }
- timestamp_utc: string (ISO 8601)

4. Domains and canonical metrics (Phase 1)

4.1 Materials/Chemistry
- Formation energy: MAE/RMSE vs. reference energies
- Lattice constants: RMS% error vs. reference lattice parameters
- Elastic constants: % error per component; aggregated MAE%
- Phase stability: F1 score of predicted-stable vs. reference convex hull
- Reaction energies/barriers: MAE vs. reference (eV or kJ/mol)
- Kinetics: log-error of rate constants (ln-scale)

4.2 Mechanics
- Beam/cantilever deflection: tip deflection % error
- Stress fields: L2 and Linf relative error on stress components
- Energy balance: residual magnitude below threshold

4.3 Thermal
- 1D/2D conduction: L2 field error of temperature vs. analytic solution
- Transient rod: Linf error over time vs. exact/semi-analytic solution

4.4 CFD
- Lid-driven cavity (Re=100): L2 velocity field error vs. benchmark
- Channel (Poiseuille): relative error of centerline velocity and flow rate
- Flat-plate: Cd/Cl/Nu nondimensional coefficients within tolerances

5. File layout

- Benchmarks:
  - bench/materials/*.json
  - bench/chemistry/*.json
  - bench/mechanics/*.json
  - bench/thermal/*.json
  - bench/cfd/*.json
- Schemas:
  - standards/schemas/test_case.schema.json
  - standards/schemas/result_record.schema.json
  - standards/schemas/provenance.schema.json

6. CLI contract (simtest)

Subcommands:
- simtest validate --tests path_or_glob [--results path_or_glob]
  - Validates JSONs against their schemas
- simtest run --suite bench/<domain> --engine qulab_unified [--out results/]
  - Loads tests, dispatches to core/unified_simulator.py domain adapters
  - Computes metrics and writes standardized results JSON
- simtest summarize --results results/ --out report.json
  - Aggregates results, computes pass rates, and emits a summary

Exit codes:
- 0 on success with all tests passing; 1 on validation or runtime errors; 2 on test failures

7. Conformance requirements

- Must pass schema validation for tests and results
- Must include complete provenance
- Must fix random seeds where applicable; if not applicable, must document stochasticity and provide repeatability stats
- Must run within the pinned container image or provide a fully reproducible environment manifest
- Must publish duration_s and hardware details

8. Tolerance semantics

- absolute: |observed - reference| <= value
- relative_pct: 100 * |observed - reference| / max(|reference|, eps) <= value
- between: lo <= observed <= hi (used for nondimensional targets)

9. Certification (Phase 1)

Levels:
- Bronze: ≥80% of mandatory tests pass in at least one domain
- Silver: ≥90% of mandatory tests pass in both domains (materials/chemistry and mechanics/thermal/CFD)
- Gold: ≥95% pass, plus full provenance, containerized runs, and published artifacts

10. Versioning and governance

- Minor versions may add optional fields and tests
- Major versions may change schemas or mandatory tests; deprecation window ≥ 6 months
- Changes tracked in this file; schema $id values include version suffixes

11. Security and safety

- No execution of untrusted code during validation
- Benchmarks and artifacts must be vetted for licensing and PII-free content

12. References

- JSON Schema (2020-12)
- Canonical benchmark papers for CFD cavity flow, Poiseuille, thermal conduction, elastic beams



# SIMTEST v1 Certification Policy

## Overview

SIMTEST certification provides standardized levels of validation for simulation engines. Engines can achieve Bronze, Silver, or Gold certification based on test pass rates, domain coverage, and compliance with reproducibility requirements.

## Certification Levels

### Bronze

**Requirements:**
- ≥80% pass rate on mandatory tests in at least one domain
- Complete provenance reporting
- All submitted results must pass schema validation

**Mandatory Tests (Phase 1):**
- Materials: `materials_si_formation_energy_v1`
- Chemistry: `chemistry_h2o_formation_energy_v1`
- Mechanics: `mech_cantilever_304ss_v1`
- Thermal: `thermal_rod_transient_v1`
- CFD: `cfd_lid_driven_cavity_Re100_v1`

**Benefits:**
- Listed in SIMTEST registry
- Bronze badge for display
- Access to baseline benchmarks

### Silver

**Requirements:**
- ≥90% pass rate on mandatory tests in **both** domains:
  - Materials/Chemistry domain (all mandatory tests)
  - Mechanics/Thermal/CFD domain (all mandatory tests)
- Complete provenance reporting
- Containerized execution or fully reproducible environment manifest
- Results must be submitted via API or validated JSON files

**Additional Tests Required:**
- Must pass at least 3 additional tests beyond mandatory set
- Documentation of engine capabilities and limitations

**Benefits:**
- Silver badge for display
- Priority listing in SIMTEST registry
- Access to extended benchmark suite
- Official certification document

### Gold

**Requirements:**
- ≥95% pass rate on mandatory tests across **all** domains (Materials, Chemistry, Mechanics, Thermal, CFD)
- Complete provenance with pinned dependency versions
- Containerized execution with published container image
- Publicly accessible artifacts (reference implementations, documentation)
- Published performance metrics (speed, accuracy, resource usage)
- Minimum 6 months of consistent results (if submitting historical data)

**Additional Requirements:**
- Must pass at least 5 additional tests beyond mandatory set
- Engine must be publicly available or have documented access policy
- Technical documentation describing engine methodology
- Open-source preferred (not mandatory)

**Benefits:**
- Gold badge for display
- Featured listing in SIMTEST registry
- Access to full benchmark suite including experimental tests
- Official certification document with detailed metrics
- Eligibility for SIMTEST steering committee voting rights

## Certification Process

### 1. Submission

Submit test results via:
- API endpoint: `POST /api/v1/test-runs`
- File upload: `POST /api/v1/test-runs/file`
- Direct JSON file to SIMTEST maintainers

### 2. Validation

SIMTEST validators will:
- Validate all result records against JSON Schemas
- Verify provenance completeness
- Check reproducibility (if containerized)
- Compute pass rates and certification eligibility

### 3. Review

- Bronze: Automated review (if all requirements met)
- Silver: Automated review + spot-check manual review
- Gold: Full manual review + reproducibility verification

### 4. Issuance

Upon approval:
- Certification badge issued
- Entry added to public leaderboard
- Certification document generated (Silver/Gold only)

## Recertification

- **Bronze**: Valid for 12 months; recertify with same requirements
- **Silver**: Valid for 12 months; recertify with same or higher requirements
- **Gold**: Valid for 24 months; recertify with same or higher requirements

## Revocation

Certification may be revoked if:
- Provenance is found to be falsified
- Results cannot be reproduced
- Engine behavior changes significantly without re-submission
- Submission of invalid or malicious test cases

## Appeals

Engines may appeal revocation or certification denial within 30 days. Appeals are reviewed by the SIMTEST steering committee.

## Phase 1 Domains

Current certification levels apply to:
- **Domain 1**: Materials and Chemistry
- **Domain 2**: Mechanics, Thermal, and CFD

Future phases will add:
- Quantum
- Frequency
- Environmental
- Cross-domain multi-physics

## Certification Badges

Badges are provided as:
- SVG for web display
- PNG for documents
- Markdown embed code

Example:
```markdown
[![SIMTEST Gold](https://qulab.ai/simtest/badges/gold.svg)](https://qulab.ai/simtest/leaderboard)
```

## Questions

For certification questions, contact: simtest@qulab.ai

---

**Last Updated**: 2025-01-XX
**Version**: 1.0.0


# Data Management Plan
## NSF SBIR Phase I: QuLab Infinite

**Company**: Work of Art Tattoo (DBA: Corporation of Light)
**EIN**: 61-6036925
**Principal Investigator**: Joshua Hendricks Cole
**Project**: QuLab Infinite - Open-Source Scientific Computing Platform

---

## 1. TYPES OF DATA PRODUCED

### Research Data

**Platform Usage Data:**
- User analytics (anonymized): page views, lab usage, session duration
- Performance metrics: API response times, computation speeds, error rates
- A/B test results: feature adoption, user interface effectiveness
- Feedback surveys: user satisfaction, feature requests, pain points

**Scientific Validation Data:**
- Algorithm correctness tests: input/output pairs, edge cases
- Performance benchmarks: computation speed vs. MATLAB/Mathematica
- Accuracy measurements: numerical precision, convergence rates
- Cross-validation results: QuLab vs. peer-reviewed published results

**Educational Impact Data:**
- Pre/post assessments: student learning outcomes
- Teacher feedback: curriculum integration success
- Institutional adoption rates: schools using QuLab
- Demographic data: reach in underserved communities (aggregated, anonymized)

### Software Artifacts

**Source Code:**
- QuLab platform code (Python, JavaScript, React)
- 100+ scientific lab implementations
- ECH0 AI autonomous lab builder
- Test suites and validation scripts

**Documentation:**
- API documentation
- User guides and tutorials
- Developer documentation
- Scientific methodology papers

**Data Formats:**
- JSON: API responses, configuration files
- CSV/TSV: Tabular data, benchmark results
- HDF5: Large scientific datasets
- Markdown: Documentation
- Git repositories: Version-controlled code

---

## 2. DATA STANDARDS AND METADATA

### Standards Compliance

**Software Development:**
- **Version Control**: Git (GitHub) with semantic versioning (SemVer)
- **Code Style**: PEP 8 (Python), ESLint (JavaScript)
- **Documentation**: NumPy docstring format, JSDoc
- **Testing**: pytest (Python), Jest (JavaScript)
- **APIs**: OpenAPI 3.0 specification

**Scientific Data:**
- **Numerical Standards**: IEEE 754 floating-point arithmetic
- **Units**: SI units with NIST constants
- **Algorithms**: Peer-reviewed publications cited for all implementations
- **Validation**: NIST reference data where available

**Privacy & Security:**
- **User Data**: GDPR and CCPA compliant
- **Anonymization**: Industry-standard techniques (k-anonymity, differential privacy)
- **Access Control**: Role-based access (RBAC)
- **Encryption**: TLS 1.3 for data in transit, AES-256 for data at rest

### Metadata Schema

**Code Repositories:**
- README.md: Project description, installation, usage
- LICENSE: MIT open-source license
- CITATION.cff: Citation metadata
- pyproject.toml: Python project metadata
- package.json: JavaScript project metadata

**Scientific Labs:**
```json
{
  "lab_id": "unique_identifier",
  "name": "Lab Name",
  "description": "What the lab does",
  "discipline": "Physics/Chemistry/Biology/...",
  "algorithms": ["Algorithm 1", "Algorithm 2"],
  "citations": ["DOI1", "DOI2"],
  "version": "1.0.0",
  "license": "MIT",
  "author": "Joshua Hendricks Cole",
  "date_created": "2025-01-01",
  "validated": true,
  "validation_date": "2025-01-15"
}
```

**Research Data:**
- Date/time of collection (ISO 8601)
- Provenance: who collected, how, with what tools
- Version: data schema version, software version used
- License: CC BY 4.0 (Creative Commons Attribution)

---

## 3. DATA ACCESS AND SHARING

### Open Science Commitment

**All Data and Code Will Be:**
- **Open**: MIT license (code), CC BY 4.0 (data)
- **Accessible**: GitHub (code), Zenodo (datasets)
- **FAIR**: Findable, Accessible, Interoperable, Reusable
- **Immediate**: No embargo period (released as created)

### Code Repositories

**Primary Repository:**
- **Location**: https://github.com/corporation-of-light/qulab-infinite
- **Access**: Public, open to all
- **License**: MIT (permissive, commercial use allowed)
- **DOI**: Zenodo integration for citeable releases

**Backup Repositories:**
- GitLab mirror (disaster recovery)
- Archive.org periodic snapshots

### Research Data

**Data Repository:**
- **Primary**: Zenodo (CERN-hosted, permanent)
- **Secondary**: figshare (backup)
- **Format**: Structured datasets with metadata
- **DOI**: Permanent identifiers for citation
- **Access**: Open download, no registration required

**Documentation Repository:**
- **Location**: ReadTheDocs (https://qulab-infinite.readthedocs.io)
- **Format**: HTML, PDF, EPUB
- **Source**: GitHub (Markdown/reStructuredText)
- **Version Control**: Tagged releases synchronized with code

### Publications

**Preprints:**
- arXiv: Computer Science, Physics, Mathematics
- bioRxiv: Biological sciences
- chemRxiv: Chemistry
- **Timing**: Immediate (same day as submission or discovery)

**Peer-Reviewed:**
- Target journals: Nature Computational Science, PLOS ONE, JOSS, JOSE
- All publications open access (Gold OA when funded, Green OA otherwise)
- Preprints always available (no paywall)

### User Data (Privacy-Protected)

**Aggregate Statistics:**
- Monthly reports: user counts, lab usage, geographic distribution
- Anonymized, aggregated (no individual identification)
- Published on project website and in annual reports

**Individual Data:**
- Not shared publicly (privacy protection)
- Users can download their own data (GDPR right)
- Researchers can request anonymized datasets (IRB approval required)

---

## 4. DATA STORAGE AND PRESERVATION

### Active Development (Phase I: 6 Months)

**Primary Storage:**
- **Code**: GitHub (redundant, geographically distributed)
- **Data**: Google Cloud Storage (GCS) - us-west1 region
- **Databases**: PostgreSQL on Google Cloud SQL
- **Backups**: Daily snapshots, 30-day retention

**Capacity:**
- Code: ~10 GB (100+ labs, tests, documentation)
- User data: ~100 GB (10,000 users estimated)
- Scientific data: ~50 GB (validation datasets, benchmarks)
- **Total**: ~200 GB (well within NSF SBIR budget)

**Redundancy:**
- 3-2-1 rule: 3 copies, 2 media types, 1 offsite
- GitHub (primary + backup servers)
- GCS (multi-region replication)
- Local backup (PI's encrypted drive)

### Long-Term Preservation (10+ Years)

**Permanent Archives:**

1. **Zenodo**
   - CERN-operated, committed to 20+ year preservation
   - All releases archived with DOI
   - Automatic from GitHub integration
   - Free for open science projects

2. **Software Heritage**
   - Universal software archive
   - Captures all public Git repositories
   - Permanent preservation of source code
   - Automatic (no action required)

3. **Internet Archive**
   - Wayback Machine: website snapshots
   - Scholar: academic papers
   - Software Library: executable preservation

**Format Migration:**
- Monitor format obsolescence (e.g., Python 3.x → 4.x)
- Migrate data to new formats as needed
- Document migration process and maintain old formats
- NSF funding covers migration costs

### Disaster Recovery

**Threats Mitigated:**
- Hardware failure: Cloud redundancy
- Data corruption: Checksums (SHA-256), version control
- Accidental deletion: 30-day backup retention, Git history
- Service shutdown: Multiple independent archives
- PI incapacitation: Successor documented, community can fork

**Recovery Time Objective (RTO):** 24 hours
**Recovery Point Objective (RPO):** 1 day (daily backups)

---

## 5. ROLES AND RESPONSIBILITIES

### Data Management Roles

**Principal Investigator (Joshua Hendricks Cole):**
- Overall responsibility for data management
- Ensure compliance with NSF policies
- Approve data releases and publications
- Coordinate with NSF program officer

**Senior Software Engineer (To Be Hired):**
- Implement data collection and storage systems
- Maintain backups and archives
- Monitor data quality and integrity
- Execute disaster recovery if needed

**DevOps Engineer (To Be Hired, Part-time):**
- Manage cloud infrastructure (GCS, Cloud SQL)
- Configure automated backups
- Monitor storage costs and optimize
- Implement security best practices

**Community:**
- Report bugs, suggest improvements
- Contribute code and documentation
- Validate scientific accuracy
- Translate content to other languages

### Oversight

**Internal:**
- Quarterly data management reviews
- Annual audit of archive integrity
- User privacy compliance checks

**External:**
- NSF program officer: Annual reports
- Community feedback: GitHub issues, discussions
- Peer review: Publications, conferences

---

## 6. BUDGET FOR DATA MANAGEMENT

### Phase I Costs (6 Months, included in $275K budget)

| Item | Cost | Notes |
|------|------|-------|
| **Google Cloud Storage** | $3,000 | 200 GB data, multi-region |
| **Cloud SQL (PostgreSQL)** | $2,000 | User data, analytics |
| **Zenodo** | $0 | Free for open science |
| **GitHub** | $0 | Free for public repos |
| **Personnel (15% of effort)** | $10,000 | Engineer time for data management |
| **Total** | **$15,000** | **5.5% of budget** |

### Long-Term Costs (Post-Phase I)

**Sustainability:**
- Cloud costs scale with usage (users pay through freemium model)
- Archives (Zenodo, Software Heritage) are free
- Maintenance covered by institutional licenses and grants

---

## 7. POLICIES AND PROVISIONS

### Compliance

**NSF Requirements:**
- Public Access Plan: All NSF-funded data and publications openly accessible
- Data Sharing: FAIR principles (Findable, Accessible, Interoperable, Reusable)
- Timeline: Immediate release (no embargo)

**Institutional:**
- No institutional affiliation yet (sole proprietor)
- Will comply with future institution's policies if affiliated

**Legal:**
- GDPR: User privacy rights (EU)
- CCPA: User privacy rights (California)
- HIPAA: Not applicable (no health data collected)
- Export Control: Open-source exemption (no restricted algorithms)

### Intellectual Property

**NSF Rights:**
- NSF has non-exclusive, royalty-free license to all NSF-funded data/code
- Worldwide right to use, reproduce, distribute for NSF purposes

**Public Rights:**
- MIT License (code): Anyone can use, modify, distribute commercially
- CC BY 4.0 (data): Anyone can share, adapt with attribution

**Patent Considerations:**
- Core algorithms are open-source (non-patentable)
- Patent-pending aspects are separate from NSF-funded work
- No restrictions on open data/code due to patents

### Data Retention

**Minimum:**
- NSF requirement: 3 years after final report
- Our commitment: 10+ years (Zenodo/Software Heritage)

**Deletion:**
- User-provided data: Deleted upon request (GDPR right to erasure)
- Research data: Never deleted (permanent archive)
- Anonymized/aggregate data: Cannot be deleted (irreversibly anonymized)

---

## 8. ETHICAL CONSIDERATIONS

### User Privacy

**Principles:**
- Minimize data collection (only what's needed)
- Anonymize wherever possible
- Transparent privacy policy
- User control over their data

**Practices:**
- No tracking of personally identifiable information (PII) without consent
- IP addresses hashed before storage
- User emails encrypted at rest
- No sale of user data (violates open science values)

### Equity and Inclusion

**Accessibility:**
- Data and code available to all (no paywalls)
- Documentation in multiple languages
- Works on low-bandwidth connections
- Screen reader compatible

**Representation:**
- Demographic data collected to measure impact in underserved communities
- Aggregated only (no individual identification)
- Optional (users can opt out)
- Used to demonstrate NSF broader impacts

### Reproducibility

**Goal:**
- Any researcher should be able to reproduce QuLab results

**Practices:**
- All code open-source (exact same software)
- All data openly available (exact same inputs)
- Version pinning (requirements.txt, package-lock.json)
- Docker containers (reproducible environment)
- CI/CD runs tests on every commit (continuous validation)

---

## 9. TIMELINE

### Phase I (Months 1-6)

**Month 1:**
- Set up GitHub repository (public)
- Configure Google Cloud Storage
- Implement basic analytics
- Draft privacy policy

**Months 2-4:**
- Collect user data (with consent)
- Benchmark QuLab vs. proprietary tools
- Validate scientific accuracy
- Write documentation

**Months 5-6:**
- Archive first release on Zenodo
- Publish preprints (arXiv, bioRxiv)
- Submit papers to peer-reviewed journals
- Final data management report to NSF

### Post-Phase I

**Ongoing:**
- Quarterly Zenodo releases
- Annual data management audits
- Continuous documentation updates
- Community data contributions

---

## 10. CONTACT

**Data Management Inquiries:**

Joshua Hendricks Cole
Principal Investigator
inventor@aios.is

**Data Repository:**
https://github.com/corporation-of-light/qulab-infinite
https://zenodo.org/communities/qulab (coming soon)

---

**This Data Management Plan ensures that NSF-funded QuLab research is open, reproducible, and has lasting impact beyond the grant period.**

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved.**

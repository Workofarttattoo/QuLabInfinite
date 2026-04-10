# OpenQDC Datasets (external cache)

The full OpenQDC datasets are too large for version control (tens of GB), so we
point to the local cache under `~/.cache/openqdc/`. After downloading with the
OpenQDC CLI, use:

```bash
./scripts/hash_openqdc_cache.py
```

This produces `openqdc_cache_hashes.txt` containing SHA-256 fingerprints for all
files so experiments remain reproducible. For benchmarking, either generate
lightweight summaries via `scripts/extract_openqdc_samples.py` (which exports
CSV snippets) or configure your calibration scripts to stream directly from the
cache paths listed in `openqdc_registry.yaml`.

Example download commands:

```bash
openqdc download QMugs
openqdc download QM7X
openqdc download Spice
```

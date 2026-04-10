# Mechanics Engines

Store documented implementations of mechanics models here. Example layout:

```
engines/
  mechanics/
    mech_johnson_cook_v2/
      README.md
      model.py
      params.json
```

Each engine directory should specify:
- governing equations / assumptions,
- validity ranges (strain rate, temperature, etc.),
- last calibration date and provenance IDs,
- links to benchmarks under `bench/`.

Current placeholder – add real engines when calibration is available.

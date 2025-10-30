# Surrogate Models

Store reduced-order / ML surrogates that approximate high-fidelity engines.
Every surrogate should declare:
- the engine + benchmark it approximates,
- feature engineering / normalization scheme,
- acceptance criteria (same or stricter than underlying engine),
- domain guardrails (automatic out-of-domain detection).

Use hashed filenames or metadata to ensure provenance and reproducibility.

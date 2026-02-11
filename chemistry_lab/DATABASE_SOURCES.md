# Chemistry Lab - Database Sources

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## Primary Data Sources

All simulations in `chemistry_lab` are grounded in peer-reviewed experimental data.

### 1. NIST Chemical Kinetics Database
- **URL:** https://kinetics.nist.gov/
- **Used for:** Activation energies, pre-exponential factors.
- **Reference:** Manion, J. A., et al., NIST Chemical Kinetics Database, NIST Standard Reference Database 17, Version 7.1 (Web Version), National Institute of Standards and Technology, Gaithersburg, MD, 20899.

### 2. CRC Handbook of Chemistry and Physics
- **Edition:** 97th Edition (2016-2017)
- **Used for:** Acid dissociation constants (pKa), standard enthalpies of formation.
- **Reference:** Haynes, W. M. (Ed.). CRC Handbook of Chemistry and Physics. CRC Press.

### 3. NIST Chemistry WebBook
- **URL:** https://webbook.nist.gov/chemistry/
- **Used for:** Thermodynamic data (ΔH, ΔS, Cp), IR spectra.
- **Reference:** Linstrom, P.J. and Mallard, W.G., Eds., NIST Chemistry WebBook, NIST Standard Reference Database Number 69, National Institute of Standards and Technology, Gaithersburg MD, 20899.

### 4. Comparison Data (Internal)
- **H2O2 Decomposition:** Confirmed against 1997COH/FIS1-143.
- **Methyl Recombination:** Confirmed against 1992BAU/COB411-429.
- **Blood Buffer:** Physiological constants from Guyton and Hall Textbook of Medical Physiology.

## Data Integrity Policy
- All constants in `fast_kinetics_solver.py` are cited.
- No "magic numbers" without physical basis.
- CODATA 2018 values used for fundamental constants (R, k, h, Na).

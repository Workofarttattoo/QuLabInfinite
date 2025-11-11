# Drug Database Reference

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## Complete Drug Catalog

QuLabInfinite Oncology Lab now includes **27 drugs** across 5 categories, all with real pharmacokinetic and pharmacodynamic parameters from published literature.

---

## Quick Start

```python
from oncology_lab import OncologyLaboratory, OncologyLabConfig, TumorType, CancerStage
from oncology_lab.drug_response import list_available_drugs

# See all available drugs
drugs = list_available_drugs()
print(f"Available: {drugs}")

# Create lab and administer drugs
lab = OncologyLaboratory(OncologyLabConfig(
    tumor_type=TumorType.BREAST_CANCER,
    stage=CancerStage.STAGE_II,
))

# Administer any combination
lab.administer_drug("cisplatin", 135.0)  # mg
lab.administer_drug("metformin", 1000.0)
lab.administer_drug("vitamin_d3", 5000.0)

# Run experiment
lab.run_experiment(duration_days=21)
results = lab.get_results()
```

---

## Drug Categories

### 1. Chemotherapy (7 drugs)

**DNA-Damaging Agents:**

| Drug | IC50 | Route | Approval | Key Features |
|------|------|-------|----------|--------------|
| **Cisplatin** | 1.5 µM | IV | 1978 | Gold standard platinum, high nephrotoxicity |
| **Carboplatin** | 5.0 µM | IV | 1989 | Less toxic than cisplatin, easier dosing |
| **Doxorubicin** | 0.5 µM | IV | 1974 | Anthracycline, cardiotoxic (lifetime limit) |

**Antimetabolites:**

| Drug | IC50 | Route | Approval | Key Features |
|------|------|-------|----------|--------------|
| **5-Fluorouracil** | 1.0 µM | IV | 1962 | Thymidylate synthase inhibitor, S-phase |
| **Gemcitabine** | 0.05 µM | IV | 1996 | Nucleoside analog, pancreatic cancer |

**Antimicrotubule:**

| Drug | IC50 | Route | Approval | Key Features |
|------|------|-------|----------|--------------|
| **Paclitaxel** | 0.01 µM | IV | 1992 | Stabilizes microtubules, M-phase specific |

**Alkylating:**

| Drug | IC50 | Route | Approval | Key Features |
|------|------|-------|----------|--------------|
| **Temozolomide** | 0.3 µM | Oral | 1999 | Excellent CNS penetration, glioblastoma |

---

### 2. Targeted Therapy (4 drugs)

**Tyrosine Kinase Inhibitors:**

| Drug | IC50 | Target | Approval | Key Features |
|------|------|--------|----------|--------------|
| **Erlotinib** | 0.002 µM | EGFR | 2004 | NSCLC with EGFR mutation |
| **Imatinib** | 0.0001 µM | BCR-ABL | 2001 | CML, revolutionized treatment |
| **Vemurafenib** | 0.001 µM | BRAF | 2011 | Melanoma with BRAF V600E |

**Monoclonal Antibodies:**

| Drug | IC50 | Target | Approval | Key Features |
|------|------|--------|----------|--------------|
| **Trastuzumab** | 0.0001 µM | HER2 | 1998 | Breast cancer (HER2+), cardiotoxic |

---

### 3. Immunotherapy (2 drugs)

**Checkpoint Inhibitors:**

| Drug | IC50 | Target | Approval | Key Features |
|------|------|--------|----------|--------------|
| **Pembrolizumab** | 0.0001 µM | PD-1 | 2014 | Broad indications, 26-day half-life |
| **Nivolumab** | 0.0001 µM | PD-1 | 2014 | Similar to pembrolizumab, Q2W dosing |

---

### 4. Antiangiogenic (1 drug)

| Drug | IC50 | Target | Approval | Key Features |
|------|------|--------|----------|--------------|
| **Bevacizumab** | 0.0005 µM | VEGF-A | 2004 | Blocks angiogenesis, 20-day half-life |

---

### 5. Metabolic Inhibitors (13 drugs)

**FDA-Approved (Repurposed):**

| Drug | IC50 | Mechanism | Key Features |
|------|------|-----------|--------------|
| **Metformin** | 50 µM | AMPK activator, Complex I inhibitor | Diabetes drug, anti-cancer effects |
| **Ivermectin** | 5 µM | PAK1/Akt/mTOR inhibitor | Antiparasitic, anti-cancer potential |
| **Mebendazole** | 0.3 µM | Tubulin inhibitor, VEGFR2 | Antiparasitic, strong anti-cancer |
| **Hydroxychloroquine** | 10 µM | Autophagy inhibitor | Antimalarial, 40-day half-life |
| **Aspirin** | 50 µM | COX-2 inhibitor | OTC, cancer prevention |

**Natural Compounds:**

| Drug | IC50 | Source | Key Features |
|------|------|--------|--------------|
| **Vitamin D3** | 50 µM | Cholecalciferol | VDR agonist, differentiation |
| **Vitamin C** | 500 µM | Ascorbic acid | Pro-oxidant at high IV doses |
| **Curcumin** | 10 µM | Turmeric | NF-κB inhibitor, poor bioavailability |
| **Quercetin** | 15 µM | Onions, berries | PI3K/Akt inhibitor, senolytic |
| **Resveratrol** | 20 µM | Grapes, wine | SIRT1 activator, anti-inflammatory |
| **EGCG** | 10 µM | Green tea | EGFR inhibitor, antioxidant |

**Experimental:**

| Drug | IC50 | Mechanism | Key Features |
|------|------|-----------|--------------|
| **Dichloroacetate** | 10 mM | PDK inhibitor | Reverses Warburg effect, neurotoxic |
| **Fenbendazole** | 0.5 µM | Tubulin + GLUT inhibitor | Veterinary, anecdotal human use |

---

## Protocol Examples

### Example 1: Standard Chemo + Metabolic Support

```python
lab.administer_drug("cisplatin", 135.0)
lab.administer_drug("metformin", 1000.0)
lab.administer_drug("vitamin_d3", 5000.0)
```

**Rationale:** Metformin may sensitize tumors to chemotherapy via metabolic stress.

---

### Example 2: Natural Compound Stack (Joe Tippens-Inspired)

```python
lab.administer_drug("fenbendazole", 222.0)
lab.administer_drug("curcumin", 1000.0)
lab.administer_drug("vitamin_c", 5000.0)
lab.administer_drug("quercetin", 500.0)
lab.administer_drug("egcg", 400.0)
```

**Rationale:** Synergistic natural compounds with minimal toxicity.

---

### Example 3: Targeted + Immunotherapy

```python
lab.administer_drug("pembrolizumab", 200.0)
lab.administer_drug("vemurafenib", 960.0)
```

**Rationale:** Checkpoint inhibitors + BRAF inhibition for melanoma.

---

### Example 4: Repurposed Drug Cocktail

```python
patient_weight_kg = 70.0
lab.administer_drug("dichloroacetate", 25.0 * patient_weight_kg)
lab.administer_drug("ivermectin", 0.2 * patient_weight_kg)
lab.administer_drug("mebendazole", 100.0)
lab.administer_drug("hydroxychloroquine", 400.0)
lab.administer_drug("aspirin", 325.0)
```

**Rationale:** Multi-pathway metabolic targeting with repurposed drugs.

---

## Running Protocol Experiments

### Basic Workflow

```python
from oncology_lab import OncologyLaboratory, OncologyLabConfig, TumorType, CancerStage
from oncology_lab.ten_field_controller import create_ech0_three_stage_protocol

# 1. Create lab
config = OncologyLabConfig(
    tumor_type=TumorType.PANCREATIC_CANCER,
    stage=CancerStage.STAGE_III,
    initial_tumor_cells=100,
)
lab = OncologyLaboratory(config)

# 2. Apply field intervention protocol (optional)
protocol = create_ech0_three_stage_protocol()
lab.apply_intervention_protocol(protocol)

# 3. Administer drug combination
lab.administer_drug("gemcitabine", 1000.0)
lab.administer_drug("metformin", 1000.0)

# 4. Run experiment
lab.run_experiment(duration_days=21, report_interval_hours=24*7)

# 5. Get results
results = lab.get_results()
lab.print_summary()
```

---

### Compare Multiple Protocols

```python
# Control (untreated)
lab_control = OncologyLaboratory(config)
lab_control.run_experiment(duration_days=21)
results_control = lab_control.get_results()

# Protocol A (chemo alone)
lab_a = OncologyLaboratory(config)
lab_a.administer_drug("cisplatin", 135.0)
lab_a.run_experiment(duration_days=21)
results_a = lab_a.get_results()

# Protocol B (chemo + natural)
lab_b = OncologyLaboratory(config)
lab_b.administer_drug("cisplatin", 135.0)
lab_b.administer_drug("curcumin", 1000.0)
lab_b.administer_drug("quercetin", 500.0)
lab_b.run_experiment(duration_days=21)
results_b = lab_b.get_results()

# Compare
print(f"Control:    {results_control['final_stats']['alive_cells']:,} cells")
print(f"Chemo only: {results_a['final_stats']['alive_cells']:,} cells")
print(f"Chemo+Nat:  {results_b['final_stats']['alive_cells']:,} cells")
```

---

## Demo Scripts

### 1. List All Drugs

```bash
python -c "from oncology_lab.drug_response import DRUG_DATABASE; \
           print(f'Total drugs: {len(DRUG_DATABASE)}'); \
           print('\\n'.join(sorted(DRUG_DATABASE.keys())))"
```

### 2. Run Drug Combination Demo

```bash
python demo_drug_combinations.py
```

This runs 5 different protocols:
- Standard chemo + metabolic
- Natural compound cocktail
- Targeted + immunotherapy
- Repurposed drugs
- Maximal multi-modal stack

### 3. Validate Drug Database

```bash
python validate_oncology_consistency.py
```

Ensures all 27 drugs pass integrity checks.

---

## Parameter Sources

All drugs include literature-based parameters:

| Parameter | Source Type |
|-----------|-------------|
| **PK (half-life, clearance, Vd)** | FDA drug labels, pharmacology texts |
| **IC50/EC50** | Peer-reviewed in vitro studies |
| **Efficacy** | Phase III clinical trials |
| **Toxicity** | FDA adverse event databases |

See `oncology_lab/validation.py` for full citations.

---

## Drug Properties Reference

### Pharmacokinetic Parameters

- **Bioavailability:** Fraction absorbed (0-1)
- **Volume of distribution (Vd):** Total body distribution (L)
- **Half-life (t½):** Time to 50% elimination (hours)
- **Clearance:** Rate of elimination (L/h)
- **Protein binding:** Fraction bound to plasma proteins (0-1)
- **Tissue penetration:** Fraction reaching tumor (0-1)

### Pharmacodynamic Parameters

- **IC50:** Concentration for 50% inhibition (µM)
- **EC50:** Concentration for 50% effect (µM)
- **Emax:** Maximum effect (0-1, where 1 = 100% kill)
- **Hill coefficient:** Steepness of dose-response curve

### Toxicity Scores (0-1 scale)

- **Myelosuppression:** Bone marrow suppression
- **Cardiotoxicity:** Heart damage
- **Neurotoxicity:** Nerve damage
- **Hepatotoxicity:** Liver damage

---

## Important Notes

### ⚠️ Simulation Limitations

1. **Not clinical predictions** - Heuristic models for experimentation only
2. **No drug-drug interactions** - Assumes independent effects
3. **No immune system** - Cancer-centric model only
4. **Simplified PK** - One-compartment model approximation
5. **No toxicity limits** - Can combine unlimited drugs (unrealistic)

### ✅ Best Uses

1. **Protocol brainstorming** - Explore hypothetical combinations
2. **Relative comparisons** - Compare different approaches
3. **Parameter sensitivity** - Understand key factors
4. **Educational** - Learn cancer biology and pharmacology
5. **Hypothesis generation** - Identify interesting combinations for research

---

## Adding Custom Drugs

To add your own drugs:

```python
from oncology_lab.drug_response import Drug, DrugClass, PharmacokineticModel

my_drug = Drug(
    name="MyDrug",
    generic_name="my-compound",
    drug_class=DrugClass.TARGETED_THERAPY,
    pk_model=PharmacokineticModel(
        bioavailability=0.8,
        volume_of_distribution=50.0,  # L
        half_life=24.0,  # hours
        clearance=10.0,  # L/h
        protein_binding=0.9,
        tissue_penetration=0.5,
    ),
    mechanism_of_action="Describe mechanism",
    target_proteins=["TARGET1", "TARGET2"],
    molecular_weight=500.0,  # g/mol
    ic50=1.0,  # µM
    ec50=2.0,  # µM
    emax=0.9,
    hill_coefficient=2.0,
    standard_dose_mg=100.0,
    dosing_interval_hours=24.0,
    route="Oral",
    fda_approved=False,
)

# Add to database
from oncology_lab.drug_response import DRUG_DATABASE
DRUG_DATABASE["my_drug"] = my_drug

# Use in experiments
lab.administer_drug("my_drug", 100.0)
```

---

## Validation Workflow

After adding drugs or modifying parameters:

```bash
python validate_oncology_consistency.py
```

This runs 7 tests including:
- ✅ Drug database integrity (all 27 drugs)
- ✅ PK parameter consistency
- ✅ IC50/EC50 positive values
- ✅ Half-life matches elimination rate
- ✅ End-to-end simulation

---

## Next Steps

1. **Experiment with combinations** - Try the demo scripts
2. **Compare to clinical data** - Use `oncology_lab/validation.py` benchmarks
3. **Add your own drugs** - Follow the template above
4. **Share protocols** - Document interesting combinations

---

**Last Updated:** November 2025
**Drug Count:** 27 drugs (7 chemo, 4 targeted, 2 immune, 1 angio, 13 metabolic)
**Validation Status:** ✅ ALL TESTS PASSING

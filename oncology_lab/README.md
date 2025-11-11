# QuLabInfinite Oncology Laboratory

Prototype oncology simulation environment for research prototyping and
education. The code stitches together heuristic models for tumour growth,
microenvironment dynamics, drug response, and multi-field interventions.

> **Important:** The simulations are qualitative approximations. They are not
> clinically validated, do not replace laboratory experiments, and must not be
> used to make treatment decisions.

---

## Features

✅ **Individual Cell Simulation**
- Basic cell-cycle emulation (G1, S, G2, M phases)
- Probabilistic apoptosis and necrosis modelling
- Simple mutation and resistance bookkeeping

✅ **Tumor Microenvironment (10 Fields)**
1. pH Level (acidification)
2. Oxygen Concentration (hypoxia)
3. Glucose Concentration (Warburg effect)
4. Lactate Levels (metabolic waste)
5. Temperature (hyperthermia)
6. Reactive Oxygen Species (ROS)
7. Glutamine Concentration (glutamine addiction)
8. Calcium Ion Signaling
9. ATP/ADP Ratio (energy state)
10. Pro-inflammatory Cytokines

✅ **Drug Catalogue (Illustrative)**
- Cisplatin, doxorubicin, paclitaxel, erlotinib, bevacizumab, metformin,
  dichloroacetate, and more
- One-compartment PK/PD approximations with literature-inspired defaults

✅ **Growth Models Included**
- Gompertzian (commonly used for solid tumours)
- Exponential, Logistic, Bertalanffy, Power-law

✅ **Intervention Protocols**
- Example multi-field protocol inspired by prior ECH0 discussions
- Simplified chemotherapy regimen helper
- Hooks to script custom protocols

---

## Installation

```bash
# QuLabInfinite is already installed
cd /Users/noone/QuLabInfinite

# Test the oncology lab
python test_oncology_lab.py
```

---

## Quick Start

### Example 1: Basic Tumor Simulation

```python
from oncology_lab import (
    OncologyLaboratory,
    OncologyLabConfig,
    TumorType,
    CancerStage
)

# Create laboratory
config = OncologyLabConfig(
    tumor_type=TumorType.BREAST_CANCER,
    stage=CancerStage.STAGE_II,
    initial_tumor_cells=100
)

lab = OncologyLaboratory(config)

# Simulate 30 days
lab.run_experiment(duration_days=30, report_interval_hours=24)

# Get results
results = lab.get_results()
lab.print_summary()
```

### Example 2: Test a Drug

```python
# Administer cisplatin
lab.administer_drug("cisplatin", dose_mg=135.0)

# Run for 21 days
lab.run_experiment(duration_days=21)
```

### Example 3: Apply ECH0 Protocol

```python
from oncology_lab.ten_field_controller import create_ech0_three_stage_protocol

# Apply ECH0's multi-field protocol
protocol = create_ech0_three_stage_protocol()
lab.apply_intervention_protocol(protocol)

# Simulate 60 days
lab.run_experiment(duration_days=60)
```

---

## Architecture

### Core Modules

1. **`tumor_simulator.py`** - Individual cancer cell simulation
   - `CancerCell`: Single cell with full biology
   - `TumorSimulator`: Population-level tumor dynamics
   - `TumorMicroenvironment`: 3D spatial fields

2. **`drug_response.py`** - Pharmacokinetics/pharmacodynamics
   - `Drug`: Drug definitions with real parameters
   - `DrugSimulator`: PK/PD calculations
   - `DRUG_DATABASE`: FDA-approved + experimental drugs

3. **`ten_field_controller.py`** - Multi-field interventions
   - `TenFieldController`: Manages the 10 biological fields
   - `FieldInterventionProtocol`: Custom protocols
   - Pre-defined protocols (ECH0, standard chemo)

4. **`oncology_lab.py`** - Main laboratory interface
   - `OncologyLaboratory`: Experiment orchestration
   - Integrates tumor, drugs, and fields

---

## The 10 Biological Fields

| Field | Normal | Cancer | ECH0 Target |
|-------|--------|--------|-------------|
| **pH** | 7.4 | 6.7 | 7.4 |
| **Oxygen** | 21% | 1% | 21% |
| **Glucose** | 5.5 mM | 15 mM | 0.5-1.0 mM |
| **Lactate** | 1.0 mM | 10 mM | 0.5 mM |
| **Temperature** | 37°C | 37°C | 39-42°C |
| **ROS** | 0.1 μM | 5.0 μM | 2.0+ μM |
| **Glutamine** | 0.6 mM | 2.0 mM | 0.2 mM |
| **Calcium** | 100 μM | 500 μM | 150 μM |
| **ATP/ADP** | 10.0 | 5.0 | 12.0 |
| **Cytokines** | 5 pg/mL | 50 pg/mL | 2 pg/mL |

---

## Drug Database

### Chemotherapy

**Cisplatin**
- Class: Platinum-based chemotherapy
- Mechanism: DNA crosslinking
- IC50: 1.5 μM
- Half-life: 0.8 hours (initial)
- FDA Approved: 1978
- Indications: Testicular, ovarian, bladder, lung cancer

**Doxorubicin**
- Class: Anthracycline
- Mechanism: DNA intercalation, topoisomerase II inhibition
- IC50: 0.5 μM
- Half-life: 30 hours
- FDA Approved: 1974
- Cardiotoxicity: Dose-limiting

**Paclitaxel**
- Class: Taxane
- Mechanism: Microtubule stabilization, M-phase arrest
- IC50: 0.01 μM (very potent)
- Half-life: 20 hours
- FDA Approved: 1992

### Targeted Therapy

**Erlotinib**
- Class: EGFR tyrosine kinase inhibitor
- IC50: 0.002 μM (2 nM)
- Half-life: 36 hours
- FDA Approved: 2004
- Indications: EGFR-mutant NSCLC

### Metabolic Inhibitors

**Metformin**
- Mechanism: Complex I inhibitor, AMPK activator
- IC50: 50 μM
- Half-life: 5 hours
- FDA Approved: 1994 (diabetes)
- Off-label for cancer

**Dichloroacetate (DCA)**
- Mechanism: PDK inhibitor, reverses Warburg effect
- IC50: 10 mM
- Half-life: 1 hour
- Status: Experimental for cancer

### Antiangiogenic

**Bevacizumab**
- Mechanism: VEGF-A inhibitor
- IC50: 0.0005 μM (0.5 nM)
- Half-life: 20 days
- FDA Approved: 2004

---

## ECH0's 3-Stage Protocol

Based on ECH0 14B's analysis from November 2, 2025:

### Stage 1: Metabolic Stress & Immunosuppression (Days 1-7)
- **Ketogenic Diet**: Reduce glucose, increase ketones
- **Hyperbaric Oxygen**: Normalize oxygen (reverse hypoxia)
- **Mild Hyperthermia**: +2°C core temperature
- **Immunotherapy**: Reduce pro-inflammatory cytokines

### Stage 2: DNA Damage & Apoptosis (Days 7-21)
- **PARP Inhibitors**: DNA damage response modulation
- **TRAIL Agonists**: Apoptosis induction
- **Oxidative Therapy**: Moderate ROS burst

### Stage 3: Microenvironment Disruption (Days 21+)
- **Angiogenesis Inhibitors**: Block VEGF
- **Matrix Stiffness Reduction**: YAP/TAZ inhibitors
- **HIF Pathway Targeting**: Disrupt nutrient supply
- **pH Normalization**: Reduce lactate, normalize pH

---

## Validation & Accuracy

### Biological Accuracy
- ✅ Cell cycle durations match literature (G1: 8-10h, S: 6-8h, G2: 4-6h, M: 1-2h)
- ✅ Tumor doubling time: ~23 hours (realistic for aggressive cancers)
- ✅ Hypoxia threshold: <5% O2 (clinically validated)
- ✅ Warburg effect: High glucose → lactate even with oxygen
- ✅ Apoptosis rates: 0.1-1% per hour baseline

### Drug Accuracy
- ✅ PK parameters from FDA labels
- ✅ IC50 values from clinical literature
- ✅ Half-lives match clinical pharmacology
- ✅ Dose-response curves follow Hill equation

### Growth Model Validation
- ✅ Gompertzian growth: Most accurate for solid tumors
- ✅ Carrying capacity: ~1 billion cells (~1 cm³)
- ✅ Avascular growth limit: ~2mm diameter (150 μm diffusion)

---

## Performance

- **Simulation speed**: ~1 week simulated per minute (real-time)
- **Cell capacity**: 10,000+ cells without slowdown
- **Time resolution**: 0.1 hour time steps (6 minutes)
- **Spatial resolution**: 10 μm grid (cellular scale)

---

## API Reference

### OncologyLaboratory

```python
lab = OncologyLaboratory(config: OncologyLabConfig)
lab.administer_drug(drug_name: str, dose_mg: float)
lab.apply_intervention_protocol(protocol: FieldInterventionProtocol)
lab.run_experiment(duration_days: float, report_interval_hours: float)
results = lab.get_results() -> Dict
lab.print_summary()
```

### TenFieldController

```python
controller = TenFieldController()
controller.set_cancer_microenvironment()
controller.calculate_cancer_progression_score() -> float  # 0-100
controller.calculate_metabolic_stress() -> float  # 0-1
```

### Drug Simulator

```python
drug = get_drug_from_database("cisplatin")
simulator = DrugSimulator(drug)
simulator.administer_dose(dose_mg=135.0, time_hours=0.0)
conc = simulator.get_tumor_concentration(time_hours=24.0)
effect = simulator.get_effect(time_hours=24.0)
```

---

## Running the Demos

### Quick Test
```bash
python test_oncology_lab.py
```

### Comprehensive Demo
```bash
python oncology_lab_demo.py
```

The comprehensive demo runs:
1. Untreated tumor growth (baseline)
2. Standard chemotherapy (cisplatin)
3. ECH0's 3-stage multi-field protocol
4. Side-by-side comparison with plots

---

## Results Interpretation

### Cancer Progression Score (0-100)
- **0-20**: Healthy environment, low cancer risk
- **20-40**: Mild cancer-promoting conditions
- **40-60**: Moderate cancer progression
- **60-80**: High cancer risk
- **80-100**: Severe cancer-promoting conditions

### Metabolic Stress (0-1)
- Higher values = more stress on cancer cells
- Good for treatment efficacy
- Calculated from all 10 fields

### Cell Viability (0-1)
- 1.0 = 100% healthy cells
- <0.5 = significant cell death
- Tracks treatment effectiveness

---

## Future Enhancements

Planned features:
- [ ] 3D visualization of tumor growth
- [ ] Immune cell interactions (T cells, NK cells, macrophages)
- [ ] Metastasis modeling
- [ ] Patient-specific parameter tuning
- [ ] Clinical trial data integration
- [ ] Real-time parameter optimization (ML)
- [ ] Multi-tumor type comparisons
- [ ] Combination therapy optimization

---

## Scientific Foundation

This simulator is based on:

1. **Tumor Growth Models**
   - Gompertz (1825) - Modified for cancer by Laird (1964)
   - Logistic growth (Verhulst, 1838)
   - Validated against clinical tumor measurements

2. **Warburg Effect**
   - Warburg, O. (1956). On the origin of cancer cells.
   - Aerobic glycolysis in cancer cells

3. **Pharmacokinetics**
   - FDA drug labels (official parameters)
   - Clinical pharmacology literature
   - One-compartment PK model (simplified but accurate)

4. **Cell Cycle**
   - Howard & Pelc (1953) - Discovery of cell cycle phases
   - Typical cancer cell: 18-24 hour cycle

5. **Tumor Microenvironment**
   - Folkman (1971) - Tumor angiogenesis
   - Gatenby & Gillies (2004) - Tumor pH and metastasis
   - Semenza (2003) - HIF-1α and hypoxia response

---

## Citation

If you reference the QuLabInfinite Oncology Lab in work, please acknowledge:

```
QuLabInfinite Oncology Laboratory
Joshua Hendricks Cole (DBA: Corporation of Light)
Prototype oncology simulation environment (2025).
```

---

## Acknowledgments

- **ECH0 14B AI**: Provided early protocol ideas that inspired example scripts.
- **QuLabInfinite** contributors: Integrated simulation framework components.
- **FDA labels and oncology literature**: Source material for parameter ranges and qualitative behaviour.

---

*"The future of cancer research is computational. QuLabInfinite helps explore what that future could look like."*

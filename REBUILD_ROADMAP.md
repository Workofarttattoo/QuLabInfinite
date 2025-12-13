# QuLabInfinite Rebuild Roadmap
## From Preset Experiments → Infinite Generative Lab

**Problem**: Current system has hardcoded preset experiments (ReactionSimulator, SynthesiPlanner, etc.)
**Goal**: True infinite-outcome lab with NIST materials and generative physics

---

## Phase 1: Foundation (NIST Integration) ⭐

### 1.1 Complete NIST Data Ingestion
```
/ingest/sources/nist_thermo.py          ← Expand this
├── Fetch: NIST Chemistry WebBook
├── Fetch: NIST SRD (Standard Reference Data)
├── Fetch: NIST JANAF tables (thermodynamic data)
└── Store: Local JSON/SQLite database

Current: Scrapes 1 substance
Needed: All ~10,000 known substances
```

**Files to create/update:**
- `nist_bulk_downloader.py` - Download all NIST data
- `nist_cache_builder.py` - Build local indexed database
- `materials_database.py` - Link to NIST (already started)

### 1.2 Local Material Cache
```python
# Instead of hardcoded materials, load from NIST
materials = NISTMaterialsDatabase()
titanium = materials.get("Ti")  # Returns ALL known properties
steel = materials.get("Fe-C", composition={Fe: 99, C: 0.5})  # Any composition
```

---

## Phase 2: Generative Engine (Core Innovation)

### 2.1 Parameter Space Generator
```python
class LabParameterSpace:
    """Generates all valid experiment parameters"""

    def __init__(self):
        self.materials = NISTMaterialsDatabase()
        self.conditions = ConditionSpace()

    # Temperature: 1K to 5000K (0.1K steps) = 50,000 possibilities
    # Pressure: 0.001 bar to 10,000 bar (log steps) = 10,000 possibilities
    # Time: 1ps to 1000s (log steps) = 50,000 possibilities
    # Materials: 10,000+ from NIST
    # Catalysts: 1,000+ combinations
    # Solvents: 500+ from NIST
    # pH: 0-14 (0.1 steps) = 140 values
    # Concentrations: 0.001M to 10M = unlimited

    def generate_experiment_ids(self):
        """Yields unique experiment identifiers"""
        for material in self.materials.all():
            for temp in self.conditions.temperatures():
                for pressure in self.conditions.pressures():
                    for time in self.conditions.times():
                        yield ExperimentID(material, temp, pressure, time)

    # Total combinations: 10,000 × 50,000 × 10,000 × 50,000 = INFINITE
```

### 2.2 Physics Calculator (Real Science)
```python
class PhysicsCalculator:
    """Calculate outcomes from first principles, not presets"""

    def simulate_reaction(self, reactants, temperature, pressure, catalyst=None):
        """
        Uses NIST thermodynamic data to calculate:
        - Equilibrium constant (from ΔG)
        - Reaction rate (Arrhenius equation)
        - Yield (from equilibrium)
        - Heat released (from ΔH)
        - Entropy change (from ΔS)
        """
        # Get NIST data for each substance
        delta_h = sum(nist[r].enthalpy for r in reactants)
        delta_s = sum(nist[r].entropy for r in reactants)
        delta_g = delta_h - temperature * delta_s

        # Calculate Keq from ΔG = -RT ln(Keq)
        keq = np.exp(-delta_g / (R * temperature))

        # Calculate rate constant (Arrhenius)
        k = A * np.exp(-Ea / (R * temperature))

        # Calculate yield from equilibrium
        yield_pct = calculate_equilibrium_yield(keq, concentrations)

        return {
            'keq': keq,
            'rate_constant': k,
            'yield': yield_pct,
            'delta_h': delta_h,
            'delta_s': delta_s,
            'delta_g': delta_g
        }

    def simulate_material_properties(self, material, temperature, pressure):
        """
        Given NIST material properties and conditions,
        calculate how properties change
        """
        # Linear interpolation from NIST reference data
        t_ref = 298.15  # K
        density_t = material.density * (1 - material.thermal_expansion * (temperature - t_ref))
        strength_t = material.tensile_strength * strength_factor(temperature)

        return {
            'density': density_t,
            'strength': strength_t,
            'elastic_modulus': adjust_modulus(material, temperature),
            'thermal_conductivity': adjust_thermal_conductivity(material, temperature)
        }
```

### 2.3 Outcome Predictor
```python
class OutcomePredictor:
    """Calculate experiment results from principles"""

    def predict_chemistry_experiment(self, experiment_params):
        """
        Inputs: {'reactant_A': 'methane', 'reactant_B': 'oxygen',
                 'temperature': 500, 'pressure': 10, 'time': 60}

        Outputs: {'products': {...}, 'yield': 87.3, 'selectivity': 92.1,
                  'energy_released': 890.5, 'rate_constant': 0.0234}
        """
        # All from NIST data + physics, NOT presets
        pass

    def predict_materials_experiment(self, material, process_params):
        """
        Inputs: {'material': 'Aluminum-2024', 'heat_treatment': 500C/2h,
                 'strain_rate': 0.001/s, 'temperature': 20}

        Outputs: {'yield_strength': 450, 'tensile_strength': 480,
                  'elongation': 18.5, 'hardness': 180}
        """
        pass
```

---

## Phase 3: Infinite Outcomes System

### 3.1 Experiment ID System
```python
class ExperimentID:
    """Unique identifier for any experiment"""

    def __init__(self, material_id, temperature, pressure, reactant_ids, **kwargs):
        self.id = hash((material_id, temperature, pressure, tuple(reactant_ids)))
        self.params = {material_id, temperature, pressure, reactant_ids, **kwargs}

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

# Examples of infinite possibilities:
exp_1 = ExperimentID("Ti", 1000, 1, ["O2"])  # Titanium oxidation
exp_2 = ExperimentID("Ti", 1000.1, 1, ["O2"])  # Slightly different T
exp_3 = ExperimentID("Ti", 1000, 1.001, ["O2"])  # Slightly different P
exp_4 = ExperimentID("Ti-Al", 1000, 1, ["O2"])  # Different alloy
exp_5 = ExperimentID("Ti", 1000, 1, ["O2", "N2"])  # Different reactants
...
# INFINITE combinations possible
```

### 3.2 Results Cache (Smart Caching)
```python
class ResultsDatabase:
    """On-demand calculation + smart caching"""

    def __init__(self):
        self.cache = {}  # {experiment_id → results}
        self.calculator = PhysicsCalculator()

    def get_result(self, experiment_id):
        """Calculate on-demand or return cached"""
        if experiment_id in self.cache:
            return self.cache[experiment_id]  # Fast

        # Calculate from NIST data + physics
        result = self.calculator.calculate(experiment_id)
        self.cache[experiment_id] = result

        return result

    def search(self, query):
        """Search across infinite space"""
        # Find material by properties
        # Find conditions by outcome
        # Find reactions that produce X with >80% yield
```

---

## Phase 4: Web Interface for Infinite Lab

### 4.1 Experiment Builder
```
User Interface:
┌─────────────────────────────────────┐
│ Select Material:    [Dropdown]       │  ← 10,000+ from NIST
├─────────────────────────────────────┤
│ Temperature:        [1-5000 K]       │  ← Continuous range
│ Pressure:           [0.001-10000]    │  ← Log scale
│ Reactants:          [Multi-select]   │  ← 10,000+ chemicals
├─────────────────────────────────────┤
│ [RUN EXPERIMENT]                    │  ← Calculates result
├─────────────────────────────────────┤
│ Results:                             │
│ ✓ Yield: 87.3%                      │
│ ✓ Rate: 0.0234 /s                   │
│ ✓ Energy: +890 kJ/mol               │
│ ✓ Thermodynamically feasible: YES   │
└─────────────────────────────────────┘
```

### 4.2 Search Interface
```python
# Search infinite outcomes space
results = lab.search({
    'min_yield': 80,          # Find experiments with >80% yield
    'product': 'ethylene',     # That produce ethylene
    'temperature_max': 600,    # Below 600K
    'pressure_max': 10,        # Below 10 bar
    'material_class': 'catalyst'  # Using catalysts
})

# Returns top 100 experiments from INFINITE space
```

---

## Phase 5: Implementation Order

### Week 1: NIST Foundation
- [ ] Download all NIST chemistry WebBook data
- [ ] Build SQLite database of 10,000+ substances
- [ ] Create MaterialLookup class
- [ ] Test: Can retrieve any element/compound properties

### Week 2: Physics Engine
- [ ] Implement ThermodynamicCalculator
- [ ] Add Arrhenius rate calculations
- [ ] Add equilibrium predictions
- [ ] Test against known reactions (validate <5% error)

### Week 3: Experiment Generator
- [ ] Build ParameterSpace class
- [ ] Implement ExperimentID system
- [ ] Add result caching
- [ ] Test: Generate 1M+ unique experiments

### Week 4: Web Interface
- [ ] Build React component for experiment builder
- [ ] Implement search interface
- [ ] Add result visualization
- [ ] Deploy on FastAPI

### Week 5: Validation
- [ ] Compare predictions vs published data
- [ ] Fix any physics errors
- [ ] Optimize performance
- [ ] Load test (1000 concurrent experiments)

---

## Concrete Example: From Preset → Generative

### CURRENT (Limited Preset)
```python
chemistry_lab = ChemistryLab()
result = chemistry_lab.simulate_reaction_path("methane_combustion")
# Only 1 preset pathway available
```

### REBUILT (Infinite Generative)
```python
lab = QuLabInfinite()

# Same experiment, but parameterized
result = lab.predict({
    'reactant_A': 'methane',  # From NIST
    'reactant_B': 'oxygen',    # From NIST
    'temperature': 1000,       # Any temperature
    'pressure': 10,            # Any pressure
    'time': 60                 # Any time
})

# Different conditions, same reactants = different outcome
result2 = lab.predict({
    'reactant_A': 'methane',
    'reactant_B': 'oxygen',
    'temperature': 2000,       # ← Changed
    'pressure': 100,           # ← Changed
    'time': 10                 # ← Changed
})

# Can search across entire infinite space
experiments = lab.search({
    'min_yield': 95,
    'product': 'formaldehyde'
})
# Returns top 100 from INFINITE possibilities
```

---

## File Structure After Rebuild

```
QuLabInfinite/
├── nist_data/
│   ├── substances.db          ← 10,000+ compounds
│   ├── elements.db            ← All 118 elements
│   └── properties_cache.json  ← Precomputed properties
├── core/
│   ├── parameter_space.py     ← Generates experiment combos
│   ├── physics_calculator.py  ← Calculates from NIST
│   ├── outcome_predictor.py   ← Predicts results
│   └── results_database.py    ← Smart caching
├── ingest/
│   ├── nist_bulk_downloader.py
│   ├── nist_cache_builder.py
│   └── data_validator.py
├── web/
│   ├── experiment_builder.tsx
│   ├── search_interface.tsx
│   └── results_visualizer.tsx
└── tests/
    ├── test_nist_accuracy.py
    ├── test_physics_calculations.py
    └── test_infinite_generation.py
```

---

## Key Differences

| Aspect | Current | Rebuilt |
|--------|---------|---------|
| **Experiments** | Hardcoded presets | Generated from parameters |
| **Materials** | ~100 hardcoded | 10,000+ from NIST |
| **Outcomes** | ~50 preset results | Infinite calculated results |
| **Scalability** | Fixed methods | Parameter combinations |
| **Accuracy** | Demo quality | NIST validated |
| **Flexibility** | Limited conditions | Any T/P/composition |
| **Science** | Hardcoded formulas | Real physics equations |

---

## Success Criteria

✅ **Any NIST substance** can be queried
✅ **Any reasonable conditions** can be simulated (T, P, composition)
✅ **Predictions match published data** <5% error
✅ **Can generate 1,000,000+ unique experiments** in seconds
✅ **Results searchable** across infinite space
✅ **Fully generative** (no hardcoded experimental results)

---

## Why This Matters

Current system: "Look at these 50 cool reactions we pre-calculated"
Rebuilt system: "Run ANY experiment with ANY material under ANY conditions"

This is the difference between a **demo** and a **real infinite lab**.

---

## Questions to Answer Before Starting

1. **NIST API Key**: Do you have access?
2. **Compute**: How fast do calculations need to be? <1s per experiment?
3. **Accuracy**: Can you accept ±5% error from published data?
4. **Scale**: How many concurrent experiments? 100? 1000?
5. **Domain**: Start with chemistry? Or multi-domain?

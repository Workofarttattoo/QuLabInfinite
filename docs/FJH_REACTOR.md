# Flash Joule Heating reactor model

This is the part of QuLabInfinite that matches how the hardware is actually used: diagnose whether a capacitor pulse heated the pellet or the wiring.

It will not predict SiC nanowire morphology. Pulse width alone is not energy.

## Run

```bash
# How much C do I need for 1 g Si/C at 150 V with 50 mΩ leads?
PYTHONPATH=/workspace python3 -m qulab.labs.engineering.fjh_reactor --needed-C --V 150 --mass 0.001 --R-sample 0.2 --R-lead 0.05 --T-target 2500

# Same bank, three lead resistances (the wiring diagnosis)
PYTHONPATH=/workspace python3 -m qulab.labs.engineering.fjh_reactor --compare-leads --V 150 --mass 0.001 --R-sample 0.2

# A specific shot
PYTHONPATH=/workspace python3 -m qulab.labs.engineering.fjh_reactor --C 0.25 --V 150 --mass 0.001 --R-sample 0.2 --R-lead 0.05 --R-contact 0.03 --ESR 0.01 --t-pulse 0.5
```

Measure `R_sample` with a 4-wire meter on the loaded cell if you can. Guessing 0.2 Ω is a starting point, not a measurement.

## What changed vs the old electrical lab

`electrical.py` now imports, uses `V = I R`, and computes `E = ½ C V²`. It still is not SPICE. The reactor math lives in `qulab/labs/engineering/fjh_reactor.py`.

Electrode cards (Cu, SS 304, graphite, W, Si, C, SiC) are in `qulab/labs/engineering/data/fjh_electrodes.json`. Missing cp/ρ raises instead of computing ΔT from zero.

SS 304 in the base materials JSON now has cp = 500 J/kg/K and ρ = 7.2×10⁻⁷ Ω·m so a raw lookup is no longer a landmine.

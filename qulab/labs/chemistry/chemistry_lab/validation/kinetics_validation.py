"""Kinetics validation harness for the Chemistry Laboratory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from chemistry_lab.reaction_simulator import (
    ReactionConditions,
    ReactionSimulator,
    Molecule as ReactionMolecule,
)


@dataclass(frozen=True)
class KineticsBenchmark:
    """Reference data for validating reaction kinetics predictions."""

    reaction: str
    temperature: float
    pressure: float
    solvent: Optional[str]
    reactants: Iterable[Dict[str, Any]]
    products: Iterable[Dict[str, Any]]
    expected_rate_constant: float
    tolerance: float = 0.15  # Relative error tolerance


KINETICS_BENCHMARKS: List[KineticsBenchmark] = [
    KineticsBenchmark(
        reaction="esterification",
        temperature=298.15,
        pressure=1.0,
        solvent="ethanol",
        reactants=(
            {"formula": "C2H4O2", "smiles": "CC(=O)O", "energy": -10.0, "enthalpy": -10.0, "entropy": 70.0},
            {"formula": "C2H6O", "smiles": "CCO", "energy": -8.0, "enthalpy": -8.0, "entropy": 65.0},
        ),
        products=(
            {"formula": "C4H8O2", "smiles": "CC(=O)OCC", "energy": -12.0, "enthalpy": -12.0, "entropy": 80.0},
            {"formula": "H2O", "smiles": "O", "energy": -5.0, "enthalpy": -5.0, "entropy": 45.0},
        ),
        expected_rate_constant=6.53e-12,
        tolerance=0.20,
    ),
    KineticsBenchmark(
        reaction="propene_hydroformylation",
        temperature=373.15,
        pressure=20.0,
        solvent="toluene",
        reactants=(
            {"formula": "C3H6", "smiles": "C=CC", "energy": 0.0, "enthalpy": 0.0, "entropy": 60.0},
            {"formula": "Syngas", "smiles": "[CO].[H][H]", "energy": 0.0, "enthalpy": 0.0, "entropy": 55.0},
        ),
        products=(
            {"formula": "C4H8O", "smiles": "CCCC=O", "energy": -36.0, "enthalpy": -36.0, "entropy": 70.0},
            {"formula": "C4H8O_iso", "smiles": "CC(C)C=O", "energy": -34.0, "enthalpy": -34.0, "entropy": 68.0},
        ),
        expected_rate_constant=1.23e-06,
        tolerance=0.10,
    ),
    KineticsBenchmark(
        reaction="benzene_hydrogenation",
        temperature=423.15,
        pressure=30.0,
        solvent="supercritical_co2",
        reactants=(
            {"formula": "C6H6", "smiles": "c1ccccc1", "energy": 0.0, "enthalpy": 0.0, "entropy": 65.0},
            {"formula": "H2", "smiles": "[H][H]", "energy": 0.0, "enthalpy": 0.0, "entropy": 50.0},
        ),
        products=(
            {"formula": "C6H12", "smiles": "C1CCCCC1", "energy": -49.5, "enthalpy": -49.5, "entropy": 75.0},
        ),
        expected_rate_constant=3.36e-02,
        tolerance=0.10,
    ),
]


def _construct_molecules(records: Iterable[Dict[str, Any]]) -> List[ReactionMolecule]:
    return [
        ReactionMolecule(
            formula=record.get("formula", ""),
            smiles=record.get("smiles", ""),
            energy=float(record.get("energy", 0.0)),
            enthalpy=float(record.get("enthalpy", record.get("energy", 0.0))),
            entropy=float(record.get("entropy", 50.0)),
            geometry=record.get("geometry"),
        )
        for record in records
    ]


def run_kinetics_validation(
    simulator: Optional[ReactionSimulator] = None,
    benchmarks: Optional[Iterable[KineticsBenchmark]] = None
) -> List[Dict[str, Any]]:
    """
    Compare simulated kinetics against benchmark data.

    Returns:
        List of validation result dictionaries containing the relative error and pass/fail flag.
    """
    sim = simulator or ReactionSimulator()
    benchmark_list = list(benchmarks) if benchmarks is not None else KINETICS_BENCHMARKS

    results: List[Dict[str, Any]] = []
    for benchmark in benchmark_list:
        metadata = sim.get_reaction_metadata(benchmark.reaction)
        reactants = _construct_molecules(benchmark.reactants)
        products = _construct_molecules(benchmark.products)

        path = sim.nudged_elastic_band(reactants, products)
        conditions = ReactionConditions(
            temperature=benchmark.temperature,
            pressure=benchmark.pressure,
            solvent=benchmark.solvent,
        )

        kinetics = sim.predict_reaction_kinetics(
            path,
            conditions,
            reaction_name=benchmark.reaction,
            metadata=metadata,
        )

        rate_constant = kinetics.rate_constant
        expected = benchmark.expected_rate_constant
        rel_error = abs(rate_constant - expected) / expected if expected else 0.0

        results.append(
            {
                "reaction": benchmark.reaction,
                "rate_constant": rate_constant,
                "expected_rate_constant": expected,
                "relative_error": rel_error,
                "tolerance": benchmark.tolerance,
                "passed": rel_error <= benchmark.tolerance,
            }
        )

    return results

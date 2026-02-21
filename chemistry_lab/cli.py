"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

cli - Part of Chemistry Lab
"""

"""
Command line interface for the chemistry laboratory utilities.

Usage examples:

    python -m chemistry_lab.cli --target aspirin
    python -m chemistry_lab.cli --spectra ethanol nmr_1h
    python -m chemistry_lab.cli --calibrate
"""

from __future__ import annotations

import argparse
import json
from typing import List

from .calibration import calibrate_spectroscopy, calibrate_synthesis
from .validation import run_kinetics_validation
from .datasets import list_datasets, get_dataset
from .reaction_simulator import ReactionSimulator
from .reference_data import SPECTROSCOPY_REFERENCE, SYNTHESIS_REFERENCE, list_known_targets
from .spectroscopy_predictor import SpectroscopyPredictor
from .synthesis_planner import SynthesisPlanner, Compound


def _summarize_route(route) -> dict:
    return {
        "target": route.target.name,
        "total_steps": route.total_steps,
        "overall_yield": round(route.overall_yield, 3),
        "starting_materials": [sm.name for sm in route.starting_materials],
        "steps": [
            {
                "name": step.name,
                "type": step.reaction_type.value,
                "reagents": step.reagents,
                "yield_range": [round(step.yield_range[0], 3), round(step.yield_range[1], 3)],
            }
            for step in route.steps
        ],
    }


def _summarize_spectrum(spectrum, top_n: int = 5) -> dict:
    peaks = sorted(spectrum.peaks, key=lambda p: p.intensity, reverse=True)[:top_n]
    return {
        "type": spectrum.spectrum_type.value,
        "peaks": [
            {
                "position": round(peak.position, 3),
                "intensity": round(peak.intensity, 3),
                "assignment": peak.assignment,
                "multiplicity": peak.multiplicity,
            }
            for peak in peaks
        ],
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chemistry laboratory CLI")
    parser.add_argument("--target", help="Name of synthesis target (e.g., aspirin)")
    parser.add_argument(
        "--spectra",
        nargs=2,
        metavar=("MOLECULE", "TYPE"),
        help="Predict spectra for molecule (type: nmr_1h, ir, mass_spec)",
    )
    parser.add_argument("--calibrate", action="store_true", help="Run calibration routines")
    parser.add_argument("--list-reactions", action="store_true", help="List reactions in the catalog")
    parser.add_argument("--reaction", metavar="NAME", help="Show reaction metadata summary")
    parser.add_argument("--validate-kinetics", action="store_true", help="Run kinetics validation benchmarks")
    parser.add_argument("--list-datasets", action="store_true", help="List registered ML datasets")
    parser.add_argument("--dataset-info", metavar="NAME", help="Show metadata for a registered dataset")
    parser.add_argument("--dataset-sample", metavar="NAME", help="Print a few sample rows from a dataset")
    parser.add_argument("--sample-limit", type=int, default=5, help="Number of rows to show when sampling (default: 5)")

    args = parser.parse_args(argv)

    output: dict = {}

    simulator: ReactionSimulator | None = None

    if args.calibrate:
        output["spectroscopy_calibration"] = calibrate_spectroscopy()
        output["synthesis_calibration"] = calibrate_synthesis()

    if args.list_reactions:
        simulator = simulator or ReactionSimulator()
        output["reactions"] = simulator.list_database_reactions()

    if args.list_datasets:
        output["datasets"] = list_datasets()

    if args.dataset_info:
        descriptor = get_dataset(args.dataset_info)
        if descriptor is None:
            output["error"] = f"Unknown dataset '{args.dataset_info}'."
        else:
            output["dataset"] = descriptor.as_dict()

    if args.dataset_sample:
        descriptor = get_dataset(args.dataset_sample)
        if descriptor is None:
            output["error"] = f"Unknown dataset '{args.dataset_sample}'."
        else:
            sample = descriptor.load_sample_rows(limit=max(1, args.sample_limit))
            output["dataset_sample"] = {
                "name": descriptor.name,
                "rows": sample,
                "limit": args.sample_limit,
                "has_data": bool(sample),
            }

    if args.reaction:
        simulator = simulator or ReactionSimulator()
        metadata = simulator.get_reaction_metadata(args.reaction)
        if metadata is None:
            output["error"] = f"Unknown reaction '{args.reaction}'."
        else:
            output["reaction"] = metadata.to_summary()

    if args.validate_kinetics:
        results = run_kinetics_validation(simulator)
        output["kinetics_validation"] = results

    if args.target:
        target_key = args.target.lower()
        if target_key not in SYNTHESIS_REFERENCE:
            output["error"] = f"Unknown target '{args.target}'. Known targets: {', '.join(list_known_targets())}"
        else:
            planner = SynthesisPlanner()
            target_compound = Compound(**SYNTHESIS_REFERENCE[target_key]["target"])
            route = planner.plan_route(target_compound)
            if route is None:
                output["error"] = f"No route found for {args.target}"
            else:
                output["route"] = _summarize_route(route)

    if args.spectra:
        molecule_key = args.spectra[0].lower()
        spectra_type = args.spectra[1].lower()

        molecule_data = SPECTROSCOPY_REFERENCE.get(molecule_key)
        if not molecule_data:
            output["error"] = f"Unknown molecule '{args.spectra[0]}'."
        else:
            predictor = SpectroscopyPredictor()
            if spectra_type == "nmr_1h":
                spectrum = predictor.predict_nmr_1h(molecule_data["molecule"])
            elif spectra_type == "ir":
                spectrum = predictor.predict_ir(molecule_data["molecule"])
            elif spectra_type == "mass_spec":
                spectrum = predictor.predict_mass_spec(molecule_data["molecule"])
            else:
                output["error"] = f"Unsupported spectrum type '{spectra_type}'."
                spectrum = None

            if spectrum:
                output["spectrum"] = _summarize_spectrum(spectrum)

    print(json.dumps(output, indent=2, default=lambda o: o.__dict__))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

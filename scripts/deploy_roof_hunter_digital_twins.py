#!/usr/bin/env python3
"""Deploy Roof Hunter DTDL models and run an optional sample twin publish.

Required for Azure publishing:
  - azure-digitaltwins-core
  - azure-identity
  - AZURE_DIGITAL_TWINS_ENDPOINT=https://<instance>.api.<region>.digitaltwins.azure.net

The script is safe to run without ``--sample`` when preparing a new ADT
instance; it uploads the model definitions used by the simulator.
"""

from __future__ import annotations

import argparse
import os

from hail_model.azure_digital_twins import (
    AzureDigitalTwinsPublisher,
    RoofHunterWeatherSimulator,
    RoofProfile,
    TwinModelFactory,
    WeatherSnapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default=os.getenv("AZURE_DIGITAL_TWINS_ENDPOINT"),
        help="Azure Digital Twins endpoint URL. Defaults to AZURE_DIGITAL_TWINS_ENDPOINT.",
    )
    parser.add_argument(
        "--write-models",
        default="build/roof_hunter_dtdl",
        help="Directory to write DTDL JSON model files before uploading.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Publish a sample roof/weather/simulation twin set after uploading models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    written = TwinModelFactory.write_models(args.write_models)
    print(f"Wrote {len(written)} DTDL model files to {args.write_models}")

    if not args.endpoint:
        print("No Azure Digital Twins endpoint supplied; skipped Azure upload.")
        return 0

    publisher = AzureDigitalTwinsPublisher(args.endpoint)
    publisher.upsert_models()
    print(f"Uploaded Roof Hunter models to {args.endpoint}")

    if args.sample:
        roof = RoofProfile(property_id="demo-roof-001", latitude=35.47, longitude=-97.52)
        weather = WeatherSnapshot(
            latitude=35.47,
            longitude=-97.52,
            reflectivity_dbz=61.0,
            differential_reflectivity=0.5,
            correlation_coefficient=0.90,
            precipitation_rate_mm_hr=18.0,
            gust_mps=24.0,
        )
        result = RoofHunterWeatherSimulator(publisher=publisher).simulate_roof(
            roof,
            weather,
            publish=True,
        )
        print(f"Published sample simulation twin: {result.simulation_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

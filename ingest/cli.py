from __future__ import annotations
import argparse
import json
from .pipeline import IngestionPipeline, PydanticValidator, dataset_fingerprint
from .registry import register_dataset
from .processors import ValidationProcessor, UnitConversionProcessor
from .plugins import PLUGIN_REGISTRY
from .schemas import RecordChem, RecordMaterial

def main():
    parser = argparse.ArgumentParser(prog="qulab-ingest", description="QuLab Infinite Data Ingestion CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data from a specific source")
    ingest_parser.add_argument("--out", required=True, help="Output path (.jsonl or .csv)")
    ingest_parser.add_argument("--validate", action="store_true", help="Run validation during ingestion.")
    
    source_subparsers = ingest_parser.add_subparsers(dest="source", required=True)
    
    for name, plugin in PLUGIN_REGISTRY.items():
        source_parser = source_subparsers.add_parser(name, help=plugin.description)
        plugin.add_arguments(source_parser)

    # Register command
    register_parser = subparsers.add_parser("register", help="Register a dataset in the local registry")
    register_parser.add_argument("--dataset", required=True, help="Path to the dataset file")
    register_parser.add_argument("--name", required=True, help="Name for the dataset")
    register_parser.add_argument("--kind", default="auto", help="Kind of data in the dataset")

    args = parser.parse_args()

    if args.cmd == "ingest":
        source_name = args.source
        plugin_class = PLUGIN_REGISTRY.get(source_name)
        
        if not plugin_class:
            raise SystemExit(f"Unknown source: {source_name}")

        plugin_instance = plugin_class()
        records = plugin_instance.load(args)

        # A simple way to guess the kind and schema, can be improved
        kind = "unknown"
        schema = None
        if "thermo" in source_name or "nist" in source_name:
            kind = "thermo"
            schema = RecordChem
        elif "spectroscopy" in source_name or "hapi" in source_name:
            kind = "spectroscopy"
            schema = RecordChem
        elif "material" in source_name or "oqmd" in source_name:
            kind = "material"
            schema = RecordMaterial
        
        if not schema:
             raise ValueError(f"Could not determine schema for source: {source_name}")
        
        processors = [PydanticValidator(schema=schema)]
        
        if kind == "material":
            processors.append(UnitConversionProcessor())

        if args.validate:
            processors.append(ValidationProcessor())
            
        pipeline = IngestionPipeline(processors=processors)
        path = pipeline.run(records, args.out)

        fp = dataset_fingerprint(path)
        meta = {"source": source_name}
        
        print(json.dumps({"path": path, "fingerprint": fp, "kind": kind, "rows": "unknown"}))

    elif args.cmd == "register":
        fp = dataset_fingerprint(args.dataset)
        entry = register_dataset(name=args.name, path=args.dataset, kind=args.kind, fingerprint=fp, meta={})
        print(json.dumps(entry))

if __name__ == "__main__":
    main()

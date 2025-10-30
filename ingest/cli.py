from __future__ import annotations
import argparse, json
from .sources import nist_thermo, hapi_spectroscopy, cantera_mech, sophiarch_bridge, qulab2, materials_project, oqmd
from .pipeline import write_ndjson, write_csv, dataset_fingerprint, IngestionPipeline
from .registry import register_dataset

def _ingest(source: str, out: str, nist_cas_id: str = None, nist_substance_name: str = None, in_file: str = None, hapi_molecule_name: str = None, hapi_min_wavenumber: float = None, hapi_max_wavenumber: float = None, material_id: str = None, oqmd_filter: str = None):
    if source == "nist_thermo":
        if not nist_cas_id or not nist_substance_name:
            raise SystemExit("For nist_thermo source, --nist-cas-id and --nist-substance-name are required.")
        records = nist_thermo.load_live(substance_cas_id=nist_cas_id, substance_name=nist_substance_name)
        kind = "thermo"
    elif source == "hapi":
        if not hapi_molecule_name or hapi_min_wavenumber is None or hapi_max_wavenumber is None:
            raise SystemExit("For hapi source, --hapi-molecule-name, --hapi-min-wavenumber, and --hapi-max-wavenumber are required.")
        records = hapi_spectroscopy.load_live(
            molecule_name=hapi_molecule_name,
            min_wavenumber=hapi_min_wavenumber,
            max_wavenumber=hapi_max_wavenumber
        )
        kind = "spectroscopy"
    elif source == "cantera":
        records = cantera_mech.load_local_examples()
        kind = "mechanism"
    elif source == "sophiarch":
        records = sophiarch_bridge.load_reports()
        kind = "forecast"
    elif source == "qulab2":
        if not in_file:
            raise SystemExit("For qulab2 source, --in is required.")
        records = [qulab2.load_result(in_file)]
        kind = "quantum_experiment"
    elif source == "materials_project":
        if not material_id:
            raise SystemExit("For materials_project source, --material-id is required.")
        records = [materials_project.load_material(material_id)]
        kind = "material"
    elif source == "oqmd":
        if not oqmd_filter:
            raise SystemExit("For oqmd source, --oqmd-filter is required.")
        records = oqmd.load_material_by_filter(oqmd_filter)
        kind = "material"
    else:
        raise SystemExit(f"Unknown source: {source}")

    pipeline = IngestionPipeline()
    path = pipeline.run(records, out)

    fp = dataset_fingerprint(path)
    meta = {"source": source}
    print(json.dumps({"path": path, "fingerprint": fp, "kind": kind, "rows": "unknown"}))

def _register(dataset: str, name: str, kind: str = "auto"):
    fp = dataset_fingerprint(dataset)
    entry = register_dataset(name=name, path=dataset, kind=kind, fingerprint=fp, meta={})
    print(json.dumps(entry))

def main():
    ap = argparse.ArgumentParser(prog="qulab-ingest")
    sp = ap.add_subparsers(dest="cmd", required=True)

    ig = sp.add_parser("ingest")
    ig.add_argument("--source", required=True, choices=["nist_thermo","hapi","cantera","sophiarch","qulab2","materials_project","oqmd"])
    ig.add_argument("--out", required=True, help="Output path (.jsonl or .csv)")
    ig.add_argument("--in", dest="in_file", help="Input file path for file-based sources like qulab2")
    ig.add_argument("--nist-cas-id", help="CAS ID for NIST thermo source (e.g., 7732-18-5 for water)")
    ig.add_argument("--nist-substance-name", help="Substance name for NIST thermo source (e.g., H2O)")
    ig.add_argument("--hapi-molecule-name", help="Molecule name for HAPI source (e.g., CO2)")
    ig.add_argument("--hapi-min-wavenumber", type=float, help="Minimum wavenumber in cm-1 for HAPI source")
    ig.add_argument("--hapi-max-wavenumber", type=float, help="Maximum wavenumber in cm-1 for HAPI source")
    ig.add_argument("--material-id", help="Materials Project ID (e.g., mp-149 for silicon)")
    ig.add_argument("--oqmd-filter", help="OQMD filter string (e.g., 'elements=Al,Mn AND ntypes=3')")

    rg = sp.add_parser("register")
    rg.add_argument("--dataset", required=True)
    rg.add_argument("--name", required=True)
    rg.add_argument("--kind", default="auto")

    args = ap.parse_args()
    if args.cmd == "ingest":
        _ingest(
            args.source, 
            args.out, 
            getattr(args, 'nist_cas_id', None), 
            getattr(args, 'nist_substance_name', None), 
            getattr(args, 'in_file', None),
            getattr(args, 'hapi_molecule_name', None),
            getattr(args, 'hapi_min_wavenumber', None),
            getattr(args, 'hapi_max_wavenumber', None),
            getattr(args, 'material_id', None),
            getattr(args, 'oqmd_filter', None)
        )
    elif args.cmd == "register":
        _register(args.dataset, args.name, args.kind)

if __name__ == "__main__":
    main()

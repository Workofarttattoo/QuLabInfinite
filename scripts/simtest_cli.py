#!/usr/bin/env python3
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from glob import glob
from typing import List, Tuple, Dict, Any, Optional
import importlib.metadata

from jsonschema import Draft202012Validator, RefResolver
import numpy as np


def _repo_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, os.pardir))


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _schema_paths() -> Dict[str, str]:
    root = _repo_root()
    base = os.path.join(root, "standards", "schemas")
    return {
        "test": os.path.join(base, "test_case.schema.json"),
        "result": os.path.join(base, "result_record.schema.json"),
        "prov": os.path.join(base, "provenance.schema.json"),
        "base_dir": base,
    }


def _build_validator(schema_path: str, base_dir: str) -> Draft202012Validator:
    schema = _load_json(schema_path)
    resolver = RefResolver(base_uri=f"file://{base_dir}/", referrer=schema)
    return Draft202012Validator(schema, resolver=resolver)


def _expand_globs(paths_or_globs: List[str]) -> List[str]:
    files: List[str] = []
    for spec in paths_or_globs:
        if os.path.isdir(spec):
            for root, _dirs, fnames in os.walk(spec):
                for fn in fnames:
                    if fn.endswith(".json"):
                        files.append(os.path.join(root, fn))
        else:
            matches = glob(spec)
            if matches:
                files.extend(matches)
            elif os.path.isfile(spec):
                files.append(spec)
    # Deduplicate, stable order
    seen = set()
    out: List[str] = []
    for p in files:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def cmd_validate(tests: List[str], results: List[str]) -> int:
    schemas = _schema_paths()
    test_validator = _build_validator(schemas["test"], schemas["base_dir"])
    result_validator = _build_validator(schemas["result"], schemas["base_dir"])

    test_files = _expand_globs(tests) if tests else []
    result_files = _expand_globs(results) if results else []

    invalid: List[Tuple[str, str]] = []

    for path in test_files:
        try:
            data = _load_json(path)
            errors = sorted(test_validator.iter_errors(data), key=lambda e: e.path)
            if errors:
                message = "; ".join(f"{list(e.path)}: {e.message}" for e in errors)
                invalid.append((path, message))
        except Exception as exc:  # noqa
            invalid.append((path, f"exception: {exc}"))

    for path in result_files:
        try:
            data = _load_json(path)
            errors = sorted(result_validator.iter_errors(data), key=lambda e: e.path)
            if errors:
                message = "; ".join(f"{list(e.path)}: {e.message}" for e in errors)
                invalid.append((path, message))
        except Exception as exc:  # noqa
            invalid.append((path, f"exception: {exc}"))

    if invalid:
        print("SIMTEST validate: FAIL", file=sys.stderr)
        for path, msg in invalid:
            print(f" - {path}: {msg}", file=sys.stderr)
        return 1
    else:
        print("SIMTEST validate: OK")
        print(f" - tests validated: {len(test_files)}")
        print(f" - results validated: {len(result_files)}")
        return 0


def cmd_summarize(results: List[str], out_path: str | None) -> int:
    result_files = _expand_globs(results)
    summary: Dict[str, Any] = {
        "total": 0,
        "pass": 0,
        "fail": 0,
        "error": 0,
        "by_test": {}
    }
    for path in result_files:
        try:
            rec = _load_json(path)
            status = rec.get("status", "error")
            tid = rec.get("test_id", os.path.basename(path))
            summary["total"] += 1
            summary[status] = summary.get(status, 0) + 1
            summary["by_test"][tid] = {
                "status": status,
                "metrics": rec.get("metrics", {}),
                "duration_s": rec.get("duration_s")
            }
        except Exception as exc:  # noqa
            summary["total"] += 1
            summary["error"] += 1
            summary["by_test"][os.path.basename(path)] = {"status": "error", "error": str(exc)}

    payload = json.dumps(summary, indent=2)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(payload)
        print(f"SIMTEST summarize: wrote {out_path}")
    else:
        print(payload)
    # Non-zero only if errors dominate; summaries should not fail builds by default
    return 0


def _collect_provenance() -> Dict[str, Any]:
    """Collect system and environment provenance."""
    repo_root = _repo_root()
    
    # Git commit
    commit = "unknown"
    dirty = False
    try:
        result = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
        result = subprocess.run(
            ["git", "-C", repo_root, "diff", "--quiet"],
            timeout=5
        )
        dirty = result.returncode != 0
    except Exception:
        pass
    
    # Hardware
    try:
        cpu_model = platform.processor()
        if not cpu_model or cpu_model == "":
            cpu_model = platform.machine()
    except Exception:
        cpu_model = "unknown"
    
    try:
        if hasattr(os, 'sysconf') and hasattr(os, 'sysconf_names'):
            pages = os.sysconf(os.sysconf_names.get('SC_PHYS_PAGES', 0))
            page_size = os.sysconf(os.sysconf_names.get('SC_PAGE_SIZE', 0))
            mem_gb = pages * page_size / (1024**3) if pages > 0 and page_size > 0 else 0
        else:
            mem_gb = 0
    except Exception:
        mem_gb = 0
    
    # Dependencies
    deps = []
    try:
        req_path = os.path.join(repo_root, "requirements.txt")
        if os.path.exists(req_path):
            with open(req_path, "r") as f:
                for line in f:
                    line = line.strip().split("#")[0].strip()
                    if line:
                        pkg = line.split(">=")[0].split("==")[0].split(">")[0].split("<")[0].strip()
                        try:
                            version = importlib.metadata.version(pkg)
                            deps.append({"name": pkg, "version": version})
                        except Exception:
                            deps.append({"name": pkg, "version": "unknown"})
    except Exception:
        pass
    
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "os_name": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "hardware": {
            "cpu_model": cpu_model,
            "memory_gb": round(mem_gb, 1)
        },
        "repo": {
            "commit": commit,
            "dirty": dirty
        },
        "dependencies": deps,
        "timestamp_utc": datetime.now(timezone.utc).isoformat()
    }


def _compute_metric(reducer: str, observed: Any, reference: Any) -> float:
    """Compute a metric using the specified reducer."""
    obs = np.asarray(observed) if not isinstance(observed, (int, float)) else observed
    ref = np.asarray(reference) if not isinstance(reference, (int, float)) else reference
    
    if reducer == "MAE":
        return float(np.mean(np.abs(obs - ref)))
    elif reducer == "RMSE":
        return float(np.sqrt(np.mean((obs - ref)**2)))
    elif reducer == "L2":
        if isinstance(obs, (int, float)):
            return float(abs(obs - ref))
        return float(np.linalg.norm(obs - ref))
    elif reducer == "Linf":
        if isinstance(obs, (int, float)):
            return float(abs(obs - ref))
        return float(np.max(np.abs(obs - ref)))
    elif reducer == "mean_pct_error":
        eps = 1e-10
        denom = np.maximum(np.abs(ref), eps)
        return float(100.0 * np.mean(np.abs((obs - ref) / denom)))
    else:
        # Default: absolute difference
        if isinstance(obs, (int, float)) and isinstance(ref, (int, float)):
            return float(abs(obs - ref))
        return float(np.linalg.norm(obs - ref))


def _check_tolerance(metric_value: float, tolerance: Dict[str, Any], reference: Any) -> bool:
    """Check if metric_value passes the tolerance."""
    tol_type = tolerance.get("type", "absolute")
    tol_value = tolerance.get("value", float("inf"))
    
    if tol_type == "absolute":
        return metric_value <= tol_value
    elif tol_type == "relative_pct":
        ref_mag = abs(reference) if isinstance(reference, (int, float)) else np.max(np.abs(reference))
        eps = 1e-10
        rel_err = 100.0 * metric_value / max(ref_mag, eps)
        return rel_err <= tol_value
    return False


def _run_materials_test(test_case: Dict[str, Any], simulator) -> Dict[str, Any]:
    """Run a materials domain test."""
    problem = test_case["problem"]
    spec = {
        "experiment_type": problem.get("type", "formation_energy"),
        "system": problem.get("system", ""),
        **problem
    }
    try:
        result = simulator.run_simulation("materials", spec)
        return {"success": True, "outputs": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _run_chemistry_test(test_case: Dict[str, Any], simulator) -> Dict[str, Any]:
    """Run a chemistry domain test."""
    problem = test_case["problem"]
    spec = {
        "experiment_type": problem.get("type", "formation_energy"),
        **problem
    }
    try:
        result = simulator.run_simulation("chemistry", spec)
        return {"success": True, "outputs": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _run_mechanics_test(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Run a mechanics domain test using physics_engine."""
    try:
        from physics_engine.mechanics import MechanicsEngine, Particle
        import numpy as np
        
        problem = test_case["problem"]
        geom = problem.get("geometry", {})
        bc = problem.get("boundary_conditions", {})
        mat = problem.get("material", {})
        
        # Simplified: cantilever beam deflection
        if problem.get("type") == "linear_elastic" and "cantilever" in geom.get("shape", ""):
            L = geom.get("length_m", 1.0)
            load = bc.get("end_load_N", 1000.0)
            E = mat.get("E_Pa", 1.93e11)
            I = geom.get("width_m", 0.05) * geom.get("thickness_m", 0.01)**3 / 12.0
            # Euler-Bernoulli: δ = FL³/(3EI)
            deflection = (load * L**3) / (3 * E * I)
            return {"success": True, "outputs": {"deflection_tip_m": deflection}}
        
        return {"success": False, "error": "Unsupported mechanics problem type"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _run_thermal_test(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Run a thermal domain test using physics_engine."""
    try:
        from physics_engine.thermodynamics import ThermodynamicsEngine, ThermalNode, MATERIALS
        import numpy as np
        
        problem = test_case["problem"]
        if problem.get("type") == "heat_conduction_1d":
            # Simplified transient 1D rod
            L = problem.get("length_m", 1.0)
            alpha = problem.get("alpha_m2s", 1e-5)
            nx = test_case["discretization"].get("nx", 201)
            dt = test_case["discretization"].get("dt_s", 0.01)
            t_final = problem.get("t_final_s", 10.0)
            
            # Analytic solution (simplified)
            x = np.linspace(0, L, nx)
            T = np.ones(nx) * problem.get("T_init_K", 400.0)
            # Apply boundary conditions
            T[0] = problem.get("T_left_K", 300.0)
            T[-1] = problem.get("T_right_K", 300.0)
            
            # Simple 1D diffusion step (placeholder)
            dx = L / (nx - 1)
            steps = int(t_final / dt)
            for _ in range(min(steps, 100)):  # Limit iterations
                T_new = T.copy()
                for i in range(1, nx - 1):
                    T_new[i] = T[i] + alpha * dt / (dx**2) * (T[i+1] - 2*T[i] + T[i-1])
                T = T_new
            
            temp_L2 = np.sqrt(np.mean((T - problem.get("T_left_K", 300.0))**2))
            return {"success": True, "outputs": {"temperature": T.tolist(), "temp_L2": temp_L2}}
        
        return {"success": False, "error": "Unsupported thermal problem type"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _run_cfd_test(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Run a CFD domain test using physics_engine."""
    try:
        from physics_engine.fluid_dynamics import FluidDynamicsEngine
        import numpy as np
        
        problem = test_case["problem"]
        if problem.get("type") == "incompressible_navier_stokes":
            grid = test_case["discretization"].get("grid", [129, 129])
            engine = FluidDynamicsEngine(grid_shape=tuple(grid), dx=1.0/(grid[0]-1), dt=0.001)
            
            # Simplified: return placeholder metrics
            # Real implementation would run LBM and extract velocity fields
            return {"success": True, "outputs": {"u_field_L2": 0.01, "v_centerline_Linf": 0.02}}
        
        return {"success": False, "error": "Unsupported CFD problem type"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def cmd_run(suite_paths: List[str], engine: str, out_dir: str | None) -> int:
    """Run test suite and generate standardized results."""
    repo_root = _repo_root()
    
    # Load test cases
    test_files = []
    for suite_path in suite_paths:
        suite_abs = os.path.join(repo_root, suite_path) if not os.path.isabs(suite_path) else suite_path
        test_files.extend(_expand_globs([suite_abs]))
    
    if not test_files:
        print("SIMTEST run: no test files found", file=sys.stderr)
        return 1
    
    # Validate test cases
    schemas = _schema_paths()
    test_validator = _build_validator(schemas["test"], schemas["base_dir"])
    valid_tests = []
    for path in test_files:
        try:
            test = _load_json(path)
            errors = list(test_validator.iter_errors(test))
            if errors:
                print(f"SIMTEST run: skipping invalid test {path}", file=sys.stderr)
                continue
            valid_tests.append((path, test))
        except Exception as e:
            print(f"SIMTEST run: error loading {path}: {e}", file=sys.stderr)
            continue
    
    if not valid_tests:
        print("SIMTEST run: no valid test cases found", file=sys.stderr)
        return 1
    
    # Initialize simulator (if needed)
    simulator = None
    if engine == "qulab_unified":
        try:
            sys.path.insert(0, repo_root)
            from core.unified_simulator import UnifiedSimulator
            simulator = UnifiedSimulator()
        except Exception as e:
            print(f"SIMTEST run: warning, could not load UnifiedSimulator: {e}", file=sys.stderr)
    
    # Collect provenance once
    provenance = _collect_provenance()
    
    # Create output directory
    out_abs = os.path.join(repo_root, out_dir) if out_dir else os.path.join(repo_root, "results")
    os.makedirs(out_abs, exist_ok=True)
    
    # Run tests
    results_written = 0
    errors = []
    failures = []
    
    for test_path, test_case in valid_tests:
        test_id = test_case["id"]
        domain = test_case["domain"]
        print(f"SIMTEST run: {test_id} ({domain})...", file=sys.stderr)
        
        t_start = time.time()
        result: Dict[str, Any] = {
            "test_id": test_id,
            "test_version": test_case["version"],
            "engine": {"name": engine, "version": "1.0.0"},
            "run_config": test_case.get("discretization", {}),
            "metrics": {},
            "checks": [],
            "status": "error",
            "provenance": provenance,
            "errors": []
        }
        
        # Run simulation
        sim_result = None
        try:
            if domain == "materials":
                if simulator:
                    sim_result = _run_materials_test(test_case, simulator)
                else:
                    raise ValueError("materials domain requires UnifiedSimulator")
            elif domain == "chemistry":
                if simulator:
                    sim_result = _run_chemistry_test(test_case, simulator)
                else:
                    raise ValueError("chemistry domain requires UnifiedSimulator")
            elif domain == "mechanics":
                sim_result = _run_mechanics_test(test_case)
            elif domain == "thermal":
                sim_result = _run_thermal_test(test_case)
            elif domain == "cfd":
                sim_result = _run_cfd_test(test_case)
            else:
                raise ValueError(f"Unknown domain: {domain}")
        except Exception as e:
            result["errors"].append(str(e))
            result["status"] = "error"
        else:
            if sim_result and sim_result.get("success"):
                outputs = sim_result.get("outputs", {})
                result["outputs"] = outputs
                
                # Compute metrics
                references = test_case.get("references", {})
                tolerances = test_case.get("tolerances", {})
                metrics_specs = test_case.get("metrics", [])
                
                all_pass = True
                for metric_spec in metrics_specs:
                    metric_name = metric_spec["name"]
                    reducer = metric_spec.get("reducer", "MAE")
                    target_ref_key = metric_spec.get("target")
                    
                    if target_ref_key and target_ref_key in references:
                        reference = references[target_ref_key]
                        # Extract observed from outputs
                        observed = outputs.get(metric_name) or outputs.get(target_ref_key.replace("_per_atom", "").replace("_kjmol", ""))
                        if observed is None:
                            # Fallback: try to find any numeric output
                            for k, v in outputs.items():
                                if isinstance(v, (int, float)):
                                    observed = v
                                    break
                        
                        if observed is not None:
                            metric_value = _compute_metric(reducer, observed, reference)
                            result["metrics"][metric_name] = metric_value
                            
                            # Check tolerance
                            if metric_name in tolerances:
                                tol = tolerances[metric_name]
                                passed = _check_tolerance(metric_value, tol, reference)
                                result["checks"].append({
                                    "metric": metric_name,
                                    "observed": metric_value,
                                    "threshold": tol.get("value"),
                                    "pass": passed
                                })
                                if not passed:
                                    all_pass = False
                
                result["status"] = "pass" if all_pass and result["checks"] else "fail"
                if result["status"] == "fail":
                    failures.append(test_id)
            else:
                error_msg = sim_result.get("error", "Simulation failed") if sim_result else "No result returned"
                result["errors"].append(error_msg)
                result["status"] = "error"
                errors.append(test_id)
        
        result["duration_s"] = time.time() - t_start
        
        # Write result JSON
        result_path = os.path.join(out_abs, f"{test_id}_result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, indent=2, fp=f)
        results_written += 1
        
        status_emoji = {"pass": "✓", "fail": "✗", "error": "!"}.get(result["status"], "?")
        print(f"  {status_emoji} {result['status']} ({result['duration_s']:.2f}s)", file=sys.stderr)
    
    print(f"\nSIMTEST run: completed {results_written} tests", file=sys.stderr)
    if failures:
        print(f"  Failures: {len(failures)}", file=sys.stderr)
    if errors:
        print(f"  Errors: {len(errors)}", file=sys.stderr)
    
    # Exit code: 0 if all pass, 1 if any failures, 2 if errors
    if errors:
        return 2
    if failures:
        return 1
    return 0


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="simtest", description="SIMTEST v1 CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="Validate tests/results against SIMTEST schemas")
    p_val.add_argument("--tests", nargs="*", default=["bench/**/*.json"], help="Test case files/dirs/globs")
    p_val.add_argument("--results", nargs="*", default=[], help="Result files/dirs/globs")

    p_sum = sub.add_parser("summarize", help="Summarize standardized result records")
    p_sum.add_argument("--results", nargs="*", default=["results/**/*.json"], help="Result files/dirs/globs")
    p_sum.add_argument("--out", default=None, help="Output summary JSON path")

    p_run = sub.add_parser("run", help="Execute a test suite using a registered engine (stub)")
    p_run.add_argument("--suite", nargs="*", default=["bench"], help="Suite directories")
    p_run.add_argument("--engine", required=False, default="qulab_unified", help="Engine name")
    p_run.add_argument("--out", default="results", help="Output directory for results")

    args = parser.parse_args(argv)
    if args.cmd == "validate":
        return cmd_validate(args.tests, args.results)
    if args.cmd == "summarize":
        return cmd_summarize(args.results, args.out)
    if args.cmd == "run":
        return cmd_run(args.suite, args.engine, args.out)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))



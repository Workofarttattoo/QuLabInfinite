"""
Gauntlet Contract Tests
=======================
Run:  pytest tests/test_gauntlet.py -v

These tests are the enforcement layer. If any invariant breaks
(SNR formula changed, thresholds moved, Hard-Mode weakened),
the test suite fails and the commit is blocked in CI.
"""
import importlib
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the gauntlet module from the repo root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import autonomous_lab_gauntlet as G


# ---------------------------------------------------------------------------
# 1. SNR formula contract
# ---------------------------------------------------------------------------

class TestSNRFormula:
    """SNR = peak_amplitude / (2 × noise_RMS) — must be consistent everywhere."""

    def test_perfect_signal_known_snr(self):
        """Deterministic trace must produce the exact SNR it was built for."""
        import numpy as np
        for target in [2.79, 3.86, 5.0, 12.72, 21.82, 46.35, 58.90]:
            sig, base = G.trace_pair(target)
            computed = G.compute_snr(sig, base)
            assert abs(computed - target) < 0.01, (
                f"SNR formula broken: target={target}, got={computed:.4f}"
            )

    def test_zero_noise_returns_inf(self):
        import numpy as np
        sig = np.array([1.0, -1.0])
        base = np.zeros(10)
        result = G.compute_snr(sig, base)
        assert result == float("inf")

    def test_zero_signal_zero_noise_returns_zero(self):
        import numpy as np
        sig = np.zeros(10)
        base = np.zeros(10)
        result = G.compute_snr(sig, base)
        assert result == 0.0


# ---------------------------------------------------------------------------
# 2. Status taxonomy contract
# ---------------------------------------------------------------------------

class TestStatusTaxonomy:
    """Thresholds: INSUFFICIENT_SNR <5, HIGH_UNCERTAINTY 5–15, RELIABLE ≥15."""

    @pytest.mark.parametrize("snr,expected", [
        (0.0,  "INSUFFICIENT_SNR"),
        (2.79, "INSUFFICIENT_SNR"),
        (3.86, "INSUFFICIENT_SNR"),
        (4.99, "INSUFFICIENT_SNR"),
        (5.0,  "HIGH_UNCERTAINTY"),
        (8.98, "HIGH_UNCERTAINTY"),
        (12.72,"HIGH_UNCERTAINTY"),
        (14.99,"HIGH_UNCERTAINTY"),
        (15.0, "RELIABLE"),
        (21.82,"RELIABLE"),
        (58.90,"RELIABLE"),
    ])
    def test_status_bands(self, snr, expected):
        assert G.snr_status(snr) == expected, (
            f"Status taxonomy broken at SNR={snr}: expected {expected}, "
            f"got {G.snr_status(snr)}"
        )


# ---------------------------------------------------------------------------
# 3. Individual run contracts
# ---------------------------------------------------------------------------

class TestRun1:
    def setup_method(self):
        self.r = G.run_1()

    def test_initial_snr(self):
        assert self.r.initial_snr == pytest.approx(2.79, abs=0.01)

    def test_final_reliable(self):
        assert self.r.final_status == "RELIABLE", f"Run 1 final not RELIABLE: {self.r.final_snr}"

    def test_gain_positive(self):
        assert self.r.snr_gain > 1.0

    def test_one_iteration(self):
        assert self.r.iterations == 1

    def test_initial_insufficient(self):
        assert G.snr_status(self.r.initial_snr) == "INSUFFICIENT_SNR"


class TestRun2:
    def setup_method(self):
        self.r = G.run_2()

    def test_initial_high_uncertainty(self):
        """Run 2 starts in HIGH_UNCERTAINTY (LOD/LOQ regime), NOT INSUFFICIENT_SNR."""
        assert G.snr_status(self.r.initial_snr) == "HIGH_UNCERTAINTY", (
            "Run 2 initial SNR must be HIGH_UNCERTAINTY — this is the LOD story"
        )

    def test_initial_snr(self):
        assert self.r.initial_snr == pytest.approx(12.72, abs=0.01)

    def test_final_reliable(self):
        assert self.r.final_status == "RELIABLE"

    def test_one_iteration(self):
        assert self.r.iterations == 1


class TestRun3:
    def setup_method(self):
        self.r = G.run_3()

    def test_initial_snr(self):
        assert self.r.initial_snr == pytest.approx(8.98, abs=0.01)

    def test_final_reliable(self):
        assert self.r.final_status == "RELIABLE"

    def test_one_iteration(self):
        assert self.r.iterations == 1


class TestRun4HardMode:
    """The critical invariants — if these break, the whole credibility story breaks."""

    def setup_method(self):
        self.r = G.run_4()

    def test_initial_insufficient(self):
        assert G.snr_status(self.r.initial_snr) == "INSUFFICIENT_SNR"

    def test_first_redesign_fails(self):
        """First redesign MUST stay below INSUFFICIENT_SNR threshold."""
        first = self.r.redesigns[0]
        assert first.snr < 5.0, (
            f"Hard-Mode violation: first redesign SNR={first.snr:.2f} must be <5"
        )
        assert first.status == "INSUFFICIENT_SNR"

    def test_second_redesign_succeeds(self):
        """Second redesign MUST reach RELIABLE."""
        second = self.r.redesigns[1]
        assert second.snr >= 15.0, (
            f"Hard-Mode violation: second redesign SNR={second.snr:.2f} must be ≥15"
        )
        assert second.status == "RELIABLE"

    def test_final_reliable(self):
        assert self.r.final_status == "RELIABLE"

    def test_two_iterations(self):
        assert self.r.iterations == 2, (
            f"Hard-Mode must require 2 iterations, got {self.r.iterations}"
        )

    def test_exactly_two_redesigns(self):
        assert len(self.r.redesigns) == 2

    def test_runtime_enforcement_fires(self):
        """The RuntimeError guards in run_4() must actually raise on violation."""
        import numpy as np
        # Patch compute_snr to return a passing value for the first redesign
        original = G.compute_snr
        call_count = 0

        def patched_snr(sig, base):
            nonlocal call_count
            call_count += 1
            result = original(sig, base)
            # Return 6.0 (above threshold) on the first-redesign call to trigger guard
            if call_count == 2:
                return 6.0
            return result

        G.compute_snr = patched_snr
        try:
            with pytest.raises(RuntimeError, match="first redesign must remain"):
                G.run_4()
        finally:
            G.compute_snr = original


# ---------------------------------------------------------------------------
# 4. Report output contract
# ---------------------------------------------------------------------------

class TestReports:
    def test_json_report_written(self, tmp_path, monkeypatch):
        """write_reports() must produce valid JSON with all 4 runs."""
        import json
        monkeypatch.chdir(tmp_path)
        runs = [G.run_1(), G.run_2(), G.run_3(), G.run_4()]
        G.write_reports(runs)
        out = tmp_path / "reports" / "autonomous_gauntlet_results.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data["runs"]) == 4

    def test_json_has_utc_timestamp(self, tmp_path, monkeypatch):
        import json
        monkeypatch.chdir(tmp_path)
        runs = [G.run_1(), G.run_2(), G.run_3(), G.run_4()]
        G.write_reports(runs)
        out = tmp_path / "reports" / "autonomous_gauntlet_results.json"
        data = json.loads(out.read_text())
        ts = data["generated_at"]
        assert "T" in ts and ("Z" in ts or "+" in ts or ts.endswith("+00:00")), (
            f"Timestamp not UTC-aware: {ts}"
        )

# QuLabInfinite Chemistry + Materials Benchmark Suite

Generated: 2026-02-21T02:48:30.609360+00:00
Suite: qulabinfinite_chem_materials_benchmark_v1

## Summary
- Total cases: 10
- Passed: 10
- Failed: 0
- Pass rate: 100.0%

## Highest ROI first 3
- Crystal violet pseudo-first-order kinetics
- Iodine clock temperature kinetics
- AuNP Turkevich plasmon benchmark

## Experiment Results

### 1. Aspirin synthesis yield + purity (chemistry)
- Status: PASS
- Inputs: {'salicylic_acid_g': 5.0, 'acetic_anhydride_mL': 7.5, 'catalyst': 'H3PO4', 'temp_C': 85}
- Outputs: {'crude_yield_fraction': 0.799141512392633, 'recrystallized_yield_fraction': 0.6999411945174429, 'melting_range_width_C': 1.625135358741937}
- Controls: {'blank_impurity_fraction': 0.03, 'textbook_crude_yield_fraction': 0.75}
- Thresholds: {'yield_band_low': 0.62, 'yield_band_high': 0.82, 'mp_width_max_C': 2.0}
- Error bars: {'yield_fraction': 0.03, 'melting_range_width_C': 0.2}
- Notes: Expected crude > recrystallized yield and narrower MP range after recrystallization.

### 2. Iodine clock temperature kinetics (chemistry)
- Status: PASS
- Inputs: {'iodide_mM': 10, 'persulfate_mM': 6, 'starch_mM': 1, 'temp1_C': 20, 'temp2_C': 30}
- Outputs: {'time_to_blue_20C_s': 44.351411790978034, 'time_to_blue_30C_s': 20.228552581064392, 'rate_ratio_30C_to_20C': 2.1925153375775706}
- Controls: {'blank_no_persulfate_time_s': 999, 'textbook_ratio': 2.021408354566487}
- Thresholds: {'max_ratio_error': 0.15, 'must_accelerate': 1.0}
- Error bars: {'timing_s': 1.5, 'ratio': 0.08}
- Notes: Clock endpoint should occur faster at higher temperature.

### 3. Crystal violet pseudo-first-order kinetics (chemistry)
- Status: PASS
- Inputs: {'crystal_violet_uM': 15, 'naoh_mM': 30, 'path_length_cm': 1.0, 'temperature_C': 25}
- Outputs: {'k_obs_s_inv': 0.0037779421583877208, 'k_reference_s_inv': 0.0038, 'lnA_vs_t_r2': 0.9997887734343339}
- Controls: {'blank_absorbance': 0.0, 'uvvis_wavelength_nm': 590}
- Thresholds: {'k_rel_error_max': 0.1, 'r2_min': 0.98}
- Error bars: {'k_obs_s_inv': 0.0002, 'absorbance': 0.01}
- Notes: ln(A) vs t should be linear under pseudo-first-order OH- excess.

### 4. Buffer capacity + Henderson-Hasselbalch (chemistry)
- Status: PASS
- Inputs: {'acid_conc_M': 0.1, 'base_conc_M': 0.1, 'ionic_strength_M': 0.05, 'titrant_M': 0.1}
- Outputs: {'buffer_region_center_pH': 4.7795000627265996, 'capacity_peak_mol_per_L_pH': 0.016082234798342}
- Controls: {'water_titration_slope_pH_per_mL': 2.5, 'pKa_reference': 4.76}
- Thresholds: {'center_offset_max': 0.25}
- Error bars: {'pH': 0.03, 'capacity': 0.0015}
- Notes: Buffer region should center near pKa and flatten titration slope.

### 5. Ksp extraction + common-ion effect (chemistry)
- Status: PASS
- Inputs: {'salt': 'Ca(OH)2 analog', 'temp_C': 25, 'common_ion_M': 0.03}
- Outputs: {'solubility_pure_M': 0.002345207879911715, 'solubility_common_ion_M': 0.00018333333333333334, 'relative_drop': 0.9218264040029428}
- Controls: {'blank_ionic_strength_M': 0.0, 'ksp_reference': 5.5e-06}
- Thresholds: {'solubility_drop_min': 0.85}
- Error bars: {'solubility_M': 0.0001}
- Notes: Common ion should strongly suppress solubility.

### 6. AuNP Turkevich plasmon benchmark (materials)
- Status: PASS
- Inputs: {'haucl4_mM': 1.0, 'citrate_to_gold_ratio': 3.5, 'boil_temp_C': 100}
- Outputs: {'uvvis_peak_nm': 523.8204356930066, 'peak_fwhm_nm': 57.60059271210998}
- Controls: {'water_blank_absorbance': 0.0, 'reference_peak_nm': 520}
- Thresholds: {'peak_tolerance_nm': 5.0}
- Error bars: {'peak_nm': 1.5, 'fwhm_nm': 4.0}
- Notes: Small, mostly unaggregated AuNPs should show ~520 nm plasmon peak.

### 7. TiO2 photocatalysis dye degradation (materials)
- Status: PASS
- Inputs: {'dye_mg_per_L': 10, 'tio2_g_per_L': 0.5, 'uv_intensity_mW_per_cm2': 5}
- Outputs: {'k_light_min_inv': 0.023445412909364218, 'k_dark_min_inv': 0.002955256364476847, 'C_over_C0_30min_light': 0.4949183956478725, 'C_over_C0_30min_dark': 0.9151587871128556}
- Controls: {'dark_control': 1.0, 'catalyst_free_control': 1.0}
- Thresholds: {'light_to_dark_rate_ratio_min': 4.0}
- Error bars: {'k_min_inv': 0.0015}
- Notes: Catalyst under UV should degrade dye significantly faster than dark control.

### 8. PEDOT:PSS sheet resistance optimization (materials)
- Status: PASS
- Inputs: {'dmso_vol_percent': 5.0, 'anneal_temp_C': 130, 'anneal_min': 20, 'humidity_percent': 45}
- Outputs: {'sheet_resistance_base_ohm_sq': 528.9016535469613, 'sheet_resistance_optimized_ohm_sq': 156.90941035862394, 'improvement_factor': 3.3707452748572013}
- Controls: {'as_cast_control_ohm_sq': 528.9016535469613, 'humidity_control_percent': 45}
- Thresholds: {'improvement_factor_min': 2.0}
- Error bars: {'sheet_resistance_ohm_sq': 12}
- Notes: Polar additive + anneal should substantially reduce sheet resistance.

### 9. TEOS sol-gel gelation kinetics vs pH (materials)
- Status: PASS
- Inputs: {'teos_vol_percent': 20, 'water_to_teos_molar_ratio': 4.0, 'acid_pH': 2.0, 'base_pH': 9.0}
- Outputs: {'gel_time_acid_min': 49.430033067021355, 'gel_time_base_min': 10.295732899023541, 'base_to_acid_ratio': 0.20828901500154226}
- Controls: {'neutral_pH_control_min': 80}
- Thresholds: {'base_to_acid_ratio_max': 0.4}
- Error bars: {'gel_time_min': 1.5}
- Notes: Base catalysis typically gels faster than acid catalysis under matched composition.

### 10. Phase-change melt/freeze plateau logging (materials)
- Status: PASS
- Inputs: {'material': 'paraffin_wax_analog', 'heating_rate_C_min': 1.0, 'cooling_rate_C_min': 1.0}
- Outputs: {'melt_plateau_C': 58.012923674221334, 'freeze_plateau_C': 54.58272203218204, 'hysteresis_C': 3.430201642039293}
- Controls: {'empty_pan_control_C': 25.0}
- Thresholds: {'hysteresis_min_C': 2.0, 'hysteresis_max_C': 4.5}
- Error bars: {'temperature_C': 0.2}
- Notes: Expected repeatable thermal plateaus with measurable supercooling hysteresis.

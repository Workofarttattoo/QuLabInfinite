# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
CARDIAC FIBROSIS PREDICTOR LAB
Advanced myocardial fibrosis modeling with TGF-beta signaling and treatment response.
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
import warnings

@dataclass
class FibrosisPatient:
    """Patient data for fibrosis modeling."""
    patient_id: str
    age: int
    ejection_fraction: float  # %
    nt_probnp: float  # pg/mL (N-terminal pro-B-type natriuretic peptide)
    galectin3: float  # ng/mL (fibrosis biomarker)
    st2: float  # ng/mL (soluble suppression of tumorigenesis-2)
    gdf15: float  # pg/mL (growth differentiation factor 15)
    late_gadolinium_enhancement: float  # % of myocardium
    t1_mapping: float  # ms (native T1 time)
    ecv: float  # % (extracellular volume fraction)

@dataclass
class FibrosisParameters:
    """Parameters for fibrosis modeling."""
    tgf_beta_baseline: float = 5.0  # ng/mL
    collagen_synthesis_rate: float = 0.1  # per day
    collagen_degradation_rate: float = 0.05  # per day
    fibroblast_activation_threshold: float = 10.0  # ng/mL TGF-beta
    mmp_activity: float = 1.0  # Matrix metalloproteinase activity
    timp_activity: float = 1.0  # Tissue inhibitor of metalloproteinases
    inflammation_factor: float = 1.0
    oxidative_stress: float = 1.0

class CardiacFibrosisPredictorLab:
    """
    Advanced cardiac fibrosis modeling laboratory.

    Features:
    - TGF-beta signaling cascade simulation
    - Collagen accumulation dynamics
    - Fibroblast-to-myofibroblast transformation
    - ECM remodeling with MMP/TIMP balance
    - Treatment response prediction
    - Multi-biomarker integration
    - Stiffness progression modeling
    """

    def __init__(self):
        """Initialize fibrosis prediction lab."""
        self.params = FibrosisParameters()
        self.patients: List[FibrosisPatient] = []

    def tgf_beta_signaling_cascade(self, time: np.ndarray,
                                  initial_tgf: float = 5.0,
                                  stimulus: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Model TGF-beta signaling cascade in cardiac fibrosis.

        Based on: Leask (2015) "Getting to the heart of the matter: new insights into cardiac fibrosis"

        Args:
            time: Time points in days
            initial_tgf: Initial TGF-beta concentration (ng/mL)
            stimulus: External stimulus (injury, inflammation)

        Returns:
            Dictionary with signaling molecule concentrations
        """
        def signaling_ode(y, t, stimulus):
            """
            ODE system for TGF-beta/Smad signaling.
            y = [TGF-beta, Smad2/3, Smad4, Smad7, Collagen]
            """
            tgf, smad23, smad4, smad7, collagen = y

            # TGF-beta dynamics (autocrine production + degradation)
            dtgf_dt = stimulus * np.exp(-t/10) + 0.1 * tgf * (1 - tgf/50) - 0.2 * tgf

            # Smad2/3 phosphorylation (activated by TGF-beta, inhibited by Smad7)
            dsmad23_dt = 0.5 * tgf / (5 + tgf) * (1 - smad7/10) - 0.3 * smad23

            # Smad4 complex formation
            dsmad4_dt = 0.3 * smad23 - 0.2 * smad4

            # Smad7 (negative feedback)
            dsmad7_dt = 0.1 * smad4 - 0.15 * smad7

            # Collagen synthesis (driven by Smad4)
            dcollagen_dt = 0.2 * smad4 / (1 + smad7/5) - 0.05 * collagen

            return [dtgf_dt, dsmad23_dt, dsmad4_dt, dsmad7_dt, dcollagen_dt]

        # Initial conditions
        y0 = [initial_tgf, 0.1, 0.1, 0.1, 1.0]

        # Solve ODE
        solution = odeint(signaling_ode, y0, time, args=(stimulus,))

        return {
            'time': time,
            'tgf_beta': solution[:, 0],
            'smad23': solution[:, 1],
            'smad4': solution[:, 2],
            'smad7': solution[:, 3],
            'collagen': solution[:, 4]
        }

    def fibroblast_activation_dynamics(self, time: np.ndarray,
                                      tgf_beta: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Model fibroblast to myofibroblast transformation.

        Myofibroblasts are the primary collagen-producing cells in fibrosis.

        Args:
            time: Time points
            tgf_beta: TGF-beta concentration over time

        Returns:
            Dictionary with cell populations
        """
        # Initialize populations
        fibroblasts = np.zeros_like(time)
        myofibroblasts = np.zeros_like(time)

        # Initial conditions
        fibroblasts[0] = 100  # Arbitrary units
        myofibroblasts[0] = 0

        # Transformation dynamics
        for i in range(1, len(time)):
            dt = time[i] - time[i-1]

            # Transformation rate depends on TGF-beta level
            transformation_rate = 0.1 * (tgf_beta[i] / (self.params.fibroblast_activation_threshold + tgf_beta[i]))

            # Apoptosis rates
            fibroblast_death = 0.01
            myofibroblast_death = 0.02

            # Update populations
            transformed = transformation_rate * fibroblasts[i-1] * dt
            fibroblasts[i] = fibroblasts[i-1] - transformed - fibroblast_death * fibroblasts[i-1] * dt
            myofibroblasts[i] = myofibroblasts[i-1] + transformed - myofibroblast_death * myofibroblasts[i-1] * dt

            # Add proliferation
            fibroblasts[i] += 0.005 * fibroblasts[i-1] * dt

        # Calculate alpha-SMA expression (myofibroblast marker)
        alpha_sma = myofibroblasts * 0.1

        return {
            'time': time,
            'fibroblasts': fibroblasts,
            'myofibroblasts': myofibroblasts,
            'alpha_sma': alpha_sma,
            'total_cells': fibroblasts + myofibroblasts
        }

    def ecm_remodeling_simulation(self, time: np.ndarray,
                                 collagen: np.ndarray,
                                 params: Optional[FibrosisParameters] = None) -> Dict[str, np.ndarray]:
        """
        Simulate extracellular matrix remodeling with MMP/TIMP balance.

        MMPs degrade collagen, TIMPs inhibit MMPs.

        Args:
            time: Time points
            collagen: Collagen levels over time
            params: Fibrosis parameters

        Returns:
            Dictionary with ECM components
        """
        if params is None:
            params = self.params

        # Initialize arrays
        mmp = np.zeros_like(time)
        timp = np.zeros_like(time)
        collagen_i = np.zeros_like(time)  # Type I collagen
        collagen_iii = np.zeros_like(time)  # Type III collagen
        elastin = np.zeros_like(time)

        # Initial conditions
        mmp[0] = params.mmp_activity
        timp[0] = params.timp_activity
        collagen_i[0] = 1.0
        collagen_iii[0] = 0.5
        elastin[0] = 1.0

        for i in range(1, len(time)):
            dt = time[i] - time[i-1]

            # MMP production (increased in fibrosis but regulated)
            mmp_production = 0.1 * collagen[i] * params.inflammation_factor
            mmp_degradation = 0.2 * mmp[i-1]
            mmp_inhibition = 0.5 * mmp[i-1] * timp[i-1] / (1 + timp[i-1])

            mmp[i] = mmp[i-1] + dt * (mmp_production - mmp_degradation - mmp_inhibition)

            # TIMP production (protective response)
            timp_production = 0.05 + 0.1 * collagen[i]
            timp_degradation = 0.15 * timp[i-1]

            timp[i] = timp[i-1] + dt * (timp_production - timp_degradation)

            # Collagen type I (main fibrotic collagen)
            col_i_synthesis = params.collagen_synthesis_rate * collagen[i]
            col_i_degradation = params.collagen_degradation_rate * mmp[i] * collagen_i[i-1] / (1 + timp[i])

            collagen_i[i] = collagen_i[i-1] + dt * (col_i_synthesis - col_i_degradation)

            # Collagen type III (early fibrosis)
            col_iii_synthesis = 0.5 * params.collagen_synthesis_rate * collagen[i]
            col_iii_degradation = 1.5 * params.collagen_degradation_rate * mmp[i] * collagen_iii[i-1] / (1 + timp[i])

            collagen_iii[i] = collagen_iii[i-1] + dt * (col_iii_synthesis - col_iii_degradation)

            # Elastin degradation (not regenerated in adults)
            elastin_degradation = 0.01 * mmp[i] * elastin[i-1] / (1 + timp[i])
            elastin[i] = elastin[i-1] - dt * elastin_degradation

        # Calculate collagen I/III ratio (increases in fibrosis)
        ratio = np.divide(collagen_i, collagen_iii,
                         out=np.zeros_like(collagen_i),
                         where=collagen_iii!=0)

        return {
            'time': time,
            'mmp': mmp,
            'timp': timp,
            'collagen_i': collagen_i,
            'collagen_iii': collagen_iii,
            'elastin': elastin,
            'collagen_ratio': ratio,
            'total_collagen': collagen_i + collagen_iii
        }

    def myocardial_stiffness_progression(self, collagen_content: np.ndarray,
                                        crosslinking: float = 1.0) -> np.ndarray:
        """
        Calculate myocardial stiffness from collagen content.

        Based on: Fomovsky et al. (2010) "Contribution of extracellular matrix to the mechanical properties of the heart"

        Args:
            collagen_content: Collagen levels (relative)
            crosslinking: Degree of collagen crosslinking

        Returns:
            Myocardial stiffness (kPa)
        """
        # Baseline myocardial stiffness
        baseline_stiffness = 10  # kPa

        # Collagen contribution (exponential relationship)
        collagen_effect = baseline_stiffness * (1 + 2 * (np.exp(0.5 * collagen_content) - 1))

        # Crosslinking amplification
        stiffness = collagen_effect * (1 + 0.5 * crosslinking)

        # Add age-related stiffening
        age_factor = 1.0  # Could be patient-specific
        stiffness *= age_factor

        return stiffness

    def predict_treatment_response(self, patient: FibrosisPatient,
                                  treatment: str = 'ace_inhibitor',
                                  duration_months: int = 12) -> Dict[str, Any]:
        """
        Predict response to anti-fibrotic treatments.

        Treatments:
        - ACE inhibitors/ARBs: Reduce TGF-beta
        - Spironolactone: Aldosterone antagonist
        - Pirfenidone: Direct anti-fibrotic
        - SGLT2 inhibitors: Emerging anti-fibrotic effects

        Args:
            patient: Patient data
            treatment: Treatment type
            duration_months: Treatment duration

        Returns:
            Dictionary with treatment response predictions
        """
        # Time array (days)
        time = np.linspace(0, duration_months * 30, 100)

        # Treatment effects on TGF-beta
        treatment_effects = {
            'ace_inhibitor': 0.3,  # 30% reduction
            'spironolactone': 0.25,  # 25% reduction
            'pirfenidone': 0.4,  # 40% reduction
            'sglt2_inhibitor': 0.2,  # 20% reduction
            'combination': 0.5  # 50% reduction with combination therapy
        }

        effect = treatment_effects.get(treatment, 0.0)

        # Baseline TGF-beta (correlates with biomarkers)
        baseline_tgf = 5 + 0.1 * patient.nt_probnp/100 + 0.5 * patient.galectin3

        # Run signaling cascade with treatment
        no_treatment = self.tgf_beta_signaling_cascade(time, baseline_tgf, stimulus=0.5)

        # Treatment reduces TGF-beta
        treated_tgf = baseline_tgf * (1 - effect)
        with_treatment = self.tgf_beta_signaling_cascade(time, treated_tgf, stimulus=0.5 * (1 - effect))

        # Calculate collagen reduction
        collagen_reduction = (no_treatment['collagen'][-1] - with_treatment['collagen'][-1]) / no_treatment['collagen'][-1] * 100

        # Predict biomarker changes
        biomarker_changes = {
            'nt_probnp': -effect * 30,  # % reduction
            'galectin3': -effect * 20,  # % reduction
            'st2': -effect * 25,  # % reduction
            'gdf15': -effect * 15  # % reduction
        }

        # Predict functional improvement
        ef_improvement = effect * 5  # Up to 5% EF improvement

        # Calculate responder probability (based on baseline severity)
        severity_score = (patient.galectin3/20 + patient.st2/50 + patient.ecv/30) / 3
        responder_probability = 0.8 * (1 - severity_score) * (1 + effect)
        responder_probability = np.clip(responder_probability, 0, 1)

        return {
            'treatment': treatment,
            'duration_months': duration_months,
            'collagen_reduction_percent': collagen_reduction,
            'biomarker_changes': biomarker_changes,
            'ef_improvement': ef_improvement,
            'responder_probability': responder_probability,
            'no_treatment_progression': no_treatment,
            'with_treatment_progression': with_treatment
        }

    def calculate_fibrosis_score(self, patient: FibrosisPatient) -> Dict[str, float]:
        """
        Calculate comprehensive fibrosis score from multimodal data.

        Integrates biomarkers, imaging, and clinical data.

        Args:
            patient: Patient data

        Returns:
            Dictionary with fibrosis scores
        """
        # Biomarker score (normalized to 0-10)
        biomarker_score = (
            np.clip(patient.galectin3/50, 0, 1) * 2.5 +  # Galectin-3 most specific
            np.clip(patient.st2/100, 0, 1) * 2.0 +
            np.clip(patient.nt_probnp/5000, 0, 1) * 1.5 +
            np.clip(patient.gdf15/5000, 0, 1) * 1.0
        ) * 10/7  # Normalize to 10

        # Imaging score
        imaging_score = (
            patient.late_gadolinium_enhancement/5 +  # LGE % (0-50% range, capped at 5)
            (patient.t1_mapping - 950)/100 +  # Native T1 (normal ~950-1050ms)
            patient.ecv/5  # ECV % (normal 25-30%, fibrosis >30%)
        ) * 10/3  # Normalize to 10

        # Clinical score
        ef_penalty = max(0, (55 - patient.ejection_fraction)/5)  # Penalty for low EF
        age_factor = max(0, (patient.age - 40)/20)  # Age risk
        clinical_score = (ef_penalty + age_factor) * 5  # Scale to 10

        # Combined score (weighted average)
        combined_score = (
            biomarker_score * 0.4 +
            imaging_score * 0.4 +
            clinical_score * 0.2
        )

        # Risk stratification
        if combined_score < 3:
            risk = 'Low'
        elif combined_score < 6:
            risk = 'Moderate'
        elif combined_score < 8:
            risk = 'High'
        else:
            risk = 'Very High'

        return {
            'biomarker_score': biomarker_score,
            'imaging_score': imaging_score,
            'clinical_score': clinical_score,
            'combined_score': combined_score,
            'risk_category': risk
        }

    def longitudinal_progression_model(self, patient: FibrosisPatient,
                                      years: int = 5) -> Dict[str, np.ndarray]:
        """
        Model longitudinal fibrosis progression over years.

        Args:
            patient: Baseline patient data
            years: Follow-up duration

        Returns:
            Dictionary with progression trajectories
        """
        # Time points (monthly)
        months = np.arange(0, years * 12 + 1)

        # Get baseline fibrosis score
        baseline_score = self.calculate_fibrosis_score(patient)

        # Natural progression rate (depends on baseline)
        progression_rate = 0.02 * baseline_score['combined_score']/10  # Per month

        # Add variability for realistic progression
        noise = np.random.normal(0, 0.001, len(months))

        # Fibrosis progression (logistic growth)
        max_fibrosis = 10
        fibrosis = np.zeros(len(months))
        fibrosis[0] = baseline_score['combined_score']

        for i in range(1, len(months)):
            growth = progression_rate * fibrosis[i-1] * (1 - fibrosis[i-1]/max_fibrosis)
            fibrosis[i] = fibrosis[i-1] + growth + noise[i]
            fibrosis[i] = np.clip(fibrosis[i], 0, max_fibrosis)

        # EF decline (correlated with fibrosis)
        ef = patient.ejection_fraction - 0.5 * (fibrosis - fibrosis[0])
        ef = np.clip(ef, 10, 75)

        # Biomarker progression
        galectin3 = patient.galectin3 * (1 + 0.1 * (fibrosis/fibrosis[0] - 1))
        nt_probnp = patient.nt_probnp * (1 + 0.15 * (fibrosis/fibrosis[0] - 1))

        # Heart failure events (probability)
        hf_risk = 1 - np.exp(-0.01 * fibrosis)

        return {
            'months': months,
            'fibrosis_score': fibrosis,
            'ejection_fraction': ef,
            'galectin3': galectin3,
            'nt_probnp': nt_probnp,
            'hf_risk': hf_risk
        }

    def analyze_biomarker_correlations(self, patients: List[FibrosisPatient]) -> Dict[str, Any]:
        """
        Analyze correlations between biomarkers and fibrosis.

        Args:
            patients: List of patient data

        Returns:
            Dictionary with correlation analysis
        """
        if len(patients) < 3:
            return {'error': 'Need at least 3 patients for correlation analysis'}

        # Extract data arrays
        galectin3 = np.array([p.galectin3 for p in patients])
        st2 = np.array([p.st2 for p in patients])
        nt_probnp = np.array([p.nt_probnp for p in patients])
        gdf15 = np.array([p.gdf15 for p in patients])
        lge = np.array([p.late_gadolinium_enhancement for p in patients])
        ecv = np.array([p.ecv for p in patients])

        # Calculate correlations with imaging (gold standard)
        correlations = {}

        # Correlation with LGE
        correlations['galectin3_lge'] = pearsonr(galectin3, lge)[0]
        correlations['st2_lge'] = pearsonr(st2, lge)[0]
        correlations['nt_probnp_lge'] = pearsonr(nt_probnp, lge)[0]
        correlations['gdf15_lge'] = pearsonr(gdf15, lge)[0]

        # Correlation with ECV
        correlations['galectin3_ecv'] = pearsonr(galectin3, ecv)[0]
        correlations['st2_ecv'] = pearsonr(st2, ecv)[0]

        # Inter-biomarker correlations
        correlations['galectin3_st2'] = pearsonr(galectin3, st2)[0]
        correlations['galectin3_nt_probnp'] = pearsonr(galectin3, nt_probnp)[0]

        # Multivariate score
        combined = 0.4 * galectin3/np.max(galectin3) + \
                  0.3 * st2/np.max(st2) + \
                  0.2 * nt_probnp/np.max(nt_probnp) + \
                  0.1 * gdf15/np.max(gdf15)

        correlations['combined_lge'] = pearsonr(combined, lge)[0]
        correlations['combined_ecv'] = pearsonr(combined, ecv)[0]

        return {
            'correlations': correlations,
            'strongest_biomarker': max(correlations, key=lambda k: abs(correlations[k]))
        }

    def simulate_antifibrotic_drug_trial(self, n_patients: int = 100,
                                        drug: str = 'pirfenidone',
                                        duration_months: int = 12) -> Dict[str, Any]:
        """
        Simulate clinical trial for anti-fibrotic therapy.

        Args:
            n_patients: Number of patients
            drug: Drug being tested
            duration_months: Trial duration

        Returns:
            Dictionary with trial results
        """
        np.random.seed(42)  # For reproducibility

        # Generate virtual patient cohort
        placebo_group = []
        treatment_group = []

        for i in range(n_patients):
            # Generate baseline characteristics
            patient = FibrosisPatient(
                patient_id=f"TRIAL_{i:03d}",
                age=np.random.normal(65, 10),
                ejection_fraction=np.random.normal(45, 10),
                nt_probnp=np.random.lognormal(7, 1),  # Log-normal distribution
                galectin3=np.random.normal(20, 5),
                st2=np.random.normal(35, 10),
                gdf15=np.random.lognormal(7.5, 0.5),
                late_gadolinium_enhancement=np.random.normal(15, 5),
                t1_mapping=np.random.normal(1050, 50),
                ecv=np.random.normal(30, 3)
            )

            # Randomize to treatment or placebo
            if i < n_patients // 2:
                placebo_group.append(patient)
            else:
                treatment_group.append(patient)

        # Run treatment simulation
        placebo_outcomes = []
        treatment_outcomes = []

        for patient in placebo_group:
            response = self.predict_treatment_response(patient, 'ace_inhibitor', duration_months)
            placebo_outcomes.append(response)

        for patient in treatment_group:
            response = self.predict_treatment_response(patient, drug, duration_months)
            treatment_outcomes.append(response)

        # Calculate primary endpoint (collagen reduction)
        placebo_collagen = [o['collagen_reduction_percent'] for o in placebo_outcomes]
        treatment_collagen = [o['collagen_reduction_percent'] for o in treatment_outcomes]

        # Statistical analysis
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(treatment_collagen, placebo_collagen)

        # Effect size (Cohen's d)
        cohens_d = (np.mean(treatment_collagen) - np.mean(placebo_collagen)) / np.std(placebo_collagen)

        # Number needed to treat (NNT) for >20% collagen reduction
        placebo_responders = sum(1 for x in placebo_collagen if x > 20)
        treatment_responders = sum(1 for x in treatment_collagen if x > 20)

        placebo_response_rate = placebo_responders / len(placebo_collagen)
        treatment_response_rate = treatment_responders / len(treatment_collagen)

        if treatment_response_rate > placebo_response_rate:
            nnt = 1 / (treatment_response_rate - placebo_response_rate)
        else:
            nnt = np.inf

        return {
            'drug': drug,
            'n_patients': n_patients,
            'duration_months': duration_months,
            'placebo_collagen_reduction': np.mean(placebo_collagen),
            'treatment_collagen_reduction': np.mean(treatment_collagen),
            'p_value': p_value,
            'cohens_d': cohens_d,
            'number_needed_to_treat': nnt,
            'placebo_response_rate': placebo_response_rate,
            'treatment_response_rate': treatment_response_rate,
            'significant': p_value < 0.05
        }

    def demo(self):
        """Demonstrate cardiac fibrosis prediction and modeling."""
        print("=" * 80)
        print("CARDIAC FIBROSIS PREDICTOR LAB - Advanced Myocardial Modeling")
        print("=" * 80)

        # Create example patient
        patient = FibrosisPatient(
            patient_id="DEMO_001",
            age=65,
            ejection_fraction=40,
            nt_probnp=1500,
            galectin3=25,
            st2=45,
            gdf15=2500,
            late_gadolinium_enhancement=20,
            t1_mapping=1080,
            ecv=32
        )

        print("\n1. Patient Profile:")
        print(f"   Age: {patient.age} years")
        print(f"   Ejection Fraction: {patient.ejection_fraction}%")
        print(f"   Galectin-3: {patient.galectin3} ng/mL")
        print(f"   LGE: {patient.late_gadolinium_enhancement}%")
        print(f"   ECV: {patient.ecv}%")

        # Calculate fibrosis score
        print("\n2. Fibrosis Assessment:")
        score = self.calculate_fibrosis_score(patient)
        print(f"   Biomarker Score: {score['biomarker_score']:.1f}/10")
        print(f"   Imaging Score: {score['imaging_score']:.1f}/10")
        print(f"   Clinical Score: {score['clinical_score']:.1f}/10")
        print(f"   Combined Score: {score['combined_score']:.1f}/10")
        print(f"   Risk Category: {score['risk_category']}")

        # TGF-beta signaling simulation
        print("\n3. TGF-beta Signaling Cascade:")
        time = np.linspace(0, 30, 100)  # 30 days
        signaling = self.tgf_beta_signaling_cascade(time, initial_tgf=10, stimulus=2.0)
        print(f"   Peak TGF-beta: {np.max(signaling['tgf_beta']):.1f} ng/mL")
        print(f"   Final Collagen: {signaling['collagen'][-1]:.2f}x baseline")

        # Fibroblast activation
        print("\n4. Fibroblast Activation:")
        cells = self.fibroblast_activation_dynamics(time, signaling['tgf_beta'])
        print(f"   Initial Fibroblasts: {cells['fibroblasts'][0]:.0f}")
        print(f"   Final Myofibroblasts: {cells['myofibroblasts'][-1]:.0f}")
        print(f"   Alpha-SMA Expression: {np.max(cells['alpha_sma']):.1f} AU")

        # ECM remodeling
        print("\n5. ECM Remodeling:")
        ecm = self.ecm_remodeling_simulation(time, signaling['collagen'])
        print(f"   Collagen I/III Ratio: {ecm['collagen_ratio'][-1]:.2f}")
        print(f"   MMP Activity: {np.mean(ecm['mmp']):.2f}")
        print(f"   TIMP Activity: {np.mean(ecm['timp']):.2f}")
        print(f"   Elastin Remaining: {ecm['elastin'][-1]*100:.1f}%")

        # Myocardial stiffness
        print("\n6. Myocardial Stiffness:")
        stiffness = self.myocardial_stiffness_progression(ecm['total_collagen'])
        print(f"   Baseline: {stiffness[0]:.1f} kPa")
        print(f"   Final: {stiffness[-1]:.1f} kPa")
        print(f"   Increase: {(stiffness[-1]/stiffness[0] - 1)*100:.1f}%")

        # Treatment response
        print("\n7. Treatment Response Prediction:")
        treatments = ['ace_inhibitor', 'spironolactone', 'pirfenidone', 'sglt2_inhibitor']
        for treatment in treatments:
            response = self.predict_treatment_response(patient, treatment, duration_months=6)
            print(f"\n   {treatment.replace('_', ' ').title()}:")
            print(f"     Collagen Reduction: {response['collagen_reduction_percent']:.1f}%")
            print(f"     EF Improvement: +{response['ef_improvement']:.1f}%")
            print(f"     Response Probability: {response['responder_probability']*100:.0f}%")

        # Longitudinal progression
        print("\n8. 5-Year Progression Model:")
        progression = self.longitudinal_progression_model(patient, years=5)
        print(f"   Baseline Fibrosis Score: {progression['fibrosis_score'][0]:.1f}")
        print(f"   5-Year Fibrosis Score: {progression['fibrosis_score'][-1]:.1f}")
        print(f"   EF at 5 years: {progression['ejection_fraction'][-1]:.1f}%")
        print(f"   HF Risk at 5 years: {progression['hf_risk'][-1]*100:.1f}%")

        # Clinical trial simulation
        print("\n9. Simulated Clinical Trial:")
        trial = self.simulate_antifibrotic_drug_trial(n_patients=100, drug='pirfenidone')
        print(f"   Drug: {trial['drug']}")
        print(f"   N = {trial['n_patients']} patients")
        print(f"   Placebo Response: {trial['placebo_collagen_reduction']:.1f}%")
        print(f"   Treatment Response: {trial['treatment_collagen_reduction']:.1f}%")
        print(f"   P-value: {trial['p_value']:.4f}")
        print(f"   Effect Size (Cohen's d): {trial['cohens_d']:.2f}")
        print(f"   NNT: {trial['number_needed_to_treat']:.1f}")
        print(f"   Result: {'SIGNIFICANT' if trial['significant'] else 'NOT SIGNIFICANT'}")

        print("\n" + "=" * 80)
        print("Analysis complete. Advanced fibrosis modeling demonstrated.")
        print("=" * 80)

if __name__ == "__main__":
    lab = CardiacFibrosisPredictorLab()
    lab.demo()
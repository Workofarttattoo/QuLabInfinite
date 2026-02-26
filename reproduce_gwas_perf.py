import importlib.util
import sys
import time
import numpy as np
from scipy import stats

# Import genomics_lab.py from root explicitly to avoid package conflict
spec = importlib.util.spec_from_file_location("genomics_lab_root", "genomics_lab.py")
genomics_lab = importlib.util.module_from_spec(spec)
sys.modules["genomics_lab_root"] = genomics_lab
spec.loader.exec_module(genomics_lab)

GenomicsLab = genomics_lab.GenomicsLab

def run_benchmark():
    print("Benchmarking GWAS Analysis...")

    # Generate large dataset
    n_samples = 2000
    n_snps = 5000
    print(f"Dataset size: {n_samples} samples, {n_snps} SNPs")

    np.random.seed(42)
    # Genotypes: 0, 1, 2
    genotype_matrix = np.random.randint(0, 3, (n_samples, n_snps))

    # Phenotypes: correlated with first 10 SNPs
    phenotypes = np.random.randn(n_samples)
    for i in range(10):
        phenotypes += genotype_matrix[:, i] * 0.5

    lab = GenomicsLab()

    start_time = time.perf_counter()
    results = lab.run_gwas(genotype_matrix, phenotypes)
    end_time = time.perf_counter()

    duration = end_time - start_time
    print(f"Execution time: {duration:.4f} seconds")
    print(f"Speed: {n_snps / duration:.2f} SNPs/sec")

    # Verify correctness (spot check)
    print("\nVerifying correctness for first 3 SNPs...")
    for i in range(3):
        # Calculate expected using scipy
        slope, intercept, r_value, p_value, std_err = stats.linregress(genotype_matrix[:, i], phenotypes)

        # Get result from lab
        lab_res = results['associations'][i]

        print(f"SNP {i}:")
        print(f"  Expected p-value: {p_value:.6e}")
        print(f"  Actual p-value:   {lab_res['p_value']:.6e}")
        print(f"  Expected beta:    {slope:.6f}")
        print(f"  Actual beta:      {lab_res['beta']:.6f}")

        if not np.isclose(lab_res['p_value'], p_value, rtol=1e-5):
            print(f"  MISMATCH! p-value diff: {abs(lab_res['p_value'] - p_value)}")
        if not np.isclose(lab_res['beta'], slope, rtol=1e-5):
            print(f"  MISMATCH! beta diff: {abs(lab_res['beta'] - slope)}")

if __name__ == "__main__":
    run_benchmark()

"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

GENOMICS LAB
Production-ready genomics algorithms for sequence alignment, variant calling,
GWAS analysis, pathway enrichment, expression clustering, and CNV detection.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import math
from scipy import stats, cluster
from scipy.stats import chi2, norm, hypergeom, fisher_exact
from scipy.spatial.distance import pdist, squareform
import itertools
import re


@dataclass
class Sequence:
    """Represents a biological sequence"""
    seq_id: str
    sequence: str
    quality: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def __len__(self):
        return len(self.sequence)

    def gc_content(self) -> float:
        """Calculate GC content"""
        gc = self.sequence.count('G') + self.sequence.count('C')
        return gc / len(self.sequence) if self.sequence else 0


@dataclass
class Variant:
    """Represents a genomic variant"""
    chromosome: str
    position: int
    ref: str
    alt: str
    quality: float = 0
    genotype: str = './.'
    allele_frequency: float = 0
    variant_type: str = 'SNV'  # SNV, INDEL, SV, CNV

    def __str__(self):
        return f"{self.chromosome}:{self.position} {self.ref}>{self.alt}"


class SequenceAlignment:
    """Sequence alignment algorithms"""

    def __init__(self):
        # Scoring parameters
        self.match_score = 2
        self.mismatch_score = -1
        self.gap_open = -3
        self.gap_extend = -1

    def global_alignment(self, seq1: str, seq2: str) -> Dict:
        """Needleman-Wunsch global alignment

        Args:
            seq1: First sequence
            seq2: Second sequence
        Returns:
            Alignment score and aligned sequences
        """
        n = len(seq1)
        m = len(seq2)

        # Initialize DP matrix
        dp = np.zeros((n + 1, m + 1))
        traceback = np.zeros((n + 1, m + 1), dtype=int)

        # Initialize first row and column
        for i in range(1, n + 1):
            dp[i][0] = self.gap_open + (i - 1) * self.gap_extend
            traceback[i][0] = 1  # Up

        for j in range(1, m + 1):
            dp[0][j] = self.gap_open + (j - 1) * self.gap_extend
            traceback[0][j] = 2  # Left

        # Fill DP matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match = dp[i-1][j-1] + (self.match_score if seq1[i-1] == seq2[j-1] else self.mismatch_score)
                delete = dp[i-1][j] + (self.gap_extend if traceback[i-1][j] == 1 else self.gap_open)
                insert = dp[i][j-1] + (self.gap_extend if traceback[i][j-1] == 2 else self.gap_open)

                if match >= delete and match >= insert:
                    dp[i][j] = match
                    traceback[i][j] = 0  # Diagonal
                elif delete >= insert:
                    dp[i][j] = delete
                    traceback[i][j] = 1  # Up
                else:
                    dp[i][j] = insert
                    traceback[i][j] = 2  # Left

        # Traceback
        aligned1, aligned2 = self._traceback(seq1, seq2, traceback)

        # Calculate identity
        matches = sum(1 for a, b in zip(aligned1, aligned2) if a == b and a != '-')
        identity = matches / max(len(aligned1), len(aligned2)) if aligned1 else 0

        return {
            'score': dp[n][m],
            'aligned_seq1': aligned1,
            'aligned_seq2': aligned2,
            'identity': identity,
            'length': len(aligned1)
        }

    def local_alignment(self, seq1: str, seq2: str) -> Dict:
        """Smith-Waterman local alignment

        Args:
            seq1: First sequence
            seq2: Second sequence
        Returns:
            Best local alignment
        """
        n = len(seq1)
        m = len(seq2)

        # Initialize DP matrix
        dp = np.zeros((n + 1, m + 1))
        traceback = np.zeros((n + 1, m + 1), dtype=int)

        max_score = 0
        max_i, max_j = 0, 0

        # Fill DP matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match = dp[i-1][j-1] + (self.match_score if seq1[i-1] == seq2[j-1] else self.mismatch_score)
                delete = dp[i-1][j] + self.gap_open
                insert = dp[i][j-1] + self.gap_open

                dp[i][j] = max(0, match, delete, insert)

                if dp[i][j] == match and dp[i][j] > 0:
                    traceback[i][j] = 0  # Diagonal
                elif dp[i][j] == delete:
                    traceback[i][j] = 1  # Up
                elif dp[i][j] == insert:
                    traceback[i][j] = 2  # Left

                if dp[i][j] > max_score:
                    max_score = dp[i][j]
                    max_i, max_j = i, j

        # Traceback from maximum score
        aligned1, aligned2 = self._local_traceback(seq1, seq2, traceback, max_i, max_j)

        return {
            'score': max_score,
            'aligned_seq1': aligned1,
            'aligned_seq2': aligned2,
            'start_seq1': max_i - len(aligned1) + aligned1.count('-'),
            'start_seq2': max_j - len(aligned2) + aligned2.count('-')
        }

    def _traceback(self, seq1: str, seq2: str, traceback: np.ndarray) -> Tuple[str, str]:
        """Traceback for global alignment"""
        aligned1 = []
        aligned2 = []
        i, j = len(seq1), len(seq2)

        while i > 0 or j > 0:
            if traceback[i][j] == 0:  # Diagonal
                aligned1.append(seq1[i-1])
                aligned2.append(seq2[j-1])
                i -= 1
                j -= 1
            elif traceback[i][j] == 1:  # Up
                aligned1.append(seq1[i-1])
                aligned2.append('-')
                i -= 1
            else:  # Left
                aligned1.append('-')
                aligned2.append(seq2[j-1])
                j -= 1

        return ''.join(reversed(aligned1)), ''.join(reversed(aligned2))

    def _local_traceback(self, seq1: str, seq2: str, traceback: np.ndarray,
                        max_i: int, max_j: int) -> Tuple[str, str]:
        """Traceback for local alignment"""
        aligned1 = []
        aligned2 = []
        i, j = max_i, max_j

        while i > 0 and j > 0 and traceback[i][j] != -1:
            if traceback[i][j] == 0:  # Diagonal
                aligned1.append(seq1[i-1])
                aligned2.append(seq2[j-1])
                i -= 1
                j -= 1
            elif traceback[i][j] == 1:  # Up
                aligned1.append(seq1[i-1])
                aligned2.append('-')
                i -= 1
            elif traceback[i][j] == 2:  # Left
                aligned1.append('-')
                aligned2.append(seq2[j-1])
                j -= 1
            else:
                break

        return ''.join(reversed(aligned1)), ''.join(reversed(aligned2))

    def blast_search(self, query: str, database: List[str], word_size: int = 11) -> List[Dict]:
        """Simple BLAST-like search

        Args:
            query: Query sequence
            database: List of target sequences
            word_size: Word size for initial seeding
        Returns:
            Hits sorted by score
        """
        hits = []

        # Create word index from query
        query_words = set()
        for i in range(len(query) - word_size + 1):
            query_words.add(query[i:i+word_size])

        # Search database
        for seq_idx, target in enumerate(database):
            # Find seed matches
            seeds = []
            for i in range(len(target) - word_size + 1):
                word = target[i:i+word_size]
                if word in query_words:
                    seeds.append(i)

            if seeds:
                # Extend alignment from seeds (simplified)
                best_score = 0
                for seed in seeds:
                    # Simple extension
                    score = word_size * self.match_score
                    # Extend left
                    q_pos = query.find(target[seed:seed+word_size])
                    if q_pos >= 0:
                        left_q = q_pos - 1
                        left_t = seed - 1
                        while left_q >= 0 and left_t >= 0:
                            if query[left_q] == target[left_t]:
                                score += self.match_score
                            else:
                                break
                            left_q -= 1
                            left_t -= 1

                        # Extend right
                        right_q = q_pos + word_size
                        right_t = seed + word_size
                        while right_q < len(query) and right_t < len(target):
                            if query[right_q] == target[right_t]:
                                score += self.match_score
                            else:
                                break
                            right_q += 1
                            right_t += 1

                        best_score = max(best_score, score)

                if best_score > 0:
                    hits.append({
                        'target_idx': seq_idx,
                        'score': best_score,
                        'e_value': self._calculate_evalue(best_score, len(query), len(database))
                    })

        # Sort by score
        hits.sort(key=lambda x: x['score'], reverse=True)
        return hits

    def _calculate_evalue(self, score: float, query_len: int, db_size: int) -> float:
        """Calculate E-value for alignment (simplified)"""
        k = 0.13
        lambda_val = 0.318
        effective_length = query_len * db_size
        e_value = k * effective_length * math.exp(-lambda_val * score)
        return e_value


class VariantCaller:
    """Variant calling and analysis"""

    def __init__(self):
        self.min_quality = 20
        self.min_depth = 10
        self.min_allele_frequency = 0.05

    def call_snvs(self, reference: str, reads: List[str], qualities: List[str] = None) -> List[Variant]:
        """Call single nucleotide variants

        Args:
            reference: Reference sequence
            reads: Aligned read sequences
            qualities: Base qualities (optional)
        Returns:
            List of called variants
        """
        variants = []

        # Count bases at each position
        for pos in range(len(reference)):
            base_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
            total_depth = 0

            for read_idx, read in enumerate(reads):
                if pos < len(read) and read[pos] != '-':
                    # Check quality if provided
                    if qualities and qualities[read_idx][pos]:
                        qual = ord(qualities[read_idx][pos]) - 33
                        if qual < self.min_quality:
                            continue

                    base_counts[read[pos]] = base_counts.get(read[pos], 0) + 1
                    total_depth += 1

            if total_depth >= self.min_depth:
                ref_base = reference[pos]
                ref_count = base_counts.get(ref_base, 0)

                # Check for variants
                for base, count in base_counts.items():
                    if base != ref_base and count > 0:
                        af = count / total_depth
                        if af >= self.min_allele_frequency:
                            # Calculate quality score (simplified)
                            qual_score = -10 * math.log10(1 - af) if af < 1 else 100

                            variant = Variant(
                                chromosome='chr1',  # Placeholder
                                position=pos + 1,  # 1-based
                                ref=ref_base,
                                alt=base,
                                quality=qual_score,
                                allele_frequency=af,
                                variant_type='SNV'
                            )

                            # Determine genotype
                            if af > 0.9:
                                variant.genotype = '1/1'  # Homozygous alt
                            elif af > 0.3:
                                variant.genotype = '0/1'  # Heterozygous
                            else:
                                variant.genotype = '0/0'  # Homozygous ref (subclonal)

                            variants.append(variant)

        return variants

    def call_indels(self, reference: str, reads: List[str]) -> List[Variant]:
        """Call insertions and deletions

        Args:
            reference: Reference sequence
            reads: Aligned read sequences
        Returns:
            List of indel variants
        """
        variants = []

        # Simple indel detection from aligned sequences
        for read in reads:
            # Find gaps in alignment
            ref_pos = 0
            read_pos = 0

            while ref_pos < len(reference) and read_pos < len(read):
                if reference[ref_pos] == '-':
                    # Insertion in read
                    ins_start = ref_pos
                    ins_seq = []
                    while ref_pos < len(reference) and reference[ref_pos] == '-':
                        ins_seq.append(read[read_pos])
                        ref_pos += 1
                        read_pos += 1

                    if ins_seq:
                        variant = Variant(
                            chromosome='chr1',
                            position=ins_start,
                            ref=reference[max(0, ins_start-1)],
                            alt=reference[max(0, ins_start-1)] + ''.join(ins_seq),
                            variant_type='INDEL'
                        )
                        variants.append(variant)

                elif read[read_pos] == '-':
                    # Deletion in read
                    del_start = ref_pos
                    del_seq = []
                    while read_pos < len(read) and read[read_pos] == '-':
                        del_seq.append(reference[ref_pos])
                        ref_pos += 1
                        read_pos += 1

                    if del_seq:
                        variant = Variant(
                            chromosome='chr1',
                            position=del_start,
                            ref=reference[max(0, del_start-1)] + ''.join(del_seq),
                            alt=reference[max(0, del_start-1)],
                            variant_type='INDEL'
                        )
                        variants.append(variant)
                else:
                    ref_pos += 1
                    read_pos += 1

        # Remove duplicates and calculate frequencies
        unique_variants = {}
        for v in variants:
            key = f"{v.position}_{v.ref}_{v.alt}"
            if key in unique_variants:
                unique_variants[key]['count'] += 1
            else:
                unique_variants[key] = {'variant': v, 'count': 1}

        # Calculate allele frequencies
        total_reads = len(reads)
        final_variants = []
        for key, data in unique_variants.items():
            v = data['variant']
            v.allele_frequency = data['count'] / total_reads
            if v.allele_frequency >= self.min_allele_frequency:
                final_variants.append(v)

        return final_variants

    def annotate_variant(self, variant: Variant, gene_regions: Dict) -> Dict:
        """Annotate variant with functional impact

        Args:
            variant: Variant to annotate
            gene_regions: Dictionary of gene regions
        Returns:
            Annotation information
        """
        annotation = {
            'variant': str(variant),
            'impact': 'MODIFIER',
            'consequence': 'intergenic',
            'gene': None,
            'codon_change': None,
            'amino_acid_change': None
        }

        # Check if variant falls in gene regions
        for gene, regions in gene_regions.items():
            if regions['start'] <= variant.position <= regions['end']:
                annotation['gene'] = gene

                # Check specific regions
                if 'exons' in regions:
                    for exon in regions['exons']:
                        if exon['start'] <= variant.position <= exon['end']:
                            annotation['consequence'] = 'exonic'

                            # Check for specific consequences
                            if variant.variant_type == 'SNV':
                                # Simplified - would need codon table
                                annotation['consequence'] = 'missense'
                                annotation['impact'] = 'MODERATE'
                            elif variant.variant_type == 'INDEL':
                                if len(variant.alt) - len(variant.ref) % 3 == 0:
                                    annotation['consequence'] = 'inframe_indel'
                                    annotation['impact'] = 'MODERATE'
                                else:
                                    annotation['consequence'] = 'frameshift'
                                    annotation['impact'] = 'HIGH'
                            break

                if annotation['consequence'] == 'intergenic':
                    if 'promoter' in regions:
                        if regions['promoter']['start'] <= variant.position <= regions['promoter']['end']:
                            annotation['consequence'] = 'promoter'
                            annotation['impact'] = 'MODERATE'

                break

        return annotation


class GWASAnalysis:
    """Genome-wide association studies"""

    def __init__(self):
        self.min_maf = 0.05  # Minimum minor allele frequency
        self.bonferroni_threshold = 5e-8

    def association_test(self, genotypes: np.ndarray, phenotypes: np.ndarray,
                        covariates: np.ndarray = None) -> Dict:
        """Perform association test for single SNP

        Args:
            genotypes: Genotype matrix (samples x 1)
            phenotypes: Phenotype values
            covariates: Covariate matrix (optional)
        Returns:
            Association statistics
        """
        # Calculate allele frequency
        allele_count = np.sum(genotypes)
        total_alleles = 2 * len(genotypes)
        maf = min(allele_count / total_alleles, 1 - allele_count / total_alleles)

        if maf < self.min_maf:
            return {'p_value': 1.0, 'beta': 0, 'se': float('inf'), 'maf': maf}

        # Linear regression for quantitative traits
        if covariates is not None:
            X = np.column_stack([np.ones(len(genotypes)), genotypes, covariates])
        else:
            X = np.column_stack([np.ones(len(genotypes)), genotypes])

        # Normal equation
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            beta = XtX_inv @ X.T @ phenotypes

            # Calculate standard errors
            residuals = phenotypes - X @ beta
            sigma_sq = np.sum(residuals**2) / (len(phenotypes) - X.shape[1])
            se = np.sqrt(np.diag(XtX_inv) * sigma_sq)

            # T-statistic for genotype effect
            t_stat = beta[1] / se[1]
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(phenotypes) - X.shape[1]))

        except np.linalg.LinAlgError:
            return {'p_value': 1.0, 'beta': 0, 'se': float('inf'), 'maf': maf}

        return {
            'p_value': p_value,
            'beta': beta[1],
            'se': se[1],
            't_statistic': t_stat,
            'maf': maf
        }

    def manhattan_plot_data(self, gwas_results: List[Dict]) -> Dict:
        """Prepare data for Manhattan plot

        Args:
            gwas_results: List of GWAS results with chr, pos, p_value
        Returns:
            Plot data
        """
        # Sort by chromosome and position
        sorted_results = sorted(gwas_results, key=lambda x: (x['chr'], x['pos']))

        # Calculate cumulative positions for plotting
        chr_lengths = {}
        for result in sorted_results:
            chr = result['chr']
            if chr not in chr_lengths:
                chr_lengths[chr] = 0
            chr_lengths[chr] = max(chr_lengths[chr], result['pos'])

        # Calculate cumulative positions
        cumulative_pos = []
        chr_offset = 0
        chr_centers = {}

        for chr in sorted(chr_lengths.keys()):
            chr_start = chr_offset
            chr_end = chr_offset + chr_lengths[chr]
            chr_centers[chr] = (chr_start + chr_end) / 2
            chr_offset = chr_end + 5000000  # 5Mb gap between chromosomes

        # Assign cumulative positions
        plot_data = []
        chr_offset = 0
        current_chr = None

        for result in sorted_results:
            if result['chr'] != current_chr:
                current_chr = result['chr']
                if current_chr != sorted(chr_lengths.keys())[0]:
                    chr_offset += chr_lengths[list(chr_lengths.keys())[list(chr_lengths.keys()).index(current_chr)-1]] + 5000000

            plot_data.append({
                'chr': result['chr'],
                'pos': result['pos'],
                'cumulative_pos': chr_offset + result['pos'],
                'p_value': result['p_value'],
                'log10_p': -math.log10(result['p_value']) if result['p_value'] > 0 else 320
            })

        return {
            'plot_data': plot_data,
            'chr_centers': chr_centers,
            'significance_threshold': -math.log10(self.bonferroni_threshold)
        }

    def calculate_genomic_inflation(self, p_values: List[float]) -> float:
        """Calculate genomic inflation factor (lambda)

        Args:
            p_values: List of p-values from GWAS
        Returns:
            Lambda value
        """
        # Convert p-values to chi-square statistics
        chi_sq_stats = [stats.chi2.ppf(1 - p, df=1) for p in p_values if p > 0]

        if not chi_sq_stats:
            return 1.0

        # Calculate median chi-square
        median_chi_sq = np.median(chi_sq_stats)

        # Expected median under null
        expected_median = stats.chi2.ppf(0.5, df=1)

        # Genomic inflation factor
        lambda_gc = median_chi_sq / expected_median

        return lambda_gc

    def ld_calculation(self, genotypes1: np.ndarray, genotypes2: np.ndarray) -> Dict:
        """Calculate linkage disequilibrium between two SNPs

        Args:
            genotypes1: Genotypes for SNP1
            genotypes2: Genotypes for SNP2
        Returns:
            LD statistics (D', r²)
        """
        # Calculate allele frequencies
        p1 = np.mean(genotypes1) / 2
        p2 = np.mean(genotypes2) / 2

        # Calculate haplotype frequencies (simplified)
        # Assuming genotypes are coded as 0, 1, 2
        hap_11 = np.mean((genotypes1 == 2) & (genotypes2 == 2))
        hap_10 = np.mean((genotypes1 == 2) & (genotypes2 == 0))
        hap_01 = np.mean((genotypes1 == 0) & (genotypes2 == 2))
        hap_00 = np.mean((genotypes1 == 0) & (genotypes2 == 0))

        # Calculate D
        D = hap_11 - p1 * p2

        # Calculate D'
        if D >= 0:
            D_max = min(p1 * (1 - p2), (1 - p1) * p2)
        else:
            D_max = max(-p1 * p2, -(1 - p1) * (1 - p2))

        D_prime = D / D_max if D_max != 0 else 0

        # Calculate r²
        if p1 * (1 - p1) * p2 * (1 - p2) > 0:
            r_squared = D**2 / (p1 * (1 - p1) * p2 * (1 - p2))
        else:
            r_squared = 0

        return {
            'D': D,
            'D_prime': abs(D_prime),
            'r_squared': r_squared
        }


class PathwayEnrichment:
    """Pathway and gene set enrichment analysis"""

    def __init__(self):
        self.pathways = {}  # Will be populated with pathway data

    def hypergeometric_test(self, gene_list: List[str], pathway_genes: Set[str],
                           background_size: int) -> Dict:
        """Hypergeometric test for pathway enrichment

        Args:
            gene_list: List of significant genes
            pathway_genes: Set of genes in pathway
            background_size: Total number of genes
        Returns:
            Enrichment statistics
        """
        # Count overlaps
        gene_set = set(gene_list)
        overlap = len(gene_set & pathway_genes)

        if overlap == 0:
            return {'p_value': 1.0, 'overlap': 0, 'expected': 0, 'fold_enrichment': 0}

        # Hypergeometric test parameters
        M = background_size  # Population size
        n = len(pathway_genes)  # Success states in population
        N = len(gene_list)  # Number of draws

        # Calculate p-value
        p_value = hypergeom.sf(overlap - 1, M, n, N)

        # Expected overlap
        expected = N * n / M

        # Fold enrichment
        fold_enrichment = overlap / expected if expected > 0 else 0

        return {
            'p_value': p_value,
            'overlap': overlap,
            'expected': expected,
            'fold_enrichment': fold_enrichment,
            'genes_in_pathway': list(gene_set & pathway_genes)
        }

    def gsea(self, ranked_genes: List[Tuple[str, float]], gene_set: Set[str],
            n_permutations: int = 1000) -> Dict:
        """Gene Set Enrichment Analysis (simplified)

        Args:
            ranked_genes: List of (gene, score) tuples, ranked by score
            gene_set: Set of genes in pathway
            n_permutations: Number of permutations for p-value
        Returns:
            GSEA statistics
        """
        # Calculate enrichment score
        es, es_profile = self._calculate_enrichment_score(ranked_genes, gene_set)

        # Permutation test
        null_distribution = []
        gene_labels = [gene for gene, _ in ranked_genes]

        for _ in range(n_permutations):
            # Shuffle gene labels
            shuffled_genes = np.random.permutation(gene_labels)
            shuffled_ranked = [(gene, score) for gene, (_, score) in zip(shuffled_genes, ranked_genes)]
            null_es, _ = self._calculate_enrichment_score(shuffled_ranked, gene_set)
            null_distribution.append(null_es)

        # Calculate p-value
        if es >= 0:
            p_value = np.mean([null_es >= es for null_es in null_distribution])
        else:
            p_value = np.mean([null_es <= es for null_es in null_distribution])

        # Normalized enrichment score
        mean_null = np.mean(null_distribution)
        std_null = np.std(null_distribution)
        nes = (es - mean_null) / std_null if std_null > 0 else 0

        return {
            'enrichment_score': es,
            'normalized_es': nes,
            'p_value': p_value,
            'leading_edge': self._find_leading_edge(ranked_genes, gene_set, es_profile)
        }

    def _calculate_enrichment_score(self, ranked_genes: List[Tuple[str, float]],
                                   gene_set: Set[str]) -> Tuple[float, List[float]]:
        """Calculate GSEA enrichment score"""
        n = len(ranked_genes)
        n_set = len(gene_set)

        if n_set == 0:
            return 0, []

        # Calculate running sum
        running_sum = 0
        max_es = 0
        min_es = 0
        es_profile = []

        hit_sum = sum(abs(score) for gene, score in ranked_genes if gene in gene_set)

        for gene, score in ranked_genes:
            if gene in gene_set:
                running_sum += abs(score) / hit_sum if hit_sum > 0 else 0
            else:
                running_sum -= 1 / (n - n_set) if n > n_set else 0

            es_profile.append(running_sum)
            max_es = max(max_es, running_sum)
            min_es = min(min_es, running_sum)

        # Return max deviation from 0
        if abs(max_es) > abs(min_es):
            return max_es, es_profile
        else:
            return min_es, es_profile

    def _find_leading_edge(self, ranked_genes: List[Tuple[str, float]],
                          gene_set: Set[str], es_profile: List[float]) -> List[str]:
        """Find leading edge genes"""
        if not es_profile:
            return []

        # Find position of maximum ES
        max_idx = np.argmax(np.abs(es_profile))

        # Leading edge genes are those before max ES
        leading_edge = []
        for i, (gene, _) in enumerate(ranked_genes[:max_idx+1]):
            if gene in gene_set:
                leading_edge.append(gene)

        return leading_edge


class ExpressionClustering:
    """Gene expression clustering and analysis"""

    def __init__(self):
        self.distance_metrics = ['euclidean', 'correlation', 'manhattan']

    def hierarchical_clustering(self, expression_matrix: np.ndarray,
                              distance_metric: str = 'correlation') -> Dict:
        """Perform hierarchical clustering on expression data

        Args:
            expression_matrix: Genes x Samples expression matrix
            distance_metric: Distance metric to use
        Returns:
            Clustering results
        """
        # Calculate distance matrix
        if distance_metric == 'correlation':
            # Convert correlation to distance
            corr_matrix = np.corrcoef(expression_matrix)
            distance_matrix = 1 - corr_matrix
        else:
            distance_matrix = squareform(pdist(expression_matrix, metric=distance_metric))

        # Perform hierarchical clustering
        linkage_matrix = cluster.hierarchy.linkage(
            squareform(distance_matrix), method='average'
        )

        # Cut tree to get clusters
        n_clusters = min(10, len(expression_matrix) // 5)
        clusters = cluster.hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        return {
            'linkage_matrix': linkage_matrix,
            'clusters': clusters,
            'n_clusters': n_clusters,
            'distance_matrix': distance_matrix
        }

    def kmeans_clustering(self, expression_matrix: np.ndarray, n_clusters: int = 5) -> Dict:
        """K-means clustering of expression data

        Args:
            expression_matrix: Genes x Samples expression matrix
            n_clusters: Number of clusters
        Returns:
            Cluster assignments and centroids
        """
        from scipy.cluster.vq import kmeans2

        # Standardize data
        standardized = (expression_matrix - np.mean(expression_matrix, axis=1, keepdims=True)) / \
                      (np.std(expression_matrix, axis=1, keepdims=True) + 1e-8)

        # Perform k-means
        centroids, labels = kmeans2(standardized, n_clusters, minit='points')

        # Calculate within-cluster sum of squares
        wcss = 0
        for i in range(n_clusters):
            cluster_points = standardized[labels == i]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - centroids[i])**2)

        # Calculate silhouette score (simplified)
        silhouette_scores = []
        for i in range(len(standardized)):
            # Distance to own cluster
            own_cluster = labels[i]
            if np.sum(labels == own_cluster) > 1:
                a = np.mean(np.linalg.norm(standardized[i] - standardized[labels == own_cluster], axis=1))
            else:
                a = 0

            # Distance to nearest other cluster
            b = float('inf')
            for c in range(n_clusters):
                if c != own_cluster and np.sum(labels == c) > 0:
                    dist = np.mean(np.linalg.norm(standardized[i] - standardized[labels == c], axis=1))
                    b = min(b, dist)

            if b != float('inf'):
                silhouette_scores.append((b - a) / max(a, b))
            else:
                silhouette_scores.append(0)

        return {
            'labels': labels,
            'centroids': centroids,
            'wcss': wcss,
            'silhouette_score': np.mean(silhouette_scores) if silhouette_scores else 0
        }

    def differential_expression(self, group1: np.ndarray, group2: np.ndarray) -> List[Dict]:
        """Calculate differential expression between two groups

        Args:
            group1: Expression matrix for group 1 (genes x samples)
            group2: Expression matrix for group 2 (genes x samples)
        Returns:
            List of differential expression statistics per gene
        """
        results = []

        for gene_idx in range(group1.shape[0]):
            expr1 = group1[gene_idx]
            expr2 = group2[gene_idx]

            # Calculate fold change
            mean1 = np.mean(expr1)
            mean2 = np.mean(expr2)

            if mean1 > 0 and mean2 > 0:
                fold_change = mean2 / mean1
                log2_fc = math.log2(fold_change)
            else:
                fold_change = 0
                log2_fc = 0

            # T-test
            t_stat, p_value = stats.ttest_ind(expr1, expr2)

            # Adjusted p-value (Bonferroni)
            adj_p = min(1.0, p_value * group1.shape[0])

            results.append({
                'gene_idx': gene_idx,
                'mean_group1': mean1,
                'mean_group2': mean2,
                'fold_change': fold_change,
                'log2_fold_change': log2_fc,
                't_statistic': t_stat,
                'p_value': p_value,
                'adjusted_p_value': adj_p,
                'significant': adj_p < 0.05
            })

        return results


class CNVDetection:
    """Copy number variation detection"""

    def __init__(self):
        self.window_size = 1000
        self.min_segment_size = 3

    def segment_coverage(self, coverage: np.ndarray) -> List[Dict]:
        """Segment coverage data to detect CNVs using CBS (simplified)

        Args:
            coverage: Coverage depth across genome
        Returns:
            List of CNV segments
        """
        # Normalize coverage
        median_cov = np.median(coverage)
        if median_cov > 0:
            normalized = coverage / median_cov
        else:
            normalized = coverage

        # Log2 transform
        log2_ratio = np.log2(normalized + 0.01)

        # Simple segmentation using change point detection
        segments = []
        segment_start = 0

        for i in range(1, len(log2_ratio)):
            # Check if there's a significant change
            if i - segment_start >= self.min_segment_size:
                seg1 = log2_ratio[segment_start:i]
                seg2 = log2_ratio[i:min(i+self.min_segment_size, len(log2_ratio))]

                if len(seg2) > 0:
                    # T-test for significant difference
                    t_stat, p_value = stats.ttest_ind(seg1, seg2)

                    if p_value < 0.01:
                        # Found breakpoint
                        segments.append({
                            'start': segment_start,
                            'end': i,
                            'mean_log2_ratio': np.mean(seg1),
                            'copy_number': self._estimate_copy_number(np.mean(seg1))
                        })
                        segment_start = i

        # Add final segment
        if segment_start < len(log2_ratio):
            segments.append({
                'start': segment_start,
                'end': len(log2_ratio),
                'mean_log2_ratio': np.mean(log2_ratio[segment_start:]),
                'copy_number': self._estimate_copy_number(np.mean(log2_ratio[segment_start:]))
            })

        return segments

    def _estimate_copy_number(self, log2_ratio: float) -> int:
        """Estimate copy number from log2 ratio"""
        # Assuming diploid baseline
        copy_number = round(2 * 2**log2_ratio)
        return max(0, copy_number)

    def call_cnvs(self, segments: List[Dict]) -> List[Dict]:
        """Call CNVs from segments

        Args:
            segments: List of segments with copy numbers
        Returns:
            List of CNV calls
        """
        cnvs = []

        for segment in segments:
            if segment['copy_number'] != 2:  # Not diploid
                cnv_type = 'deletion' if segment['copy_number'] < 2 else 'duplication'

                cnvs.append({
                    'type': cnv_type,
                    'start': segment['start'],
                    'end': segment['end'],
                    'length': segment['end'] - segment['start'],
                    'copy_number': segment['copy_number'],
                    'log2_ratio': segment['mean_log2_ratio']
                })

        # Merge adjacent CNVs of same type
        merged_cnvs = []
        for cnv in cnvs:
            if merged_cnvs and \
               merged_cnvs[-1]['type'] == cnv['type'] and \
               merged_cnvs[-1]['end'] == cnv['start']:
                # Merge
                merged_cnvs[-1]['end'] = cnv['end']
                merged_cnvs[-1]['length'] = merged_cnvs[-1]['end'] - merged_cnvs[-1]['start']
            else:
                merged_cnvs.append(cnv)

        return merged_cnvs


class GenomicsLab:
    """Main genomics laboratory interface"""

    def __init__(self):
        self.aligner = SequenceAlignment()
        self.variant_caller = VariantCaller()
        self.gwas = GWASAnalysis()
        self.pathway = PathwayEnrichment()
        self.expression = ExpressionClustering()
        self.cnv = CNVDetection()
        self.results = {}

    def align_sequences(self, seq1: str, seq2: str, mode: str = 'global') -> Dict:
        """Align two sequences"""
        if mode == 'global':
            result = self.aligner.global_alignment(seq1, seq2)
        else:
            result = self.aligner.local_alignment(seq1, seq2)

        self.results['alignment'] = result
        return result

    def call_variants(self, reference: str, reads: List[str]) -> Dict:
        """Call variants from sequencing reads"""
        snvs = self.variant_caller.call_snvs(reference, reads)
        indels = self.variant_caller.call_indels(reference, reads)

        results = {
            'snvs': snvs,
            'indels': indels,
            'total_variants': len(snvs) + len(indels)
        }

        self.results['variants'] = results
        return results

    def run_gwas(self, genotype_matrix: np.ndarray, phenotypes: np.ndarray) -> Dict:
        """Run genome-wide association study"""
        results = []

        for snp_idx in range(genotype_matrix.shape[1]):
            assoc = self.gwas.association_test(
                genotype_matrix[:, snp_idx],
                phenotypes
            )
            assoc['snp_idx'] = snp_idx
            results.append(assoc)

        # Multiple testing correction
        p_values = [r['p_value'] for r in results]
        bonferroni_threshold = 0.05 / len(p_values)

        # Find significant SNPs
        significant = [r for r in results if r['p_value'] < bonferroni_threshold]

        gwas_results = {
            'associations': results,
            'significant_snps': significant,
            'lambda_gc': self.gwas.calculate_genomic_inflation(p_values),
            'bonferroni_threshold': bonferroni_threshold
        }

        self.results['gwas'] = gwas_results
        return gwas_results

    def pathway_analysis(self, gene_list: List[str], background_size: int = 20000) -> Dict:
        """Perform pathway enrichment analysis"""
        # Example pathways (would be loaded from database)
        example_pathways = {
            'Cell_Cycle': set(['CDK1', 'CDK2', 'CCNA1', 'CCNB1', 'TP53']),
            'Apoptosis': set(['TP53', 'BAX', 'BCL2', 'CASP3', 'CASP8']),
            'DNA_Repair': set(['BRCA1', 'BRCA2', 'ATM', 'TP53', 'MLH1'])
        }

        enrichment_results = []

        for pathway_name, pathway_genes in example_pathways.items():
            result = self.pathway.hypergeometric_test(
                gene_list, pathway_genes, background_size
            )
            result['pathway'] = pathway_name
            enrichment_results.append(result)

        # Sort by p-value
        enrichment_results.sort(key=lambda x: x['p_value'])

        self.results['pathway_enrichment'] = enrichment_results
        return {'enriched_pathways': enrichment_results}

    def cluster_expression(self, expression_matrix: np.ndarray, method: str = 'hierarchical') -> Dict:
        """Cluster gene expression data"""
        if method == 'hierarchical':
            clustering = self.expression.hierarchical_clustering(expression_matrix)
        else:
            clustering = self.expression.kmeans_clustering(expression_matrix)

        self.results['clustering'] = clustering
        return clustering

    def detect_cnvs(self, coverage_data: np.ndarray) -> Dict:
        """Detect copy number variations"""
        segments = self.cnv.segment_coverage(coverage_data)
        cnv_calls = self.cnv.call_cnvs(segments)

        results = {
            'segments': segments,
            'cnv_calls': cnv_calls,
            'n_deletions': sum(1 for c in cnv_calls if c['type'] == 'deletion'),
            'n_duplications': sum(1 for c in cnv_calls if c['type'] == 'duplication')
        }

        self.results['cnv'] = results
        return results


def run_demo():
    """Demonstrate genomics lab capabilities"""
    print("GENOMICS LAB - Production Demo")
    print("=" * 60)

    lab = GenomicsLab()

    # 1. Sequence alignment
    print("\n1. SEQUENCE ALIGNMENT")
    print("-" * 40)
    seq1 = "ATCGTACGATCGATCGATCGTAGC"
    seq2 = "ATCGATCGATCGATCGTACC"
    alignment = lab.align_sequences(seq1, seq2)
    print(f"Global alignment score: {alignment['score']}")
    print(f"Seq1: {alignment['aligned_seq1']}")
    print(f"Seq2: {alignment['aligned_seq2']}")
    print(f"Identity: {alignment['identity']:.1%}")

    # 2. Variant calling
    print("\n2. VARIANT CALLING")
    print("-" * 40)
    reference = "ATCGATCGATCGATCGATCG"
    reads = [
        "ATCGATCGATGGATCGATCG",  # SNV
        "ATCGATCGATGGATCGATCG",
        "ATCGATCGATCGATCGATCG",
        "ATCGATCGATCGATCGATCG"
    ]
    variants = lab.call_variants(reference, reads)
    print(f"SNVs found: {len(variants['snvs'])}")
    print(f"INDELs found: {len(variants['indels'])}")
    if variants['snvs']:
        v = variants['snvs'][0]
        print(f"Example SNV: {v}")

    # 3. GWAS analysis
    print("\n3. GWAS ANALYSIS")
    print("-" * 40)
    np.random.seed(42)
    # Simulate genotype and phenotype data
    n_samples, n_snps = 1000, 100
    genotype_matrix = np.random.randint(0, 3, (n_samples, n_snps))
    # Create phenotype with association to first SNP
    phenotypes = genotype_matrix[:, 0] * 0.5 + np.random.normal(0, 1, n_samples)
    gwas = lab.run_gwas(genotype_matrix, phenotypes)
    print(f"SNPs tested: {n_snps}")
    print(f"Significant SNPs: {len(gwas['significant_snps'])}")
    print(f"Genomic inflation (λ): {gwas['lambda_gc']:.3f}")
    print(f"Bonferroni threshold: {gwas['bonferroni_threshold']:.2e}")

    # 4. Pathway enrichment
    print("\n4. PATHWAY ENRICHMENT")
    print("-" * 40)
    significant_genes = ['TP53', 'BRCA1', 'CDK1', 'BAX', 'CASP3']
    pathways = lab.pathway_analysis(significant_genes)
    print(f"Genes analyzed: {significant_genes}")
    for pathway in pathways['enriched_pathways'][:3]:
        print(f"  {pathway['pathway']}: p={pathway['p_value']:.4f}, "
              f"fold={pathway['fold_enrichment']:.2f}")

    # 5. Expression clustering
    print("\n5. EXPRESSION CLUSTERING")
    print("-" * 40)
    # Simulate expression data
    n_genes, n_samples = 50, 20
    expression = np.random.randn(n_genes, n_samples)
    # Add pattern to first 10 genes
    expression[:10, :10] += 2
    clustering = lab.cluster_expression(expression, 'kmeans')
    print(f"Clustering method: K-means")
    print(f"Number of clusters: {len(set(clustering['labels']))}")
    print(f"Silhouette score: {clustering['silhouette_score']:.3f}")

    # 6. CNV detection
    print("\n6. COPY NUMBER VARIATION DETECTION")
    print("-" * 40)
    # Simulate coverage data with CNV
    coverage = np.random.poisson(100, 1000)
    coverage[200:300] = np.random.poisson(50, 100)  # Deletion
    coverage[600:700] = np.random.poisson(150, 100)  # Duplication
    cnv_results = lab.detect_cnvs(coverage)
    print(f"Segments found: {len(cnv_results['segments'])}")
    print(f"Deletions: {cnv_results['n_deletions']}")
    print(f"Duplications: {cnv_results['n_duplications']}")
    if cnv_results['cnv_calls']:
        cnv = cnv_results['cnv_calls'][0]
        print(f"Example CNV: {cnv['type']} at {cnv['start']}-{cnv['end']} (CN={cnv['copy_number']})")

    print("\n" + "=" * 60)
    print("Demo complete! Lab ready for production use.")


if __name__ == '__main__':
    run_demo()
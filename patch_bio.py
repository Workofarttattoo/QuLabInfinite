with open("qulab/labs/biology/bioinformatics.py", "r") as f:
    content = f.read()

# I am looking for the place in `bioinformatics.py` to optimize.
# But it seems the current code is buggy: `list(codon_table.keys()).index(amino_acid)` is looking for the amino acid value (e.g., 'I') in the keys of codon_table (e.g. 'ATA').

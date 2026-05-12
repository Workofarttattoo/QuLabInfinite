with open("qulab/labs/biology/metabolomics_lab/metabolomics_engine.py", "r") as f:
    content = f.read()

content = content.replace(
"""            for met, coef in rxn.substrates.items():
                i = metabolites_list.index(met)
                S[i, j] -= coef
            for met, coef in rxn.products.items():
                i = metabolites_list.index(met)
                S[i, j] += coef""",
"""            for met, coef in rxn.substrates.items():
                i = self.metabolites.index(met)
                S[i, j] -= coef
            for met, coef in rxn.products.items():
                i = self.metabolites.index(met)
                S[i, j] += coef""")

with open("qulab/labs/biology/metabolomics_lab/metabolomics_engine.py", "w") as f:
    f.write(content)

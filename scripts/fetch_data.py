import hapi

# Fetch spectroscopy data (HITRAN)
# Molecule: CO2, Isotope: 1, Wavenumber range: 2000-2300 cm-1
hapi.fetch('CO2', 2, 1, 2000, 2300)


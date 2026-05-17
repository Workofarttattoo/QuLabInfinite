# qulab-mcp

**QuLab Infinite as a Model Context Protocol (MCP) server** — 80+ scientific
labs available as tool calls inside Claude, Cursor, and any MCP-compatible AI
assistant.

> Copyright © 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
> All Rights Reserved. PATENT PENDING.

---

## 30-Second Setup

### 1 — Install

```bash
pip install qulab-mcp
```

Or with `uv` (no virtual env needed):

```bash
uvx qulab-mcp   # run directly, nothing installed globally
```

### 2 — Add to Claude Desktop

Open `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows) and add:

```json
{
  "mcpServers": {
    "qulab": {
      "command": "qulab-mcp"
    }
  }
}
```

With `uvx` (no pip install required):

```json
{
  "mcpServers": {
    "qulab": {
      "command": "uvx",
      "args": ["qulab-mcp"]
    }
  }
}
```

### 3 — Add to Claude Code (CLI)

```bash
claude mcp add qulab -- qulab-mcp
```

Restart Claude. QuLab tools are now available.

---

## Remote Access (HTTP+SSE)

Run the server over HTTP so any MCP-compatible client can reach it across a
network — no local install required on the client side.

```bash
# Install with HTTP extras
pip install "qulab-mcp[http]"

# Local network only (safe default)
qulab-mcp --http

# Exposed on all interfaces (put behind a reverse proxy in production)
qulab-mcp --http --host 0.0.0.0 --port 8000

# With API key authentication
QULAB_API_KEY=your-secret-key qulab-mcp --http --host 0.0.0.0
```

Connect any MCP client to: `http://<host>:8000/mcp`

### Adding to Claude Desktop via HTTP

```json
{
  "mcpServers": {
    "qulab-remote": {
      "url": "http://your-server:8000/mcp"
    }
  }
}
```

### Pay-per-Use Architecture

The HTTP mode is designed to sit behind a billing proxy:

```
Client ──► Billing Proxy (FastAPI + Stripe/Paddle)
              ├── validate API token → look up account
              ├── forward to qulab-mcp on 127.0.0.1:8000
              └── on success response → debit usage (per-tool pricing)
```

Run `qulab-mcp --http --host 127.0.0.1` (not exposed publicly) and let the
proxy handle auth + metering. Each tool call is independently billed and
can have per-tool pricing (e.g. nano/quantum tools cost more than lookups).

---

## Available Tools (41)

### Quantum Computing
| Tool | Description |
|------|-------------|
| `quantum_bell_state` | Prepare maximally entangled Bell states |
| `quantum_grovers_search` | Grover's search on n-qubit space |
| `quantum_teleportation` | Full teleportation protocol with fidelity |

### Particle Physics
| Tool | Description |
|------|-------------|
| `particle_cross_section` | QED/QCD cross-sections (e⁺e⁻, pp) |
| `particle_breit_wigner` | Resonance cross-section (Z peak, Higgs, …) |
| `particle_decay_rate` | Partial decay rates and branching ratios |

### Astrophysics
| Tool | Description |
|------|-------------|
| `astro_lane_emden` | Polytropic stellar structure (Chandrasekhar) |
| `astro_cepheid_luminosity` | Leavitt Law period-luminosity relation |
| `astro_schwarzschild` | Schwarzschild metric, redshift, escape velocity |

### Thermodynamics
| Tool | Description |
|------|-------------|
| `thermo_equilibrium_constant` | K from ΔG° at temperature T |
| `thermo_clausius_clapeyron` | Vapour pressure vs temperature |

### Genomics
| Tool | Description |
|------|-------------|
| `genomics_align` | NW global / SW local sequence alignment |
| `genomics_call_variants` | SNV and indel calling from reads |

### Pharmacology
| Tool | Description |
|------|-------------|
| `pharma_pk_model` | One-compartment PK with Cmax, AUC, t½ |
| `pharma_emax_model` | Hill/Emax PD model |

### Chemistry
| Tool | Description |
|------|-------------|
| `chem_molecular_energy` | MM / AM1 / DFT molecular energy from atom coordinates |
| `chem_lattice_energy` | Born-Haber lattice energy (Madelung) |
| `chem_band_gap` | Semiconductor band gap from DFT band edges |
| `chem_crystal_field` | Crystal field splitting Δ for transition metal complexes |
| `chem_redox_potential` | Electrochemical cell potential from standard electrode potentials |
| `chem_activation_energy` | Arrhenius k₂/k₁ ratio from Ea and temperatures |
| `chem_nernst_potential` | Nernst equation E = E° – (RT/nF) ln(Q) |
| `chem_kinetic_rms_velocity` | Kinetic theory v_rms = √(3RT/M) |
| `chem_carnot_efficiency` | Carnot cycle efficiency = 1 – T_cold/T_hot |
| `chem_catalysis_simulate` | Langmuir-Hinshelwood catalytic reaction dynamics |
| `chem_polymer_properties` | Random-walk end-to-end distance, dielectric screening |

### Materials Science (1 619-material database)
| Tool | Description |
|------|-------------|
| `materials_lookup` | Look up a material by name (60+ properties returned) |
| `materials_search` | Filter by category, text, density, strength, modulus |
| `materials_categories` | List all 18 categories and statistics |
| `materials_design` | Voigt bulk/shear/Young's modulus from elastic constant matrix |
| `materials_recommend` | **Design-for-manufacture selector** — rank materials by strength, weight, cost, operating temp, corrosion resistance |

### Nanotechnology
| Tool | Description |
|------|-------------|
| `nano_quantum_dot_bandgap` | Brus equation: quantum confinement bandgap |
| `nano_surface_area` | BET specific surface area for nanoparticles |
| `nano_melting_point_depression` | Gibbs-Thomson melting point depression |
| `nano_ostwald_ripening` | Nanoparticle coarsening simulation |
| `nano_drug_release` | Korsmeyer-Peppas controlled drug release |

### Semiconductor Devices
| Tool | Description |
|------|-------------|
| `semi_mosfet_iv` | MOSFET I-V characteristic curves |
| `semi_threshold_voltage` | Threshold voltage from oxide thickness and doping |
| `semi_pn_junction` | p-n junction built-in potential and depletion width |
| `semi_quantum_well` | Quantum well confined energy levels |
| `semi_diffusion_profile` | Dopant diffusion profile (Fick's law) |

---

## Usage inside Claude

Once configured, you can ask Claude things like:

> *"Run Grover's algorithm to find the number 13 in a 4-qubit space."*

> *"What is the vapour pressure of water at 80°C? Use the Clausius-Clapeyron equation."*

> *"Align these two DNA sequences globally: ATCGATCG and ATCGAAGC."*

> *"What is the equilibrium constant for ATP hydrolysis (ΔG=-32.3 kJ/mol) at 37°C?"*

Claude will call the appropriate QuLab tool and return the result with full
numerical precision.

---

## Running from the Monorepo

If you have the full `QuLabInfinite` repository checked out:

```bash
cd QuLabInfinite
pip install -e packages/qulab-mcp   # editable install
qulab-mcp                            # starts the stdio server
```

The server auto-detects the repo root and imports labs directly.

---

## Development / Adding More Tools

Tools are defined in `qulab_mcp/server.py`. Each tool requires:
1. A `Tool(...)` entry in the `TOOLS` list with `inputSchema`.
2. An `async def _handle_<name>(args) -> CallToolResult` function.
3. An entry in `_HANDLERS`.

---

## License

Proprietary — All Rights Reserved. Patent Pending.

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

## Available Tools

| Tool | Domain | Description |
|------|--------|-------------|
| `quantum_bell_state` | Quantum | Prepare maximally entangled Bell states |
| `quantum_grovers_search` | Quantum | Grover's search on n-qubit space |
| `quantum_teleportation` | Quantum | Full teleportation protocol with fidelity |
| `particle_cross_section` | Particle Physics | QED/QCD cross-sections (e⁺e⁻, pp) |
| `particle_breit_wigner` | Particle Physics | Resonance cross-section (Z peak, Higgs, …) |
| `particle_decay_rate` | Particle Physics | Partial decay rates and branching ratios |
| `astro_lane_emden` | Astrophysics | Polytropic stellar structure (Chandrasekhar) |
| `astro_cepheid_luminosity` | Astrophysics | Leavitt Law period-luminosity relation |
| `astro_schwarzschild` | Astrophysics | Schwarzschild metric, redshift, escape velocity |
| `thermo_equilibrium_constant` | Thermodynamics | K from ΔG° at temperature T |
| `thermo_clausius_clapeyron` | Thermodynamics | Vapour pressure vs temperature |
| `genomics_align` | Genomics | NW global / SW local sequence alignment |
| `genomics_call_variants` | Genomics | SNV and indel calling from reads |
| `pharma_pk_model` | Pharmacology | One-compartment PK with Cmax, AUC, t½ |
| `pharma_emax_model` | Pharmacology | Hill/Emax PD model |

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

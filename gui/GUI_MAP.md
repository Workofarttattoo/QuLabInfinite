# QuLab Infinite - GUI Architecture

Based on user workflow requirements and the existing PySide6 application structure.

## Overview
The application is divided into specialized "Labs" (tabs) for different scientific domains, unified by a common workflow engine ("Stitch").

## Tab Structure

1.  **Physics Lab** (Existing)
    *   **Controls:** Simulation selector, gravity, timestep, restitution.
    *   **Visualization:** 3D PyVista view of particle/body simulations.
    *   **Engine:** `physics_engine` (mechanics, thermodynamics).

2.  **Chemistry Lab** (Existing)
    *   **Controls:** Dataset selector, load button.
    *   **Visualization:** DataFrame viewer for chemical properties.
    *   **Engine:** `chemistry_lab`.

3.  **Materials Lab** (New)
    *   **Controls:** Search bar (filter by name, property), Property filters (Density range, etc.).
    *   **Visualization:**
        *   List/Table of materials.
        *   Detail view for selected material (NIST data, external links).
        *   "Add to Simulation" button.
    *   **Engine:** `materials_lab.materials_database`.

4.  **Stitch Workflow** (New)
    *   **Purpose:** Connect outputs from one lab to inputs of another.
    *   **Visualization:** Node-based graph editor (future) or linear workflow steps.
    *   **Example Workflow:**
        1.  Select Material (Materials Lab) -> "Titanium Alloy"
        2.  Define Geometry (Physics Lab) -> "Turbine Blade"
        3.  Apply Environment (Atmospheric Lab) -> "High Altitude, -50C"
        4.  Run Simulation.

## Technical Implementation
*   **Main Window:** `gui/main_window.py` - `QTabWidget` hosting the labs.
*   **Materials Widget:** `gui/materials_widget.py` (Planned) - Interfaces with `MaterialsDatabase`.
*   **Workflow Widget:** `gui/workflow_widget.py` (Planned) - Visualizes the `stitch` process.

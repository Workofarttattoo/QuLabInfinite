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

3.  **Materials Lab** (Implemented)
    *   **Controls:** Search bar (filter by name, property), Property filters (Density range, etc.).
    *   **Visualization:**
        *   List/Table of materials.
        *   Detail view for selected material (properties + safety lookup).
        *   "Use In Stitch Workflow" action.
    *   **Engine:** `materials_lab.materials_database`.

4.  **Stitch Workflow** (Implemented)
    *   **Purpose:** Connect outputs from one lab to inputs of another.
    *   **Visualization:** Linear workflow builder with node table + JSON export preview.
    *   **Natural Language Mode:** Free-text experiment requests are converted to workflows.
        *   Deterministic templates are used when prompt confidence is high.
        *   Open-ended fallback path is used for arbitrary/novel requests.
        *   `lab_apparatus_simulator` step models beakers, burners, and lab tools from prompt context.
    *   **Example Workflow:**
        1.  Select Material (Materials Lab) -> "Titanium Alloy"
        2.  Add workflow nodes and connect source->target edges
        3.  Validate dataflow graph
        4.  Execute workflow via `WorkflowEngine`.

5.  **Stitch Screens** (New)
    *   **Purpose:** Browse the provided Stitch QuLab reference screen packs.
    *   **Visualization:**
        *   Pack filter + screen list.
        *   Full-size `screen.png` preview.
        *   Paired `code.html` preview panel.
    *   **Source:** `gui/assets/stitch_qulab/*`.

## Technical Implementation
*   **Main Window:** `gui/main_window.py` - `QTabWidget` hosting the labs.
*   **Materials Widget:** `gui/materials_widget.py` - Interfaces with `MaterialsDatabase`.
*   **Workflow Widget:** `gui/workflow_widget.py` - Uses `WorkflowEngine` for create/add/connect/validate/execute/export.
*   **Stitch Screen Browser:** `gui/stitch_screens_widget.py` - Renders imported reference screens from disk.
*   **Cross-tab Integration:** Material selection in Materials Lab updates Stitch workflow context.

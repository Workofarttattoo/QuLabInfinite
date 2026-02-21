# Stitch QuLab Analysis - User Workflow

**Context:** The user requested integration of workflow examples ("stitch qulab"). As the original file was not found, this document reconstructs the intended user experience based on the prompt ("showing different screens for different things").

## Core Concept: "Stitching" Labs
Users do not work in isolation. A discovery in the *Materials Lab* must be "stitched" into a simulation in the *Physics Lab*, and validated against *NIST Standards*.

## User Stories / Screens

### Screen 1: The Inventory (Materials Lab)
*   **User Action:** User searches for "Aerogel".
*   **System Response:** Displays "Airloy X103" with properties (Density: 144 kg/m³, Thermal Cond: 0.014 W/mK).
*   **Workflow:** User selects "Airloy X103" and clicks "Send to Physics Engine".

### Screen 2: The Environment (Atmospheric/Physics Lab)
*   **User Action:** User configures a simulation environment.
*   **System Response:** Shows default parameters (Gravity 9.81, Temp 293K).
*   **Workflow:** User imports "Airloy X103" object. User sets environment to "Mars Atmosphere" (from Atmospheric Lab).

### Screen 3: The Experiment (Simulation View)
*   **User Action:** User runs thermal conductivity test.
*   **System Response:** Visualizer shows heat propagation through the aerogel block in the Mars environment.
*   **Output:** Graph of Temperature vs Time.

### Screen 4: The Report (Stitch Dashboard)
*   **User Action:** User reviews the "Stitched" workflow.
*   **System Response:** Summary showing:
    *   Material: Airloy X103 (Verified against NIST data? Yes)
    *   Environment: Mars (Atmospheric Model v1.2)
    *   Result: Pass/Fail.

## Implementation Plan
1.  **Phase 1 (Current):** Add placeholders in GUI.
2.  **Phase 2:** Implement `MaterialsWidget` with search.
3.  **Phase 3:** Implement data passing (Signal/Slot) between tabs.

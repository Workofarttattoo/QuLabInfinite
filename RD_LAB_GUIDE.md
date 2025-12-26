# QuLabInfinite R&D Engineering Lab
## Complete Guide for Rapid Hardware Development

**Version**: 1.0
**Status**: Production Ready
**Components**: 4 integrated modules + 1M+ materials database

---

## Overview

The R&D Engineering Lab provides rapid prototyping and analysis tools for hardware development, including:

- **PCB Design Module** (`rd_engineering_lab.py`) - Circuit board layout, routing, design rule checking
- **Housing Design Module** (`rd_engineering_lab.py`) - 3D enclosure design with thermal/mechanical analysis
- **Reverse Engineering Module** (`rd_reverse_engineering_module.py`) - Component identification, schematic reconstruction, damage assessment
- **3D Visualization Engine** (`rd_3d_visualizer.py`) - Interactive 3D PCB and housing visualization with thermal mapping
- **Materials Integration** - Access to 1M+ materials database for component and enclosure material properties

---

## Quick Start

### 1. Installation

```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# Materials database should be pre-built
# If not, build it:
python qulab_ingest_materials.py --quick  # For testing
python qulab_ingest_materials.py --standard  # For full features
```

### 2. Start Using the R&D Lab

```python
from rd_engineering_lab import RDEngineeringLab, ComponentType

# Create lab instance
lab = RDEngineeringLab()

# Create a PCB project
pcb = lab.create_pcb_project(
    name="IoT Device v2",
    width_mm=100,
    height_mm=80,
    num_layers=2
)

# Add components
lab.add_component(
    pcb=pcb,
    reference="U1",
    component_type=ComponentType.MICROCONTROLLER,
    part_number="STM32F407VG",
    x_mm=25,
    y_mm=40
)

# View the design
lab.create_3d_viewer(pcb)
```

---

## Module Reference

### PCB Design Module

#### Creating a PCB Project

```python
from rd_engineering_lab import RDEngineeringLab

lab = RDEngineeringLab()

pcb = lab.create_pcb_project(
    name="My Circuit",
    width_mm=100,
    height_mm=80,
    num_layers=2,  # 2 or 4
    thickness_mm=1.6
)
```

**Parameters:**
- `name` (str): Project name
- `width_mm` (float): PCB width in millimeters
- `height_mm` (float): PCB height in millimeters
- `num_layers` (int): 2 (standard) or 4 (advanced)
- `thickness_mm` (float): PCB thickness, typically 1.6mm

#### Adding Components

```python
lab.add_component(
    pcb=pcb,
    reference="U1",
    component_type=ComponentType.MICROCONTROLLER,
    part_number="STM32F407VG",
    x_mm=25,
    y_mm=40,
    rotation_degrees=0,
    on_top=True
)

# Add passive components
lab.add_component(
    pcb=pcb,
    reference="C1",
    component_type=ComponentType.CAPACITOR,
    part_number="100uF 1206",
    x_mm=35,
    y_mm=40
)

# Add connectors
lab.add_component(
    pcb=pcb,
    reference="J1",
    component_type=ComponentType.CONNECTOR,
    part_number="USB-C Right Angle",
    x_mm=10,
    y_mm=75
)
```

**Component Types:**
- `ComponentType.MICROCONTROLLER` - Main processor
- `ComponentType.MEMORY` - RAM, Flash, EEPROM
- `ComponentType.RESISTOR` - Current limiting, pull-up/down
- `ComponentType.CAPACITOR` - Decoupling, filtering
- `ComponentType.INDUCTOR` - Power supply, filtering
- `ComponentType.DIODE` - Protection, rectification
- `ComponentType.TRANSISTOR` - Switching, amplification
- `ComponentType.IC` - General integrated circuit
- `ComponentType.CONNECTOR` - External connections
- `ComponentType.SWITCH` - User input
- `ComponentType.LED` - Status indicators

#### Adding Traces (Signal Connections)

```python
lab.add_trace(
    pcb=pcb,
    from_ref="U1",
    from_pin=14,
    to_ref="C1",
    to_pin=1,
    signal_type="power",
    impedance_ohms=50,
    width_mil=10  # 10 mils = 254 microns
)

# Add a differential pair (for high-speed signals)
lab.add_trace(
    pcb=pcb,
    from_ref="U1",
    from_pin=25,
    to_ref="J1",
    to_pin=1,
    signal_type="differential_data+",
    impedance_ohms=90,
    width_mil=5
)
```

**Signal Types:**
- `"power"` - Power distribution (VCC, VDD)
- `"ground"` - Ground connections
- `"signal"` - Logic signals
- `"clock"` - High-frequency clock signals
- `"differential_data+"` - High-speed differential pairs
- `"analog"` - Analog signals

#### Validating PCB Layout

```python
# Run design rule checks
errors = lab.validate_layout(pcb)

if errors:
    for error in errors:
        print(f"⚠️ {error['type']}: {error['message']}")
        print(f"   Location: {error['component']} at ({error['x']}, {error['y']})")
else:
    print("✓ PCB layout is valid!")
```

**Design Rule Checks:**
- Minimum trace width
- Minimum spacing (trace-to-trace, trace-to-pad)
- Copper-to-edge clearance
- Via hole size validation
- Component spacing
- High-voltage isolation
- Thermal management

#### Bill of Materials (BOM)

```python
# Generate BOM
bom = lab.get_bom(pcb)

# Save to file
lab.export_bom(pcb, format="csv")  # outputs: pcb_name_bom.csv

# Print BOM
print("Bill of Materials")
print("-" * 60)
for entry in bom:
    print(f"{entry['reference']:5} {entry['part_number']:20} "
          f"Qty: {entry['quantity']:3} Cost: ${entry['estimated_cost']:.2f}")

print(f"\nTotal estimated cost: ${sum(e['estimated_cost'] for e in bom):.2f}")
```

**BOM Includes:**
- Reference designators
- Part numbers
- Quantities
- Package types
- Estimated unit cost
- Supplier information
- Availability

#### Exporting PCB Design

```python
# Export to various formats
lab.export_pcb(pcb, format="gerber")   # Gerber files for manufacturing
lab.export_pcb(pcb, format="kicad")    # KiCAD project
lab.export_pcb(pcb, format="pdf")      # PDF schematic
lab.export_pcb(pcb, format="step")     # 3D model
```

---

### Housing Design Module

#### Creating Housing Design

```python
from rd_engineering_lab import RDEngineeringLab, HousingMaterial

lab = RDEngineeringLab()

# Create housing for PCB
housing = lab.create_housing(
    name="Enclosure v2",
    material=HousingMaterial.ALUMINUM_6061,
    width_mm=110,
    height_mm=90,
    depth_mm=30,
    wall_thickness_mm=2.0
)
```

**Available Materials:**
- `HousingMaterial.ALUMINUM_6061` - Excellent thermal, lightweight
- `HousingMaterial.STEEL_304` - Very strong, moderate thermal
- `HousingMaterial.ABS_PLASTIC` - Lightweight, cost-effective
- `HousingMaterial.POLYCARBONATE` - Impact resistant, transparent options
- `HousingMaterial.TITANIUM` - Extreme strength, expensive
- `HousingMaterial.COPPER` - Best thermal conductivity

#### Adding Component Cavities

```python
# Add cavity for display connector
lab.add_component_cavity(
    housing=housing,
    component_ref="J1",  # References PCB component
    cavity_width_mm=15,
    cavity_height_mm=8,
    cavity_depth_mm=3,
    location_x_mm=50,
    location_y_mm=5,
    position="top"
)

# Add mounting posts for PCB
lab.add_mounting_post(
    housing=housing,
    x_mm=15,
    y_mm=15,
    height_mm=5,
    diameter_mm=4,
    thread_type="M3"  # Metric 3mm
)

# Add heat sink mounting for high-power components
lab.add_heat_sink_mount(
    housing=housing,
    component_ref="U1",  # High-power IC
    type="passive_fin",
    surface_area_cm2=25
)
```

#### Thermal Analysis

```python
# Set operating conditions
lab.set_thermal_conditions(
    housing=housing,
    ambient_temp_c=25,
    max_component_power_w=5.0,  # Total power dissipation
    airflow_type="natural"  # "natural", "forced", "liquid"
)

# Analyze thermal performance
thermal_results = lab.analyze_thermal(housing)

print(f"Max internal temp: {thermal_results['max_temp_c']:.1f}°C")
print(f"Hot spot: {thermal_results['hottest_component']}")
print(f"Thermal resistance: {thermal_results['theta_ja']:.2f}°C/W")

if thermal_results['max_temp_c'] > 85:
    print("⚠️ WARNING: Exceeds recommended operating temperature!")
    print("   Recommendations:")
    print("   - Increase airflow")
    print("   - Add heat sink")
    print("   - Reduce internal heat sources")
```

#### Mechanical Analysis

```python
# Analyze structural properties
mechanical_results = lab.analyze_mechanical(
    housing=housing,
    load_type="drop",  # "drop", "vibration", "compression"
    max_load_n=100  # Force in Newtons
)

print(f"Maximum stress: {mechanical_results['max_stress_mpa']:.1f} MPa")
print(f"Safety factor: {mechanical_results['safety_factor']:.2f}")

if mechanical_results['safety_factor'] < 1.5:
    print("⚠️ Low safety factor - consider thicker walls")
```

#### Manufacturing Feasibility

```python
# Check manufacturability
feasibility = lab.check_manufacturability(housing)

print("Manufacturing Analysis:")
print(f"  Wall thickness OK: {feasibility['wall_thickness_ok']}")
print(f"  Undercuts present: {feasibility['has_undercuts']}")
print(f"  Cooling time (injection molding): {feasibility['cooling_time_s']:.1f}s")
print(f"  Estimated cost: ${feasibility['estimated_cost']:.2f}")

if not feasibility['wall_thickness_ok']:
    print("  → Adjust wall thickness for manufacturing")
```

#### Exporting Housing Design

```python
lab.export_housing(housing, format="step")     # 3D CAD model
lab.export_housing(housing, format="stl")      # 3D printing
lab.export_housing(housing, format="dxf")      # 2D drawings
lab.export_housing(housing, format="gcode")    # CNC machining
```

---

### Reverse Engineering Module

#### Identifying Components

```python
from rd_reverse_engineering_module import ReverseEngineeringEngine, ComponentSignature

# Create reverse engineering engine
engine = ReverseEngineeringEngine()

# Identify component from visual signature
signature = ComponentSignature(
    package_type="QFP",
    num_pins=100,
    marking_text="STM32F407",
    color="black",
    estimated_function="ARM Cortex-M4 Microcontroller",
    common_variants=["STM32F405", "STM32F407", "STM32F415", "STM32F417"]
)

identified_part = engine.identify_component_from_signature(signature)
print(f"Identified: {identified_part}")
```

#### Adding Identified Components

```python
from rd_reverse_engineering_module import IdentifiedComponent, ComponentFamily

# Manually add identified component
component = IdentifiedComponent(
    reference_designator="U1",
    part_number="STM32F407VG",
    description="ARM Cortex-M4 Microcontroller, 100-pin",
    family=ComponentFamily.MICROCONTROLLER,
    package="LQFP",
    pins=100,
    voltage_rating=3.3,
    location_x=50,
    location_y=40,
    connected_nets=["VCC", "GND", "SWCLK", "SWDIO"],
    confidence=0.95
)

engine.add_identified_component(component)
```

#### Marking Damage

```python
# Record suspected damage
engine.detect_damage(
    reference="U1",
    damage_type="burn",  # "burn", "corrosion", "crack", "short"
    severity="medium"    # "minor", "medium", "severe"
)

# Component confidence is reduced when damage is detected
```

#### Finding Replacement Parts

```python
# Get suggested replacements
replacements = engine.suggest_replacements("U1")

print("Suggested replacements:")
for part in replacements:
    print(f"  - {part}")
```

#### Extracting Bill of Materials

```python
# Extract BOM from identified components
bom = engine.extract_bom()

print("Bill of Materials (from reverse engineering):")
print("-" * 80)
for entry in bom:
    print(f"{entry['part_number']:20} x{entry['quantity']}")
    print(f"  References: {', '.join(entry['references'])}")
    print(f"  Package: {entry['package']}")
    print()
```

#### Generating Reports

```python
# Generate comprehensive analysis report
report = engine.generate_analysis_report()

# Export as JSON
engine.export_to_json("reverse_eng_report.json")

# Export as KiCAD schematic stub
engine.export_to_kicad_schematic("reverse_eng.kicad_sch")

# Access report data
print(f"Total components identified: {report['statistics']['total_components']}")
print(f"Average confidence: {report['statistics']['avg_confidence']:.2%}")
print(f"Components by type:")
for family, count in report['statistics']['components_by_family'].items():
    print(f"  - {family}: {count}")
```

#### Re-importing Analysis

```python
# Load previous reverse engineering analysis
engine.import_from_json("previous_analysis.json")

# Continue analysis from saved state
```

---

### 3D Visualization Engine

#### Interactive 3D Viewer

```python
from rd_3d_visualizer import Viewer3D

# Create viewer
viewer = Viewer3D(
    width=1200,
    height=800,
    background_color=(0.2, 0.2, 0.2)
)

# Load PCB
viewer.load_pcb(pcb_project)

# Load housing
viewer.load_housing(housing_design)

# Show thermal map (color-coded by temperature)
viewer.show_thermal_map(
    property_name="temperature",
    min_value=25,
    max_value=85,
    colormap="hot"  # "hot", "cool", "viridis", "plasma"
)

# Start interactive viewer
viewer.interactive_view()
```

**Interactive Controls:**
- **Mouse Left + Drag** - Rotate view
- **Mouse Right + Drag** - Pan view
- **Mouse Wheel** - Zoom in/out
- **W** - Toggle wireframe
- **T** - Show thermal map
- **S** - Show stress map
- **M** - Show material map
- **C** - Show component labels
- **F** - Fit all to view
- **Space** - Reset view
- **Q** - Quit viewer

#### Viewing Specific Components

```python
# Highlight and isolate components
viewer.highlight_component("U1")          # Show only U1
viewer.hide_component("C1")               # Hide capacitors
viewer.show_connections("U1")             # Show all traces from U1
viewer.isolate_net("VCC")                 # Show VCC power distribution
```

#### Exporting Views

```python
# Export as static image
viewer.export_image("pcb_top_view.png", angle=(0, 0, 0))
viewer.export_image("pcb_3d_view.png", angle=(45, 45, 0))

# Export as interactive HTML
viewer.export_html("interactive_pcb_viewer.html")

# Export as 3D model
viewer.export_model("pcb_assembly.step")
viewer.export_model("pcb_assembly.stl")
```

#### Creating Technical Documentation

```python
# Generate exploded view
viewer.create_exploded_view(
    direction="up",
    distance_mm=50,
    export_path="exploded_view.png"
)

# Create assembly instruction animation
viewer.create_assembly_animation(
    steps=[
        ("Insert PCB", 1.0),
        ("Attach connectors", 2.0),
        ("Close housing", 1.0)
    ],
    export_video="assembly_instructions.mp4"
)
```

---

## Practical Workflows

### Workflow 1: PCB Design from Scratch

```python
from rd_engineering_lab import RDEngineeringLab, ComponentType

lab = RDEngineeringLab()

# 1. Create project
pcb = lab.create_pcb_project("Wireless Sensor", 80, 60, 2)

# 2. Add core components
lab.add_component(pcb, "U1", ComponentType.MICROCONTROLLER, "nRF52840", 20, 20)
lab.add_component(pcb, "Y1", ComponentType.IC, "32MHz Crystal", 35, 20)
lab.add_component(pcb, "U2", ComponentType.POWER, "TPS62011", 50, 30)

# 3. Add support components
for i, value in enumerate(["100nF", "10uF", "10uF"], 1):
    lab.add_component(pcb, f"C{i}", ComponentType.CAPACITOR, value, 40+i*5, 40)

# 4. Add external connectors
lab.add_component(pcb, "J1", ComponentType.CONNECTOR, "USB-C", 10, 50)
lab.add_component(pcb, "J2", ComponentType.CONNECTOR, "Battery Holder", 70, 50)

# 5. Add power and ground traces
lab.add_trace(pcb, "J2", 1, "U2", 1, "power", width_mil=20)
lab.add_trace(pcb, "U2", 3, "U1", 1, "power", width_mil=15)

# 6. Add signal traces
lab.add_trace(pcb, "J1", 3, "U1", 5, "signal")  # Data+
lab.add_trace(pcb, "J1", 2, "U1", 6, "signal")  # Data-

# 7. Validate
errors = lab.validate_layout(pcb)
if not errors:
    print("✓ Design is valid!")

# 8. Export BOM
bom = lab.get_bom(pcb)
lab.export_bom(pcb, "csv")

# 9. Generate 3D view
viewer = lab.create_3d_viewer(pcb)
```

### Workflow 2: Reverse Engineering a Circuit Board

```python
from rd_reverse_engineering_module import ReverseEngineeringEngine

# 1. Create analysis engine
engine = ReverseEngineeringEngine()

# 2. Identify main processor
u1 = IdentifiedComponent(
    reference_designator="U1",
    part_number="STM32L476RG",
    description="Low-power ARM Cortex-M4",
    family=ComponentFamily.MICROCONTROLLER,
    package="LQFP",
    pins=64,
    voltage_rating=3.3,
    location_x=30,
    location_y=30,
    confidence=0.95
)
engine.add_identified_component(u1)

# 3. Add power regulation
u2 = IdentifiedComponent(
    reference_designator="U2",
    part_number="TPS73633",
    description="LDO Regulator",
    family=ComponentFamily.POWER,
    package="SOT23-5",
    pins=5,
    voltage_rating=5.0,
    location_x=10,
    location_y=20,
    confidence=0.92
)
engine.add_identified_component(u2)

# 4. Add support components
# (typically passives that are harder to identify visually)

# 5. Identify functional blocks
blocks = engine.identify_functional_blocks()

# 6. Generate reports
engine.export_to_json("reverse_engineering_report.json")
engine.export_to_kicad_schematic("reconstructed_schematic.kicad_sch")

# 7. Extract BOM
bom = engine.extract_bom()
print(f"Identified {len(bom)} unique parts")
```

### Workflow 3: Complete Hardware Project

```python
from rd_engineering_lab import RDEngineeringLab, ComponentType, HousingMaterial

lab = RDEngineeringLab()

# PHASE 1: PCB DESIGN
pcb = lab.create_pcb_project("IoT Gateway v2", 150, 100, 4)
# ... add all components and traces ...
lab.validate_layout(pcb)

# PHASE 2: HOUSING DESIGN
housing = lab.create_housing(
    "Gateway Enclosure",
    HousingMaterial.ALUMINUM_6061,
    160, 110, 40,
    wall_thickness_mm=2.5
)

# Add mounting for PCB
for x, y in [(20,20), (20,80), (130,20), (130,80)]:
    lab.add_mounting_post(housing, x, y, 5, 3)

# Add I/O cavities
lab.add_component_cavity(housing, "J1", 15, 10, 3, 10, 95, "top")  # USB
lab.add_component_cavity(housing, "J2", 12, 12, 2, 150, 50, "side")  # Ethernet

# PHASE 3: THERMAL ANALYSIS
lab.set_thermal_conditions(housing, 25, 3.5, "natural")
thermal = lab.analyze_thermal(housing)
print(f"Max temp: {thermal['max_temp_c']:.1f}°C")

# PHASE 4: MECHANICAL ANALYSIS
mechanical = lab.analyze_mechanical(housing, "drop", 50)
print(f"Safety factor: {mechanical['safety_factor']:.1f}")

# PHASE 5: 3D VISUALIZATION
viewer = lab.create_3d_viewer(pcb, housing)
viewer.show_thermal_map("temperature", 25, 85)
viewer.export_html("project_viewer.html")

# PHASE 6: EXPORT FOR MANUFACTURING
lab.export_pcb(pcb, "gerber")
lab.export_bom(pcb, "csv")
lab.export_housing(housing, "step")
```

---

## Integration with Materials Database

The R&D lab integrates with QuLabInfinite's 1M+ materials database for real material properties:

```python
from materials_api import MaterialsDatabase

# Initialize database connection
materials_db = MaterialsDatabase()
materials_db.connect()

# Find suitable enclosure materials
candidates = materials_db.search(
    category="metal",
    min_thermal_conductivity=100,  # W/(m·K)
    max_density=3000,              # kg/m³
    max_cost=50                    # $/kg
)

print(f"Found {len(candidates)} suitable enclosure materials:")
for material in candidates[:5]:
    print(f"  - {material.name}: TC={material.thermal_conductivity}W/(m·K), "
          f"Cost=${material.cost_per_kg}/kg")

# Use material properties in design
selected = candidates[0]
housing = lab.create_housing_with_material(
    name="Thermal Housing",
    material_name=selected.name,
    thermal_conductivity=selected.thermal_conductivity,
    density=selected.density
)
```

---

## Troubleshooting

### Issue: "PCB layout validation failed"

**Solution:**
```python
# Get detailed error information
errors = lab.validate_layout(pcb, verbose=True)

for error in errors:
    print(f"Error at {error['x']}, {error['y']}: {error['message']}")

# Common fixes:
# - Increase trace width: width_mil=10
# - Move components further apart
# - Use multiple layers
# - Add more vias for ground distribution
```

### Issue: "3D viewer crashes on large projects"

**Solution:**
```python
# Simplify the view
viewer.hide_traces()        # Hide all traces
viewer.group_components()   # Simplify component representation
viewer.reduce_quality()     # Lower mesh resolution

# Or use streaming mode for large designs
viewer = Viewer3D(streaming_mode=True)
```

### Issue: "Component not found in database"

**Solution:**
```python
# Manually specify component properties
component = IdentifiedComponent(
    reference_designator="U1",
    part_number="CUSTOM-IC-001",
    description="Custom IC (not in standard database)",
    # ... manually fill all fields ...
    confidence=0.5  # Lower confidence for unknown parts
)
engine.add_identified_component(component)

# Or find close equivalent
similar = engine.suggest_replacements("U1")
```

### Issue: "Thermal analysis shows overheating"

**Solution:**
```python
# Options to reduce temperature:

# 1. Add heat sink
lab.add_heat_sink_mount(housing, "U1", "passive_fin", 30)

# 2. Increase airflow
lab.set_thermal_conditions(housing, 25, 3.5, "forced")  # Add fan

# 3. Improve material thermal conductivity
lab.change_housing_material(housing, HousingMaterial.COPPER)

# 4. Reduce internal power
# - Use lower power components
# - Optimize software algorithms
# - Use sleep modes
```

---

## Performance Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| PCB project creation | <10ms | Instant |
| Add component | <5ms | Per component |
| Layout validation | 50-200ms | Depends on complexity |
| Thermal analysis | 1-5s | 2-layer PCB |
| 3D view render | 100-500ms | Depends on complexity |
| BOM export | <50ms | CSV format |
| Reverse engineering analysis | 100-300ms | Per component |

---

## Best Practices

1. **PCB Design**
   - Start with power distribution (VCC, GND)
   - Add bypass capacitors close to power pins
   - Keep high-speed signals separate
   - Use multiple ground vias
   - Follow manufacturer PCB layout guidelines

2. **Housing Design**
   - Account for component height when sizing
   - Add adequate thermal paths
   - Use standard screw sizes (M2, M3, M4)
   - Consider manufacturing process (injection molding vs. CNC)
   - Add cable strain relief

3. **Reverse Engineering**
   - Document component markings accurately
   - Check for date codes and variants
   - Note any damage or modifications
   - Verify identified components against documentation
   - Test assumptions with component datasheets

4. **3D Visualization**
   - Export high-quality images for documentation
   - Use exploded views for assembly instructions
   - Color-code by function for clarity
   - Generate step-by-step assembly animations

---

## API Reference Summary

### RDEngineeringLab Class

**Key Methods:**
- `create_pcb_project(name, width, height, layers, thickness)` → PCBProject
- `add_component(pcb, ref, type, part_number, x, y, rotation, on_top)` → None
- `add_trace(pcb, from_ref, from_pin, to_ref, to_pin, signal_type, impedance, width)` → None
- `validate_layout(pcb, verbose)` → List[Dict]
- `get_bom(pcb)` → List[Dict]
- `export_bom(pcb, format)` → None
- `export_pcb(pcb, format)` → None
- `create_housing(name, material, width, height, depth, thickness)` → HousingDesign
- `add_component_cavity(housing, ref, width, height, depth, x, y, position)` → None
- `add_mounting_post(housing, x, y, height, diameter, thread_type)` → None
- `set_thermal_conditions(housing, ambient, power, airflow)` → None
- `analyze_thermal(housing)` → Dict
- `analyze_mechanical(housing, load_type, max_load)` → Dict
- `create_3d_viewer(pcb, housing)` → Viewer3D

### ReverseEngineeringEngine Class

**Key Methods:**
- `identify_component_from_signature(signature)` → str (part number)
- `add_identified_component(component)` → str (reference)
- `identify_functional_blocks()` → List[FunctionalBlock]
- `detect_damage(reference, damage_type, severity)` → None
- `suggest_replacements(reference)` → List[str]
- `extract_bom()` → List[Dict]
- `generate_analysis_report()` → Dict
- `export_to_json(filepath)` → None
- `export_to_kicad_schematic(filepath)` → None

### Viewer3D Class

**Key Methods:**
- `load_pcb(pcb_project)` → None
- `load_housing(housing_design)` → None
- `show_thermal_map(property, min_val, max_val, colormap)` → None
- `highlight_component(reference)` → None
- `hide_component(reference)` → None
- `interactive_view()` → None
- `export_image(filepath, angle)` → None
- `export_html(filepath)` → None
- `export_model(filepath)` → None

---

## Support and Examples

**Example Projects:**
- `examples/wireless_sensor_pcb.py` - Complete wireless sensor design
- `examples/industrial_controller_housing.py` - Housing with thermal analysis
- `examples/legacy_board_reverse_engineering.py` - RE a commercial product

**Documentation:**
- PCB Design Guide: See `docs/pcb_design.md`
- Housing Design Guide: See `docs/housing_design.md`
- 3D Visualization: See `docs/visualization.md`

---

**Last Updated**: December 26, 2025
**Version**: 1.0 Production Release

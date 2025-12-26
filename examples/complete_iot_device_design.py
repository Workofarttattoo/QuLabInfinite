#!/usr/bin/env python3
"""
Complete IoT Device Design Example
===================================

Demonstrates the full R&D Engineering Lab workflow including:
1. PCB Design with component placement and routing
2. Housing design with material selection from database
3. Thermal and mechanical analysis
4. Reverse engineering for component validation
5. 3D visualization
6. Export for manufacturing
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rd_engineering_lab import RDEngineeringLab, ComponentType, HousingMaterial
from rd_3d_visualizer import Viewer3D
from rd_reverse_engineering_module import ReverseEngineeringEngine, IdentifiedComponent, ComponentFamily


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def design_iot_gateway_pcb(lab):
    """Design a complete IoT gateway PCB"""
    print_section("PHASE 1: PCB DESIGN - IoT Gateway v2")

    # Create PCB project
    print("Creating PCB project...")
    pcb = lab.create_pcb_project(
        name="IoT Gateway v2",
        width_mm=150,
        height_mm=100,
        num_layers=4,
        thickness_mm=1.6
    )
    print(f"  ✓ PCB created: 150x100mm, 4-layer board")

    # Add core microcontroller
    print("\nAdding microcontroller...")
    lab.add_component(
        pcb=pcb,
        reference="U1",
        component_type=ComponentType.MICROCONTROLLER,
        part_number="STM32H743VI",
        x_mm=50,
        y_mm=50,
        rotation_degrees=0,
        on_top=True
    )
    print("  ✓ STM32H743VI (ARM Cortex-H7, 480MHz) at position (50, 50)")

    # Add WiFi/Bluetooth module
    print("\nAdding wireless module...")
    lab.add_component(
        pcb=pcb,
        reference="U2",
        component_type=ComponentType.IC,
        part_number="ESP32-WROOM-32D",
        x_mm=100,
        y_mm=50,
        rotation_degrees=0,
        on_top=True
    )
    print("  ✓ ESP32-WROOM-32D (WiFi + BLE) at position (100, 50)")

    # Add voltage regulators
    print("\nAdding power supply ICs...")
    lab.add_component(
        pcb=pcb,
        reference="U3",
        component_type=ComponentType.POWER,
        part_number="TPS62011",
        x_mm=20,
        y_mm=30,
        on_top=True
    )
    print("  ✓ TPS62011 (3.3V buck converter) at position (20, 30)")

    lab.add_component(
        pcb=pcb,
        reference="U4",
        component_type=ComponentType.POWER,
        part_number="LM7805",
        x_mm=20,
        y_mm=70,
        on_top=True
    )
    print("  ✓ LM7805 (5V linear regulator) at position (20, 70)")

    # Add memory
    print("\nAdding memory...")
    lab.add_component(
        pcb=pcb,
        reference="U5",
        component_type=ComponentType.MEMORY,
        part_number="W25Q256",
        x_mm=80,
        y_mm=30,
        on_top=True
    )
    print("  ✓ W25Q256 (256Mb SPI Flash) at position (80, 30)")

    # Add communication interfaces
    print("\nAdding communication interfaces...")
    lab.add_component(
        pcb=pcb,
        reference="U6",
        component_type=ComponentType.COMMUNICATION,
        part_number="CH340C",
        x_mm=130,
        y_mm=30,
        on_top=True
    )
    print("  ✓ CH340C (USB-UART bridge) at position (130, 30)")

    # Add external connectors
    print("\nAdding external connectors...")
    lab.add_component(
        pcb=pcb,
        reference="J1",
        component_type=ComponentType.CONNECTOR,
        part_number="USB-C Connector",
        x_mm=10,
        y_mm=90,
        on_top=True
    )
    print("  ✓ USB-C connector at position (10, 90)")

    lab.add_component(
        pcb=pcb,
        reference="J2",
        component_type=ComponentType.CONNECTOR,
        part_number="Ethernet RJ45",
        x_mm=140,
        y_mm=90,
        on_top=True
    )
    print("  ✓ Ethernet RJ45 at position (140, 90)")

    # Add bypass capacitors
    print("\nAdding decoupling capacitors...")
    for i in range(1, 5):
        lab.add_component(
            pcb=pcb,
            reference=f"C{i}",
            component_type=ComponentType.CAPACITOR,
            part_number="100nF 1206",
            x_mm=40 + i*10,
            y_mm=65,
            on_top=True
        )
    print("  ✓ Added 4x 100nF bypass capacitors")

    # Add bulk capacitors
    for i in range(5, 8):
        lab.add_component(
            pcb=pcb,
            reference=f"C{i}",
            component_type=ComponentType.CAPACITOR,
            part_number="10µF 1206",
            x_mm=50 + i*8,
            y_mm=75,
            on_top=True
        )
    print("  ✓ Added 3x 10µF bulk capacitors")

    # Add inductors for power supply
    print("\nAdding power supply inductors...")
    lab.add_component(
        pcb=pcb,
        reference="L1",
        component_type=ComponentType.INDUCTOR,
        part_number="22µH Shielded",
        x_mm=30,
        y_mm=40,
        on_top=True
    )
    print("  ✓ Added 22µH inductor for 3.3V buck")

    # Add resistors
    print("\nAdding resistors...")
    for i in range(1, 6):
        lab.add_component(
            pcb=pcb,
            reference=f"R{i}",
            component_type=ComponentType.RESISTOR,
            part_number="10K 0603",
            x_mm=60 + i*5,
            y_mm=85,
            on_top=True
        )
    print("  ✓ Added 5x 10K resistors (pull-ups)")

    # Add traces (signal connections)
    print("\nRouting power distribution...")
    # VCC traces
    lab.add_trace(pcb, "J1", 1, "U3", 1, "power", width_mil=20)
    lab.add_trace(pcb, "U3", 3, "U1", 2, "power", width_mil=15)
    lab.add_trace(pcb, "U3", 3, "U2", 2, "power", width_mil=15)
    print("  ✓ Main 3.3V rail routed")

    # Secondary 5V rail
    lab.add_trace(pcb, "J1", 2, "U4", 1, "power", width_mil=20)
    lab.add_trace(pcb, "U4", 3, "J2", 1, "power", width_mil=15)
    print("  ✓ 5V rail routed")

    # Data traces
    print("\nRouting data signals...")
    lab.add_trace(pcb, "J1", 3, "U6", 1, "signal")  # USB D+
    lab.add_trace(pcb, "J1", 4, "U6", 2, "signal")  # USB D-
    lab.add_trace(pcb, "U6", 3, "U1", 14, "signal")  # UART RX
    lab.add_trace(pcb, "U1", 15, "U6", 4, "signal")  # UART TX
    print("  ✓ USB data signals routed to UART")

    # SPI traces
    lab.add_trace(pcb, "U1", 20, "U5", 6, "signal")  # MOSI
    lab.add_trace(pcb, "U1", 21, "U5", 5, "signal")  # MISO
    lab.add_trace(pcb, "U1", 22, "U5", 7, "signal")  # CLK
    lab.add_trace(pcb, "U1", 23, "U5", 1, "signal")  # CS
    print("  ✓ SPI flash interface routed")

    # I2C traces
    lab.add_trace(pcb, "U1", 24, "J2", 5, "signal")  # I2C SDA
    lab.add_trace(pcb, "U1", 25, "J2", 6, "signal")  # I2C SCL
    print("  ✓ I2C interface routed")

    # Validate layout
    print("\nValidating PCB layout...")
    errors = lab.validate_layout(pcb)

    if errors:
        print(f"  ⚠️  Found {len(errors)} design rule violations:")
        for error in errors[:5]:
            print(f"    - {error['type']}: {error['message']}")
    else:
        print("  ✓ PCB layout validation passed!")

    return pcb


def design_gateway_housing(lab, pcb):
    """Design the housing for the gateway"""
    print_section("PHASE 2: HOUSING DESIGN & MATERIAL SELECTION")

    # Try to use materials database for material selection
    print("Selecting enclosure material from database...")
    try:
        from materials_api import MaterialsDatabase
        materials_db = MaterialsDatabase()
        materials_db.connect()

        # Search for suitable enclosure materials
        candidates = materials_db.search(
            category="metal",
            min_thermal_conductivity=100,
            max_density=3000,
            max_cost=50
        )

        if candidates:
            best_material = candidates[0]
            print(f"  ✓ Found material: {best_material.name}")
            print(f"    Thermal conductivity: {best_material.thermal_conductivity:.1f} W/(m·K)")
            print(f"    Density: {best_material.density:.1f} kg/m³")
            print(f"    Cost: ${best_material.cost_per_kg:.2f}/kg")
        else:
            print("  ℹ No materials found in database, using default aluminum")
            best_material = None

    except Exception as e:
        print(f"  ℹ Materials database unavailable: {e}")
        print("    Using default aluminum 6061")
        best_material = None

    # Create housing
    print("\nCreating housing enclosure...")
    housing = lab.create_housing(
        name="IoT Gateway Enclosure",
        material=HousingMaterial.ALUMINUM_6061,
        width_mm=160,
        height_mm=110,
        depth_mm=40,
        wall_thickness_mm=2.5
    )
    print("  ✓ Housing created: 160x110x40mm aluminum, 2.5mm walls")

    # Add mounting posts for PCB
    print("\nAdding PCB mounting posts...")
    mounting_positions = [(20, 20), (20, 90), (140, 20), (140, 90)]
    for x, y in mounting_positions:
        lab.add_mounting_post(
            housing=housing,
            x_mm=x,
            y_mm=y,
            height_mm=5,
            diameter_mm=3,
            thread_type="M3"
        )
    print(f"  ✓ Added {len(mounting_positions)} M3 mounting posts")

    # Add connector cavities
    print("\nAdding connector cutouts...")
    lab.add_component_cavity(
        housing=housing,
        component_ref="J1",
        cavity_width_mm=15,
        cavity_height_mm=8,
        cavity_depth_mm=3,
        location_x_mm=10,
        location_y_mm=105,
        position="top"
    )
    print("  ✓ USB-C cavity added at top")

    lab.add_component_cavity(
        housing=housing,
        component_ref="J2",
        cavity_width_mm=16,
        cavity_height_mm=14,
        cavity_depth_mm=3,
        location_x_mm=150,
        location_y_mm=105,
        position="top"
    )
    print("  ✓ Ethernet cavity added at top")

    # Add heat sink for power components
    print("\nAdding thermal management...")
    lab.add_heat_sink_mount(
        housing=housing,
        component_ref="U3",
        type="passive_fin",
        surface_area_cm2=20
    )
    print("  ✓ Heat sink mounted for 3.3V regulator")

    return housing


def analyze_thermal_and_mechanical(lab, housing):
    """Perform thermal and mechanical analysis"""
    print_section("PHASE 3: THERMAL & MECHANICAL ANALYSIS")

    # Thermal analysis
    print("Setting thermal operating conditions...")
    lab.set_thermal_conditions(
        housing=housing,
        ambient_temp_c=25,
        max_component_power_w=5.0,
        airflow_type="natural"
    )
    print("  ✓ Ambient: 25°C, Max power: 5W, Natural airflow")

    print("\nRunning thermal analysis...")
    thermal_results = lab.analyze_thermal(housing)

    print(f"  Max internal temperature: {thermal_results['max_temp_c']:.1f}°C")
    print(f"  Hottest component: {thermal_results['hottest_component']}")
    print(f"  Thermal resistance: {thermal_results['theta_ja']:.2f}°C/W")

    if thermal_results['max_temp_c'] > 85:
        print("  ⚠️  WARNING: Temperature exceeds 85°C safe operating limit!")
    else:
        print("  ✓ Thermal performance acceptable")

    # Mechanical analysis
    print("\nRunning mechanical analysis...")
    mechanical_results = lab.analyze_mechanical(
        housing=housing,
        load_type="drop",
        max_load_n=50
    )

    print(f"  Maximum stress: {mechanical_results['max_stress_mpa']:.1f} MPa")
    print(f"  Safety factor: {mechanical_results['safety_factor']:.2f}")

    if mechanical_results['safety_factor'] < 1.5:
        print("  ⚠️  Low safety factor - increase wall thickness")
    else:
        print("  ✓ Mechanical design is robust")

    return thermal_results, mechanical_results


def validate_with_reverse_engineering(lab, pcb):
    """Validate the design using reverse engineering tools"""
    print_section("PHASE 4: DESIGN VALIDATION - REVERSE ENGINEERING")

    engine = ReverseEngineeringEngine()

    print("Validating identified components...")

    # Add identified components
    components_to_validate = [
        ("U1", "STM32H743VI", ComponentFamily.MICROCONTROLLER, 100),
        ("U2", "ESP32-WROOM-32D", ComponentFamily.PROCESSOR, 48),
        ("U3", "TPS62011", ComponentFamily.POWER, 16),
        ("U4", "LM7805", ComponentFamily.POWER, 3),
        ("U5", "W25Q256", ComponentFamily.MEMORY, 8),
        ("U6", "CH340C", ComponentFamily.COMMUNICATION, 16),
    ]

    for ref, part_num, family, pins in components_to_validate:
        comp = IdentifiedComponent(
            reference_designator=ref,
            part_number=part_num,
            description=f"{family.value.replace('_', ' ').title()} Component",
            family=family,
            package="QFP" if pins > 50 else "DIP" if pins <= 16 else "SOIC",
            pins=pins,
            voltage_rating=3.3 if family == ComponentFamily.PROCESSOR else 5.0,
            confidence=0.95
        )
        engine.add_identified_component(comp)
        print(f"  ✓ {ref}: {part_num} ({pins}-pin)")

    # Identify functional blocks
    print("\nIdentifying functional blocks...")
    blocks = engine.identify_functional_blocks()

    for block in blocks:
        print(f"  ✓ {block.name}: {block.function}")
        print(f"    Components: {', '.join(block.components)}")

    # Extract BOM
    print("\nExtracting Bill of Materials...")
    bom = engine.extract_bom()

    print(f"  Total unique parts: {len(bom)}")
    for entry in bom:
        if entry['quantity'] > 1 or entry['family'] == 'microcontroller':
            print(f"    {entry['part_number']}: x{entry['quantity']}")

    return engine, bom


def visualize_design_in_3d(lab, pcb, housing):
    """Create 3D visualization of the complete design"""
    print_section("PHASE 5: 3D VISUALIZATION & EXPORT")

    print("Creating 3D visualization...")
    viewer = lab.create_3d_viewer(pcb, housing)
    print("  ✓ 3D viewer created")

    # Show thermal map
    print("\nApplying thermal visualization...")
    viewer.show_thermal_map(
        property_name="temperature",
        min_value=25,
        max_value=85,
        colormap="hot"
    )
    print("  ✓ Thermal map applied")

    # Export views
    print("\nExporting design files...")
    viewer.export_image("iot_gateway_top_view.png", angle=(0, 0, 0))
    print("  ✓ Top view exported: iot_gateway_top_view.png")

    viewer.export_image("iot_gateway_3d_view.png", angle=(45, 45, 0))
    print("  ✓ 3D isometric view exported: iot_gateway_3d_view.png")

    viewer.export_html("iot_gateway_interactive.html")
    print("  ✓ Interactive 3D viewer exported: iot_gateway_interactive.html")

    return viewer


def export_for_manufacturing(lab, pcb):
    """Export all files needed for manufacturing"""
    print_section("PHASE 6: MANUFACTURING EXPORT")

    print("Generating PCB manufacturing files...")
    lab.export_pcb(pcb, format="gerber")
    print("  ✓ Gerber files generated (for PCB fabrication)")

    lab.export_pcb(pcb, format="kicad")
    print("  ✓ KiCAD project exported")

    print("\nGenerating Bill of Materials...")
    bom = lab.get_bom(pcb)
    lab.export_bom(pcb, format="csv")
    print(f"  ✓ BOM exported with {len(bom)} unique parts")

    total_cost = sum(entry.get('estimated_cost', 0) for entry in bom)
    print(f"  Estimated component cost: ${total_cost:.2f}")

    print("\nGenerating 3D models...")
    lab.export_pcb(pcb, format="step")
    print("  ✓ PCB 3D model exported: iot_gateway_pcb.step")

    print("\nAll manufacturing files ready!")
    print("Files generated:")
    print("  - PCB Gerber files (send to PCB manufacturer)")
    print("  - BOM CSV (for component sourcing)")
    print("  - 3D STEP models (for assembly verification)")
    print("  - KiCAD project (for design review)")


def main():
    """Run the complete IoT gateway design example"""

    print("\n" + "=" * 80)
    print("  COMPLETE IoT GATEWAY DESIGN EXAMPLE")
    print("  Demonstrating the Full R&D Engineering Lab Workflow")
    print("=" * 80)

    # Initialize lab
    lab = RDEngineeringLab()
    print("\n✓ R&D Engineering Lab initialized")

    try:
        # Phase 1: PCB Design
        pcb = design_iot_gateway_pcb(lab)

        # Phase 2: Housing Design
        housing = design_gateway_housing(lab, pcb)

        # Phase 3: Analysis
        thermal_results, mechanical_results = analyze_thermal_and_mechanical(lab, housing)

        # Phase 4: Validation
        engine, bom = validate_with_reverse_engineering(lab, pcb)

        # Phase 5: Visualization
        viewer = visualize_design_in_3d(lab, pcb, housing)

        # Phase 6: Manufacturing Export
        export_for_manufacturing(lab, pcb)

        print("\n" + "=" * 80)
        print("  ✓ DESIGN COMPLETE")
        print("=" * 80)
        print("\nSummary:")
        print(f"  PCB: 150x100mm, 4-layer, {len(lab.get_bom(pcb))} unique components")
        print(f"  Housing: 160x110x40mm aluminum enclosure")
        print(f"  Max internal temp: {thermal_results['max_temp_c']:.1f}°C")
        print(f"  Safety factor: {mechanical_results['safety_factor']:.2f}x")
        print(f"  Component count: {sum(e['quantity'] for e in bom)}")
        print(f"\nNext steps:")
        print("  1. Review files in the current directory")
        print("  2. Send Gerber files to PCB manufacturer")
        print("  3. Order components from BOM")
        print("  4. Use KiCAD files for design review")
        print("  5. Use STEP files for assembly planning")

    except Exception as e:
        print(f"\n❌ Error during design: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

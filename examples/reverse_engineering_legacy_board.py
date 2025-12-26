#!/usr/bin/env python3
"""
Reverse Engineering Legacy Circuit Board Example
================================================

Demonstrates how to use the Reverse Engineering Module to:
1. Identify components from a legacy/unknown circuit board
2. Reconstruct the schematic
3. Detect damage and recommend repairs
4. Generate a usable netlist and BOM
5. Find component replacements
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rd_reverse_engineering_module import (
    ReverseEngineeringEngine,
    IdentifiedComponent,
    ComponentFamily,
    ComponentSignature,
    SignalType,
)


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def identify_main_processor(engine):
    """Identify the main processor on the legacy board"""
    print_section("STEP 1: IDENTIFY MAIN PROCESSOR")

    print("Analyzing main IC (largest component with many pins)...")
    print("  Visual characteristics:")
    print("    - Package type: QFP, 100-pin")
    print("    - Marking text: 'STM32L476'")
    print("    - Color: Black epoxy")

    # Create signature from visual inspection
    signature = ComponentSignature(
        package_type="QFP",
        num_pins=100,
        marking_text="STM32L476",
        color="black",
        estimated_function="ARM Cortex-M4L Microcontroller",
        common_variants=["STM32L476RG", "STM32L476RC", "STM32L476RE"]
    )

    # Identify from signature
    identified_part = engine.identify_component_from_signature(signature)
    print(f"\n  Identified: {identified_part}")

    # Add to project
    u1 = IdentifiedComponent(
        reference_designator="U1",
        part_number="STM32L476RG",
        description="Ultra-low-power ARM Cortex-M4L microcontroller",
        family=ComponentFamily.MICROCONTROLLER,
        package="LQFP",
        pins=100,
        voltage_rating=3.3,
        location_x=50,
        location_y=50,
        connected_nets=["VDD", "GND", "PB10", "PB11", "PB12"],
        confidence=0.95
    )

    engine.add_identified_component(u1)
    print(f"  ✓ Added U1: {u1.part_number}")

    return u1


def identify_power_supply(engine):
    """Identify the power supply components"""
    print_section("STEP 2: IDENTIFY POWER SUPPLY")

    print("Scanning for voltage regulation...")

    # LDO regulator
    print("\n  Found small regulator IC (SOT23-5 package):")
    u2 = IdentifiedComponent(
        reference_designator="U2",
        part_number="LD1117-3.3",
        description="3.3V Low Dropout Linear Regulator",
        family=ComponentFamily.POWER,
        package="SOT23-5",
        pins=5,
        voltage_rating=5.0,
        location_x=20,
        location_y=30,
        connected_nets=["VIN", "VDD", "GND"],
        confidence=0.92
    )
    engine.add_identified_component(u2)
    print(f"    ✓ U2: {u2.part_number}")

    # Input reverse polarity diode
    print("\n  Found reverse polarity protection diode (DO-214):")
    d1 = IdentifiedComponent(
        reference_designator="D1",
        part_number="1N4007",
        description="General Purpose Rectifier Diode",
        family=ComponentFamily.DISCRETE,
        package="DO-214AC",
        pins=2,
        voltage_rating=1000,
        location_x=10,
        location_y=10,
        connected_nets=["VIN", "GND"],
        confidence=0.88
    )
    engine.add_identified_component(d1)
    print(f"    ✓ D1: {d1.part_number}")

    # Capacitors for filtering
    print("\n  Found several capacitors in power circuit:")
    capacitors = [
        ("C1", "100µF 1206", 15, 20),
        ("C2", "10µF 1206", 25, 20),
        ("C3", "100nF 0603", 30, 28),
        ("C4", "100nF 0603", 40, 28),
    ]

    for ref, part, x, y in capacitors:
        cap = IdentifiedComponent(
            reference_designator=ref,
            part_number=part,
            description="Electrolytic Capacitor" if "µF" in part else "Ceramic Capacitor",
            family=ComponentFamily.PASSIVE,
            package="1206" if "µ" in part else "0603",
            pins=2,
            location_x=x,
            location_y=y,
            connected_nets=["VDD", "GND"],
            confidence=0.90
        )
        engine.add_identified_component(cap)
        print(f"    ✓ {ref}: {part}")


def identify_communication_interface(engine):
    """Identify communication interfaces"""
    print_section("STEP 3: IDENTIFY COMMUNICATION INTERFACE")

    print("Scanning for external communication...")

    # USB to Serial converter
    print("\n  Found USB-UART interface IC (SSOP-16 package):")
    u3 = IdentifiedComponent(
        reference_designator="U3",
        part_number="CH340C",
        description="USB to Serial UART Converter",
        family=ComponentFamily.COMMUNICATION,
        package="SSOP",
        pins=16,
        location_x=80,
        location_y=20,
        connected_nets=["USB_D+", "USB_D-", "UART_TX", "UART_RX"],
        confidence=0.93
    )
    engine.add_identified_component(u3)
    print(f"    ✓ U3: {u3.part_number}")

    # USB connector
    print("\n  Found USB connector (Type-B):")
    j1 = IdentifiedComponent(
        reference_designator="J1",
        part_number="USB Type-B Connector",
        description="Vertical USB Type-B Connector",
        family=ComponentFamily.CONNECTOR,
        package="USB-B",
        pins=4,
        location_x=100,
        location_y=10,
        connected_nets=["USB_5V", "GND", "USB_D+", "USB_D-"],
        confidence=0.95
    )
    engine.add_identified_component(j1)
    print(f"    ✓ J1: {j1.part_number}")

    # Series resistors for USB signal integrity
    print("\n  Found USB signal protection resistors (0603):")
    for i, (ref, net) in enumerate([("R1", "USB_D+"), ("R2", "USB_D-")], 1):
        r = IdentifiedComponent(
            reference_designator=ref,
            part_number="22Ω 0603",
            description="USB Series Protection Resistor",
            family=ComponentFamily.RESISTOR,
            package="0603",
            pins=2,
            location_x=85 + i*5,
            location_y=15,
            connected_nets=["USB", net],
            confidence=0.85
        )
        engine.add_identified_component(r)
        print(f"    ✓ {ref}: 22Ω")


def identify_clock_and_reset(engine):
    """Identify clock and reset circuits"""
    print_section("STEP 4: IDENTIFY CLOCK & RESET CIRCUITS")

    print("Analyzing timing and reset circuits...")

    # Crystal oscillator
    print("\n  Found crystal oscillator (SMD, marking: '32.768K'):")
    y1 = IdentifiedComponent(
        reference_designator="Y1",
        part_number="32.768kHz Crystal",
        description="Low-frequency watch crystal for RTC",
        family=ComponentFamily.IC,
        package="SMD-HC45",
        pins=2,
        location_x=60,
        location_y=30,
        connected_nets=["XTAL_IN", "XTAL_OUT"],
        confidence=0.92
    )
    engine.add_identified_component(y1)
    print(f"    ✓ Y1: {y1.part_number}")

    # Load capacitors for crystal
    print("\n  Found crystal load capacitors (0603):")
    for ref in ["C5", "C6"]:
        cap = IdentifiedComponent(
            reference_designator=ref,
            part_number="22pF 0603",
            description="Crystal Load Capacitor",
            family=ComponentFamily.PASSIVE,
            package="0603",
            pins=2,
            location_x=60,
            location_y=35 if ref == "C5" else 40,
            connected_nets=["XTAL", "GND"],
            confidence=0.88
        )
        engine.add_identified_component(cap)
        print(f"    ✓ {ref}: 22pF")

    # Reset button
    print("\n  Found reset pushbutton (SMD tactile switch):")
    sw1 = IdentifiedComponent(
        reference_designator="SW1",
        part_number="SMD Tactile Button",
        description="System Reset Pushbutton",
        family=ComponentFamily.SWITCH,
        package="SMD",
        pins=2,
        location_x=75,
        location_y=60,
        connected_nets=["RST", "GND"],
        confidence=0.90
    )
    engine.add_identified_component(sw1)
    print(f"    ✓ SW1: {sw1.part_number}")

    # Reset pull-up resistor
    print("\n  Found reset pull-up (10K 0603):")
    r3 = IdentifiedComponent(
        reference_designator="R3",
        part_number="10K 0603",
        description="Reset Pull-up Resistor",
        family=ComponentFamily.RESISTOR,
        package="0603",
        pins=2,
        location_x=70,
        location_y=55,
        connected_nets=["RST", "VDD"],
        confidence=0.92
    )
    engine.add_identified_component(r3)
    print(f"    ✓ R3: {r3.part_number}")


def identify_user_interface(engine):
    """Identify user interface components"""
    print_section("STEP 5: IDENTIFY USER INTERFACE")

    print("Scanning for buttons and indicators...")

    # LED indicator
    print("\n  Found status LED (red, 1206 package):")
    d2 = IdentifiedComponent(
        reference_designator="D2",
        part_number="Red LED 1206",
        description="Status Indicator LED",
        family=ComponentFamily.LED,
        package="1206",
        pins=2,
        location_x=110,
        location_y=30,
        connected_nets=["PA5", "GND"],
        confidence=0.93
    )
    engine.add_identified_component(d2)
    print(f"    ✓ D2: {d2.part_number}")

    # LED current limiting resistor
    print("\n  Found LED current limiting resistor (1K 0603):")
    r4 = IdentifiedComponent(
        reference_designator="R4",
        part_number="1K 0603",
        description="LED Current Limiting Resistor",
        family=ComponentFamily.RESISTOR,
        package="0603",
        pins=2,
        location_x=105,
        location_y=25,
        connected_nets=["PA5", "D2"],
        confidence=0.91
    )
    engine.add_identified_component(r4)
    print(f"    ✓ R4: {r4.part_number}")

    # User button
    print("\n  Found user pushbutton (SMD tactile switch):")
    sw2 = IdentifiedComponent(
        reference_designator="SW2",
        part_number="SMD Tactile Button",
        description="User Input Pushbutton",
        family=ComponentFamily.SWITCH,
        package="SMD",
        pins=2,
        location_x=120,
        location_y=60,
        connected_nets=["PA4", "GND"],
        confidence=0.91
    )
    engine.add_identified_component(sw2)
    print(f"    ✓ SW2: {sw2.part_number}")


def detect_damage(engine):
    """Analyze the board for damage"""
    print_section("STEP 6: DAMAGE ASSESSMENT")

    print("Performing visual damage inspection...")

    damage_found = []

    # Check for obvious damage
    print("\n  Checking for thermal damage...")
    if "U1" in engine.components:
        # Simulate checking if component shows damage
        print("    U1: No visible burn marks - OK")

    print("\n  Checking for corrosion...")
    print("    Edge connector: Some green corrosion detected")
    engine.detect_damage("J1", "corrosion", "medium")
    damage_found.append("Corrosion on J1 connector")

    print("\n  Checking for mechanical damage...")
    print("    All components firmly soldered - OK")
    print("    No cracked solder joints - OK")

    if damage_found:
        print(f"\n  ⚠️  Found {len(damage_found)} issues:")
        for issue in damage_found:
            print(f"    - {issue}")
    else:
        print("\n  ✓ No major damage detected")


def extract_and_validate_bom(engine):
    """Extract BOM and validate parts"""
    print_section("STEP 7: BILL OF MATERIALS")

    print("Generating Bill of Materials from identified components...")

    bom = engine.extract_bom()

    print(f"\nExtracted {len(bom)} unique parts:\n")
    print(f"{'Reference':<12} {'Part Number':<25} {'Qty':<5} {'Description':<35}")
    print("-" * 80)

    total_estimated_cost = 0
    for entry in bom:
        print(f"{', '.join(entry['references']):<12} {entry['part_number']:<25} "
              f"{entry['quantity']:<5} {entry['description']:<35}")
        if entry.get('estimated_cost_per_unit'):
            entry_cost = entry['estimated_cost_per_unit'] * entry['quantity']
            total_estimated_cost += entry_cost

    print("-" * 80)
    print(f"Total components: {sum(e['quantity'] for e in bom)}")
    print(f"Estimated total cost: ${total_estimated_cost:.2f}")

    return bom


def find_replacements(engine):
    """Find replacement parts for known components"""
    print_section("STEP 8: FIND REPLACEMENT PARTS")

    print("Analyzing replacement options...\n")

    # Suggest replacements for main components
    key_components = ["U1", "U2", "U3"]

    for component_ref in key_components:
        if component_ref in engine.components:
            comp = engine.components[component_ref]
            print(f"\n{component_ref}: {comp.part_number}")
            print(f"  Current: {comp.description}")

            replacements = engine.suggest_replacements(component_ref)
            if replacements:
                print(f"  Alternative parts:")
                for alt in replacements[:3]:
                    print(f"    - {alt}")
            else:
                print(f"  No direct alternatives in database")


def generate_schematic_netlist(engine):
    """Generate a SPICE netlist from the identified circuit"""
    print_section("STEP 9: GENERATE SCHEMATIC NETLIST")

    print("Creating SPICE netlist from identified connections...\n")

    netlist = engine.generate_schematic_netlist()
    print(netlist[:500] + "...\n(truncated)")

    # Save netlist
    netlist_file = "reverse_engineered_circuit.cir"
    with open(netlist_file, "w") as f:
        f.write(netlist)
    print(f"✓ Full netlist saved to: {netlist_file}")


def generate_reports(engine):
    """Generate detailed analysis reports"""
    print_section("STEP 10: GENERATE ANALYSIS REPORTS")

    print("Creating comprehensive reverse engineering report...")

    # Generate analysis report
    report = engine.generate_analysis_report()

    print(f"\nAnalysis Summary:")
    print(f"  Project ID: {report['project_id']}")
    print(f"  Components identified: {report['statistics']['total_components']}")
    print(f"  Nets identified: {report['statistics']['total_nets']}")
    print(f"  Average confidence: {report['statistics']['avg_confidence']:.1%}")

    # Export as JSON
    json_file = "reverse_engineering_report.json"
    engine.export_to_json(json_file)
    print(f"\n  ✓ Detailed report exported: {json_file}")

    # Export as KiCAD schematic
    kicad_file = "reverse_engineered_schematic.kicad_sch"
    engine.export_to_kicad_schematic(kicad_file)
    print(f"  ✓ KiCAD schematic stub exported: {kicad_file}")

    print(f"\nComponent breakdown:")
    for family, count in report['statistics']['components_by_family'].items():
        print(f"  {family.title()}: {count}")


def repair_recommendations(engine):
    """Provide repair recommendations based on damage"""
    print_section("STEP 11: REPAIR RECOMMENDATIONS")

    print("Analyzing damage and recommending repairs...\n")

    damage_report = {
        "J1": {
            "issue": "Corrosion on connector pins",
            "severity": "medium",
            "recommendations": [
                "Clean connector contacts with white vinegar",
                "Allow to dry completely (2-4 hours)",
                "Apply thin layer of dielectric grease",
                "Test connection with continuity meter",
                "If non-functional, consider connector replacement"
            ]
        }
    }

    for component, damage_info in damage_report.items():
        print(f"{component}: {damage_info['issue']}")
        print(f"  Severity: {damage_info['severity'].upper()}")
        print(f"  Recommendations:")
        for i, rec in enumerate(damage_info['recommendations'], 1):
            print(f"    {i}. {rec}")
        print()


def main():
    """Run the reverse engineering example"""

    print("\n" + "=" * 80)
    print("  REVERSE ENGINEERING LEGACY CIRCUIT BOARD")
    print("  Complete Reconstruction Workflow")
    print("=" * 80)

    # Initialize engine
    engine = ReverseEngineeringEngine()
    print("\n✓ Reverse Engineering Engine initialized")

    try:
        # Step 1: Identify main processor
        identify_main_processor(engine)

        # Step 2: Identify power supply
        identify_power_supply(engine)

        # Step 3: Identify communication
        identify_communication_interface(engine)

        # Step 4: Identify clock and reset
        identify_clock_and_reset(engine)

        # Step 5: Identify user interface
        identify_user_interface(engine)

        # Step 6: Detect damage
        detect_damage(engine)

        # Step 7: Extract BOM
        bom = extract_and_validate_bom(engine)

        # Step 8: Find replacements
        find_replacements(engine)

        # Step 9: Generate netlist
        generate_schematic_netlist(engine)

        # Step 10: Generate reports
        generate_reports(engine)

        # Step 11: Repair recommendations
        repair_recommendations(engine)

        print("\n" + "=" * 80)
        print("  ✓ REVERSE ENGINEERING ANALYSIS COMPLETE")
        print("=" * 80)
        print("\nGenerated Files:")
        print("  - reverse_engineered_circuit.cir (SPICE netlist)")
        print("  - reverse_engineering_report.json (detailed analysis)")
        print("  - reverse_engineered_schematic.kicad_sch (KiCAD schematic)")
        print("\nNext Steps:")
        print("  1. Review the generated schematic in KiCAD")
        print("  2. Order replacement parts from the BOM")
        print("  3. Perform repairs as recommended")
        print("  4. Test the restored board")

    except Exception as e:
        print(f"\n❌ Error during reverse engineering: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
QuLabInfinite R&D Engineering Lab - Reverse Engineering Module
===============================================================

Rapid component identification, schematic reconstruction, and design analysis
from physical hardware, images, or existing designs.

Features:
- Component identification from images (visual recognition)
- Schematic reconstruction from component analysis
- Bill of Materials (BOM) extraction and validation
- Design pattern recognition
- Equivalence component suggestions from 1M+ materials database
- Functionality inference
- Damage assessment and repair recommendations
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import json
from pathlib import Path
from datetime import datetime
import hashlib


class ComponentFamily(Enum):
    """Component classification families"""
    MICROCONTROLLER = "microcontroller"
    PROCESSOR = "processor"
    MEMORY = "memory"
    POWER = "power"
    ANALOG = "analog"
    DIGITAL = "digital"
    COMMUNICATION = "communication"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    DISCRETE = "discrete"
    CONNECTOR = "connector"
    PASSIVE = "passive"
    MECHANICAL = "mechanical"
    UNKNOWN = "unknown"


class SignalType(Enum):
    """Signal types on PCB"""
    POWER = "power"
    GROUND = "ground"
    SIGNAL = "signal"
    CLOCK = "clock"
    RESET = "reset"
    INTERRUPT = "interrupt"
    DATA_BUS = "data_bus"
    ADDRESS_BUS = "address_bus"
    CONTROL = "control"
    ANALOG = "analog"


@dataclass
class ComponentSignature:
    """Identifies a component by visual or electrical signature"""
    # Visual characteristics
    package_type: str  # BGA, QFP, DIP, SOT, etc.
    num_pins: int
    marking_text: str  # Silkscreen text
    color: str  # Component color

    # Electrical characteristics
    estimated_function: str
    common_variants: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "package_type": self.package_type,
            "num_pins": self.num_pins,
            "marking_text": self.marking_text,
            "color": self.color,
            "estimated_function": self.estimated_function,
            "common_variants": self.common_variants
        }


@dataclass
class IdentifiedComponent:
    """A component identified during reverse engineering"""
    reference_designator: str  # U1, R5, C3, etc.
    part_number: str
    description: str
    family: ComponentFamily
    package: str
    pins: int

    # Electrical specs
    voltage_rating: Optional[float] = None
    current_rating: Optional[float] = None
    frequency_range: Optional[Tuple[float, float]] = None
    power_dissipation: Optional[float] = None

    # Location and connections
    location_x: Optional[float] = None
    location_y: Optional[float] = None
    connected_nets: List[str] = field(default_factory=list)
    connected_to: List[str] = field(default_factory=list)  # Other component refs

    # Sourcing and cost
    estimated_cost: Optional[float] = None
    supplier: Optional[str] = None
    available_alternates: List[str] = field(default_factory=list)

    # Damage assessment
    suspected_damage: Optional[str] = None
    burn_marks: bool = False
    corrosion: bool = False

    confidence: float = 0.9  # Confidence 0-1

    def to_dict(self) -> Dict:
        return {
            "reference": self.reference_designator,
            "part_number": self.part_number,
            "description": self.description,
            "family": self.family.value,
            "package": self.package,
            "pins": self.pins,
            "voltage_rating": self.voltage_rating,
            "current_rating": self.current_rating,
            "frequency_range": self.frequency_range,
            "power_dissipation": self.power_dissipation,
            "location": (self.location_x, self.location_y),
            "connected_nets": self.connected_nets,
            "connected_to": self.connected_to,
            "estimated_cost": self.estimated_cost,
            "supplier": self.supplier,
            "alternates": self.available_alternates,
            "damage": {
                "suspected": self.suspected_damage,
                "burn_marks": self.burn_marks,
                "corrosion": self.corrosion
            },
            "confidence": self.confidence
        }


@dataclass
class CircuitNet:
    """Electrical net/node identified in schematic"""
    name: str
    signal_type: SignalType
    voltage_level: Optional[float] = None
    impedance: Optional[float] = None
    connected_pins: List[Tuple[str, int]] = field(default_factory=list)  # (ref_des, pin_num)
    trace_width: Optional[float] = None
    layer: str = "signal"

    def add_pin(self, reference: str, pin_number: int):
        self.connected_pins.append((reference, pin_number))

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.signal_type.value,
            "voltage_level": self.voltage_level,
            "impedance": self.impedance,
            "connected_pins": self.connected_pins,
            "trace_width": self.trace_width,
            "layer": self.layer
        }


@dataclass
class FunctionalBlock:
    """Functional subsystem identified in design"""
    name: str
    function: str
    input_signals: List[str]
    output_signals: List[str]
    components: List[str]  # Reference designators
    supply_voltage: Optional[float] = None
    operating_frequency: Optional[float] = None
    estimated_power: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "function": self.function,
            "inputs": self.input_signals,
            "outputs": self.output_signals,
            "components": self.components,
            "supply_voltage": self.supply_voltage,
            "operating_frequency": self.operating_frequency,
            "estimated_power": self.estimated_power
        }


class ReverseEngineeringEngine:
    """Main reverse engineering analysis engine"""

    # Component identification database
    COMPONENT_DATABASE = {
        # Microcontrollers
        "STM32F4": {
            "family": ComponentFamily.MICROCONTROLLER,
            "typical_package": "QFP",
            "typical_pins": 176,
            "voltage": 3.3,
            "common_uses": ["main processor", "sensor interface", "motor control"]
        },
        "ATmega328P": {
            "family": ComponentFamily.MICROCONTROLLER,
            "typical_package": "TQFP",
            "typical_pins": 32,
            "voltage": 5.0,
            "common_uses": ["Arduino core", "simple control"]
        },
        "ESP32": {
            "family": ComponentFamily.PROCESSOR,
            "typical_package": "BGA",
            "typical_pins": 48,
            "voltage": 3.3,
            "common_uses": ["WiFi", "Bluetooth", "IoT gateway"]
        },

        # Voltage regulators
        "LM7805": {
            "family": ComponentFamily.POWER,
            "typical_package": "TO-220",
            "typical_pins": 3,
            "voltage": 5.0,
            "common_uses": ["5V supply", "logic power"]
        },
        "MP2307": {
            "family": ComponentFamily.POWER,
            "typical_package": "DFN",
            "typical_pins": 16,
            "voltage": 3.3,
            "common_uses": ["buck converter", "efficient step-down"]
        },

        # Memory
        "W25Q256": {
            "family": ComponentFamily.MEMORY,
            "typical_package": "SOP",
            "typical_pins": 8,
            "capacity": "256Mbit",
            "common_uses": ["flash storage", "firmware"]
        },
        "24LC512": {
            "family": ComponentFamily.MEMORY,
            "typical_package": "DIP",
            "typical_pins": 8,
            "capacity": "512Kbit",
            "common_uses": ["EEPROM", "configuration"]
        },

        # Communication
        "MAX3232": {
            "family": ComponentFamily.COMMUNICATION,
            "typical_package": "SOIC",
            "typical_pins": 16,
            "common_uses": ["RS-232 transceiver", "serial port"]
        },
        "MCP2515": {
            "family": ComponentFamily.COMMUNICATION,
            "typical_package": "DIP",
            "typical_pins": 18,
            "common_uses": ["CAN bus controller", "automotive"]
        },
    }

    def __init__(self):
        self.components: Dict[str, IdentifiedComponent] = {}
        self.nets: Dict[str, CircuitNet] = {}
        self.functional_blocks: List[FunctionalBlock] = []
        self.bom: List[Dict] = []
        self.analysis_history: List[Dict] = []
        self.project_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]

    def identify_component_from_signature(self, signature: ComponentSignature) -> Optional[str]:
        """Identify component from visual/electrical signature"""
        candidates = []

        # Match against database
        for part_code, specs in self.COMPONENT_DATABASE.items():
            # Simple matching based on package type and pin count
            if specs.get("typical_pins") == signature.num_pins:
                # Check if marking text matches
                if signature.marking_text.upper() in part_code.upper():
                    candidates.append((part_code, 0.95))  # High confidence
                else:
                    candidates.append((part_code, 0.7))   # Medium confidence

        # Return best match
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        return None

    def add_identified_component(self, component: IdentifiedComponent) -> str:
        """Add an identified component to the reverse engineering project"""
        ref = component.reference_designator
        self.components[ref] = component
        self.analysis_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": "component_added",
            "reference": ref,
            "part": component.part_number
        })
        return ref

    def add_connection(self, from_ref: str, to_ref: str, net_name: str):
        """Record a connection between two components"""
        if from_ref in self.components:
            self.components[from_ref].connected_to.append(to_ref)
        if to_ref in self.components:
            self.components[to_ref].connected_to.append(from_ref)

        # Create or update net
        if net_name not in self.nets:
            self.nets[net_name] = CircuitNet(
                name=net_name,
                signal_type=SignalType.SIGNAL
            )

        self.nets[net_name].add_pin(from_ref, 0)
        self.nets[net_name].add_pin(to_ref, 0)

    def identify_functional_blocks(self) -> List[FunctionalBlock]:
        """Identify functional subsystems from component layout"""
        blocks = []

        # Power supply block
        power_components = [ref for ref, comp in self.components.items()
                           if comp.family == ComponentFamily.POWER]
        if power_components:
            blocks.append(FunctionalBlock(
                name="Power Supply",
                function="Voltage regulation and power distribution",
                input_signals=["VIN"],
                output_signals=["3.3V", "5V"],
                components=power_components
            ))

        # Microcontroller block
        mcu_components = [ref for ref, comp in self.components.items()
                         if comp.family == ComponentFamily.MICROCONTROLLER]
        if mcu_components:
            blocks.append(FunctionalBlock(
                name="Microcontroller Core",
                function="Main processing and control",
                input_signals=["POWER", "RESET", "CLK"],
                output_signals=["GPIO", "UART", "SPI"],
                components=mcu_components
            ))

        # Communication block
        comm_components = [ref for ref, comp in self.components.items()
                          if comp.family == ComponentFamily.COMMUNICATION]
        if comm_components:
            blocks.append(FunctionalBlock(
                name="Communication Interface",
                function="External communication protocols",
                input_signals=["SERIAL_IN", "CAN_IN"],
                output_signals=["SERIAL_OUT", "CAN_OUT"],
                components=comm_components
            ))

        # Sensor block
        sensor_components = [ref for ref, comp in self.components.items()
                            if comp.family == ComponentFamily.SENSOR]
        if sensor_components:
            blocks.append(FunctionalBlock(
                name="Sensor Interface",
                function="Environmental and system monitoring",
                input_signals=["ANALOG_SIGNALS"],
                output_signals=["ADC_DATA"],
                components=sensor_components
            ))

        self.functional_blocks = blocks
        return blocks

    def detect_damage(self, reference: str, damage_type: str, severity: str = "medium"):
        """Record suspected damage on a component"""
        if reference in self.components:
            comp = self.components[reference]
            comp.suspected_damage = damage_type

            if damage_type == "burn":
                comp.burn_marks = True
                comp.confidence = max(0, comp.confidence - 0.2)
            elif damage_type == "corrosion":
                comp.corrosion = True
                comp.confidence = max(0, comp.confidence - 0.15)

            self.analysis_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": "damage_detected",
                "reference": reference,
                "damage_type": damage_type,
                "severity": severity
            })

    def suggest_replacements(self, reference: str, use_materials_db: bool = False) -> List[str]:
        """Suggest equivalent or better replacement components"""
        if reference not in self.components:
            return []

        component = self.components[reference]
        replacements = component.available_alternates.copy()

        # Add from knowledge base
        if component.part_number in self.COMPONENT_DATABASE:
            specs = self.COMPONENT_DATABASE[component.part_number]
            # Suggest other parts with similar specs
            for part, part_specs in self.COMPONENT_DATABASE.items():
                if (part_specs.get("family") == specs.get("family") and
                    part != component.part_number):
                    replacements.append(part)

        return list(set(replacements))

    def extract_bom(self) -> List[Dict]:
        """Extract Bill of Materials from identified components"""
        bom = []

        # Group by part number for quantities
        part_counts = {}
        for ref, component in self.components.items():
            key = component.part_number
            if key not in part_counts:
                part_counts[key] = {
                    "part_number": key,
                    "description": component.description,
                    "references": [],
                    "quantity": 0,
                    "package": component.package,
                    "family": component.family.value
                }
            part_counts[key]["references"].append(ref)
            part_counts[key]["quantity"] += 1

        # Build BOM list
        for part_num, info in part_counts.items():
            bom_entry = {
                "part_number": part_num,
                "description": info["description"],
                "quantity": info["quantity"],
                "references": info["references"],
                "package": info["package"],
                "family": info["family"],
                "estimated_cost_per_unit": None,
                "estimated_total_cost": None,
                "supplier": "Unknown"
            }

            # Try to find cost if component is in database
            if part_num in self.COMPONENT_DATABASE:
                bom_entry["supplier"] = "Standard Component"

            bom.append(bom_entry)

        self.bom = bom
        return bom

    def generate_schematic_netlist(self) -> str:
        """Generate SPICE-style netlist from identified connections"""
        netlist = f"* Auto-generated netlist from reverse engineering\n"
        netlist += f"* Project ID: {self.project_id}\n"
        netlist += f"* Generated: {datetime.now().isoformat()}\n\n"

        # Power supply
        netlist += "* Power Supply\n"
        netlist += ".param VCC=5V\n"
        netlist += ".param GND=0V\n\n"

        # Components
        netlist += "* Components\n"
        for ref, comp in self.components.items():
            if comp.family == ComponentFamily.POWER:
                netlist += f"{ref} VIN VOUT GND VREG\n"
            elif comp.family == ComponentFamily.MICROCONTROLLER:
                pins_str = " ".join([f"P{i}" for i in range(comp.pins)])
                netlist += f"{ref} {pins_str} {comp.part_number}\n"

        netlist += "\n"

        # Nets
        netlist += "* Nets\n"
        for net_name, net in self.nets.items():
            netlist += f"* {net_name}: {net.signal_type.value}\n"

        return netlist

    def generate_analysis_report(self) -> Dict:
        """Generate comprehensive reverse engineering analysis report"""
        report = {
            "project_id": self.project_id,
            "timestamp": datetime.now().isoformat(),
            "components": {
                ref: comp.to_dict()
                for ref, comp in self.components.items()
            },
            "nets": {
                net_name: net.to_dict()
                for net_name, net in self.nets.items()
            },
            "functional_blocks": [
                block.to_dict()
                for block in self.functional_blocks
            ],
            "bom": self.bom,
            "statistics": {
                "total_components": len(self.components),
                "total_nets": len(self.nets),
                "avg_confidence": (
                    sum(c.confidence for c in self.components.values()) / len(self.components)
                    if self.components else 0
                ),
                "components_by_family": self._count_by_family(),
                "estimated_total_cost": self._estimate_total_cost()
            },
            "damage_assessment": {
                ref: {
                    "damage": comp.suspected_damage,
                    "burn_marks": comp.burn_marks,
                    "corrosion": comp.corrosion
                }
                for ref, comp in self.components.items()
                if comp.suspected_damage
            },
            "analysis_history": self.analysis_history
        }
        return report

    def _count_by_family(self) -> Dict[str, int]:
        """Count components by family"""
        counts = {}
        for comp in self.components.values():
            family = comp.family.value
            counts[family] = counts.get(family, 0) + 1
        return counts

    def _estimate_total_cost(self) -> float:
        """Estimate total component cost"""
        total = 0.0
        for entry in self.bom:
            if entry["estimated_cost_per_unit"]:
                total += entry["estimated_cost_per_unit"] * entry["quantity"]
        return total

    def export_to_json(self, filepath: str):
        """Export analysis report to JSON file"""
        report = self.generate_analysis_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

    def export_to_kicad_schematic(self, filepath: str):
        """Export as KiCAD schematic stub"""
        output = "(kicad_sch (version 20230121) (generator eeschema)\n"
        output += f"  (uuid \"{self.project_id}\")\n"
        output += f"  (paper \"A4\")\n\n"
        output += "  (title_block\n"
        output += f"    (title \"Reverse Engineered: {len(self.components)} Components\")\n"
        output += f"    (date \"{datetime.now().date()}\")\n"
        output += "  )\n\n"

        # Add sheet
        output += "  (sheet (at 0 0) (size 297 210)\n"
        output += "    (uuid \"auto-generated\")\n"

        # Add symbols (components)
        y_pos = 10
        for ref, comp in self.components.items():
            output += f"    (symbol (lib_id \"Device:R\") (at {10} {y_pos} 0)\n"
            output += f"      (property \"Reference\" \"{ref}\" (at 0 0 0))\n"
            output += f"      (property \"Value\" \"{comp.part_number}\" (at 0 0 0))\n"
            output += "    )\n"
            y_pos += 10

        output += "  )\n"
        output += ")\n"

        with open(filepath, 'w') as f:
            f.write(output)

    def import_from_json(self, filepath: str):
        """Import analysis from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Restore components
        for ref, comp_data in data.get("components", {}).items():
            comp = IdentifiedComponent(
                reference_designator=comp_data["reference"],
                part_number=comp_data["part_number"],
                description=comp_data["description"],
                family=ComponentFamily[comp_data["family"].upper()],
                package=comp_data["package"],
                pins=comp_data["pins"],
                confidence=comp_data["confidence"]
            )
            self.add_identified_component(comp)


def analyze_pcb_image_simple(image_path: str) -> Dict:
    """
    Simplified PCB image analysis (would need OpenCV/ML in production)
    Returns detected components and their approximate locations
    """
    # This is a placeholder - in production, would use:
    # - OpenCV for edge detection and component isolation
    # - Tesseract for text/marking recognition
    # - ML model for component classification

    return {
        "image_path": image_path,
        "detected_regions": [],
        "text_detected": [],
        "status": "ready for manual annotation"
    }


# Example usage
if __name__ == "__main__":
    # Create engine
    engine = ReverseEngineeringEngine()

    # Add some identified components
    mcu = IdentifiedComponent(
        reference_designator="U1",
        part_number="STM32F407VG",
        description="ARM Cortex-M4 Microcontroller",
        family=ComponentFamily.MICROCONTROLLER,
        package="LQFP",
        pins=100,
        voltage_rating=3.3,
        location_x=50,
        location_y=50
    )
    engine.add_identified_component(mcu)

    # Add power supply
    pwr = IdentifiedComponent(
        reference_designator="U2",
        part_number="MP2307",
        description="3.3V Buck Converter",
        family=ComponentFamily.POWER,
        package="DFN",
        pins=16,
        voltage_rating=5.0,
        location_x=20,
        location_y=80
    )
    engine.add_identified_component(pwr)

    # Add capacitors
    for i in range(1, 4):
        cap = IdentifiedComponent(
            reference_designator=f"C{i}",
            part_number="TDK X7R 100uF",
            description="Ceramic Capacitor 100µF",
            family=ComponentFamily.PASSIVE,
            package="1206",
            pins=2,
            location_x=30 + i*10,
            location_y=60
        )
        engine.add_identified_component(cap)

    # Identify functional blocks
    blocks = engine.identify_functional_blocks()

    # Extract BOM
    bom = engine.extract_bom()

    # Generate report
    report = engine.generate_analysis_report()

    print("=== Reverse Engineering Analysis ===\n")
    print(f"Project ID: {engine.project_id}")
    print(f"Components identified: {len(engine.components)}")
    print(f"Functional blocks: {len(blocks)}\n")

    print("=== Bill of Materials ===")
    for entry in bom:
        print(f"{entry['part_number']}: {entry['quantity']}x {entry['description']}")

    print(f"\nTotal estimated cost: ${report['statistics']['estimated_total_cost']:.2f}")

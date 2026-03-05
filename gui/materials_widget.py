"""
Materials Lab widget for browsing and selecting materials.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QDoubleSpinBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPlainTextEdit,
    QSplitter,
)

from materials_lab.materials_database import MaterialProperties, MaterialsDatabase


class MaterialsWidget(QWidget):
    """Search and inspect materials from the MaterialsDatabase."""

    material_selected = Signal(str)
    status_message = Signal(str, int)

    def __init__(self):
        super().__init__()

        self.db = MaterialsDatabase()
        self.current_results: List[MaterialProperties] = []
        self._build_ui()
        self._load_categories()
        self.refresh_results()

    def _build_ui(self):
        root = QVBoxLayout(self)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("name, notes, subcategory")
        filters.addWidget(self.search_input, stretch=2)

        filters.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.addItem("All")
        filters.addWidget(self.category_combo, stretch=1)

        filters.addWidget(QLabel("Density Min:"))
        self.min_density = QDoubleSpinBox()
        self.min_density.setRange(0, 1_000_000)
        self.min_density.setDecimals(2)
        self.min_density.setSingleStep(10)
        filters.addWidget(self.min_density)

        filters.addWidget(QLabel("Density Max:"))
        self.max_density = QDoubleSpinBox()
        self.max_density.setRange(0, 1_000_000)
        self.max_density.setDecimals(2)
        self.max_density.setValue(1_000_000)
        self.max_density.setSingleStep(100)
        filters.addWidget(self.max_density)

        filters.addWidget(QLabel("Min Strength (MPa):"))
        self.min_strength = QDoubleSpinBox()
        self.min_strength.setRange(0, 1_000_000)
        self.min_strength.setDecimals(2)
        self.min_strength.setSingleStep(10)
        filters.addWidget(self.min_strength)

        self.search_button = QPushButton("Search")
        self.reset_button = QPushButton("Reset")
        filters.addWidget(self.search_button)
        filters.addWidget(self.reset_button)
        root.addLayout(filters)

        splitter = QSplitter()

        self.results_table = QTableWidget(0, 7)
        self.results_table.setHorizontalHeaderLabels(
            [
                "Name",
                "Category",
                "Subcategory",
                "Density (kg/m3)",
                "Strength (MPa)",
                "Thermal k",
                "Cost ($/kg)",
            ]
        )
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SingleSelection)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        splitter.addWidget(self.results_table)

        detail_panel = QWidget()
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.addWidget(QLabel("Material Details"))
        self.detail_text = QPlainTextEdit()
        self.detail_text.setReadOnly(True)
        detail_layout.addWidget(self.detail_text, stretch=1)

        self.select_button = QPushButton("Use In Stitch Workflow")
        detail_layout.addWidget(self.select_button)
        splitter.addWidget(detail_panel)

        splitter.setSizes([700, 500])
        root.addWidget(splitter, stretch=1)

        self.search_button.clicked.connect(self.refresh_results)
        self.reset_button.clicked.connect(self.reset_filters)
        self.search_input.returnPressed.connect(self.refresh_results)
        self.results_table.itemSelectionChanged.connect(self.show_selected_details)
        self.select_button.clicked.connect(self.emit_selected_material)

    def _load_categories(self):
        self.category_combo.blockSignals(True)
        self.category_combo.clear()
        self.category_combo.addItem("All")
        for category in self.db.list_categories():
            self.category_combo.addItem(category)
        self.category_combo.blockSignals(False)

    def refresh_results(self):
        category_text = self.category_combo.currentText().strip()
        category = category_text if category_text and category_text != "All" else None
        query = self.search_input.text().strip() or None

        results = self.db.search_materials(
            category=category,
            min_density=self.min_density.value(),
            max_density=self.max_density.value(),
            min_strength=self.min_strength.value(),
            text=query,
        )
        self.current_results = sorted(results, key=lambda m: m.name.lower())
        self._populate_results_table(self.current_results)
        self.status_message.emit(f"Materials search: {len(self.current_results)} result(s)", 5000)

    def reset_filters(self):
        self.search_input.clear()
        self.category_combo.setCurrentIndex(0)
        self.min_density.setValue(0)
        self.max_density.setValue(1_000_000)
        self.min_strength.setValue(0)
        self.refresh_results()

    def _populate_results_table(self, materials: List[MaterialProperties]):
        self.results_table.setRowCount(len(materials))

        for row, material in enumerate(materials):
            values = [
                material.name,
                material.category,
                material.subcategory,
                f"{material.density:.2f}",
                f"{material.tensile_strength:.2f}",
                f"{material.thermal_conductivity:.4g}",
                f"{material.cost_per_kg:.2f}",
            ]
            for col, value in enumerate(values):
                self.results_table.setItem(row, col, QTableWidgetItem(value))

        if materials:
            self.results_table.selectRow(0)
            self.show_selected_details()
        else:
            self.detail_text.setPlainText("No materials match current filters.")

    def _selected_material(self) -> Optional[MaterialProperties]:
        row = self.results_table.currentRow()
        if row < 0 or row >= len(self.current_results):
            return None
        return self.current_results[row]

    def show_selected_details(self):
        material = self._selected_material()
        if material is None:
            self.detail_text.setPlainText("Select a material to inspect properties.")
            return

        safety_data = self.db.get_safety_data(material.name)
        lines = [
            f"Name: {material.name}",
            f"Category: {material.category} / {material.subcategory}",
            "",
            "Mechanical:",
            f"  Density: {material.density:.3f} kg/m3",
            f"  Young's modulus: {material.youngs_modulus:.3f} GPa",
            f"  Tensile strength: {material.tensile_strength:.3f} MPa",
            f"  Yield strength: {material.yield_strength:.3f} MPa",
            "",
            "Thermal:",
            f"  Conductivity: {material.thermal_conductivity:.6g} W/(m*K)",
            f"  Melting point: {material.melting_point:.3f} K",
            "",
            "Other:",
            f"  Corrosion resistance: {material.corrosion_resistance}",
            f"  Cost: ${material.cost_per_kg:.2f}/kg",
            f"  Availability: {material.availability}",
            "",
        ]
        if material.notes:
            lines.extend(["Notes:", material.notes, ""])
        if safety_data:
            lines.extend(["Safety:", str(safety_data)])
        else:
            lines.append("Safety: No specific safety record found.")

        self.detail_text.setPlainText("\n".join(lines))

    def emit_selected_material(self):
        material = self._selected_material()
        if material is None:
            self.status_message.emit("Select a material first.", 4000)
            return
        self.material_selected.emit(material.name)
        self.status_message.emit(f"Selected material: {material.name}", 5000)


"""
Stitch Workflow widget backed by WorkflowEngine.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSplitter,
)

from workflow_engine import WorkflowEngine


class WorkflowWidget(QWidget):
    """Interactive workflow builder and executor for Stitch flows."""

    status_message = Signal(str, int)

    def __init__(self):
        super().__init__()

        self.engine = WorkflowEngine()
        self.current_workflow_id: Optional[str] = None
        self.selected_material: Optional[str] = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        workflow_row = QHBoxLayout()
        workflow_row.addWidget(QLabel("Workflow Name:"))
        self.workflow_name_input = QLineEdit("Materials To Physics")
        workflow_row.addWidget(self.workflow_name_input, stretch=2)
        self.create_workflow_button = QPushButton("Create Workflow")
        workflow_row.addWidget(self.create_workflow_button)
        root.addLayout(workflow_row)

        prompt_row = QVBoxLayout()
        prompt_row.addWidget(QLabel("Natural Language Experiment Request"))
        self.nl_prompt_input = QPlainTextEdit()
        self.nl_prompt_input.setPlaceholderText(
            "Example: Run a high-temperature corrosion experiment on titanium with beakers and burners, "
            "then compare tensile behavior."
        )
        self.nl_prompt_input.setFixedHeight(90)
        prompt_row.addWidget(self.nl_prompt_input)

        prompt_actions = QHBoxLayout()
        self.generate_from_prompt_button = QPushButton("Draft From Prompt")
        self.generate_and_execute_button = QPushButton("Draft + Execute")
        prompt_actions.addWidget(self.generate_from_prompt_button)
        prompt_actions.addWidget(self.generate_and_execute_button)
        self.mode_label = QLabel("Mode: manual")
        prompt_actions.addWidget(self.mode_label)
        prompt_row.addLayout(prompt_actions)
        root.addLayout(prompt_row)

        material_row = QHBoxLayout()
        self.material_context_label = QLabel("Material Context: none")
        material_row.addWidget(self.material_context_label, stretch=1)
        self.add_material_node_button = QPushButton("Add Material Node")
        material_row.addWidget(self.add_material_node_button)
        root.addLayout(material_row)

        node_row = QHBoxLayout()
        node_row.addWidget(QLabel("Lab:"))
        self.lab_input = QLineEdit("materials_selector")
        node_row.addWidget(self.lab_input, stretch=1)
        node_row.addWidget(QLabel("Parameters (JSON):"))
        self.params_input = QLineEdit("{}")
        node_row.addWidget(self.params_input, stretch=2)
        self.add_node_button = QPushButton("Add Node")
        node_row.addWidget(self.add_node_button)
        root.addLayout(node_row)

        edge_row = QHBoxLayout()
        edge_row.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        edge_row.addWidget(self.source_combo, stretch=1)
        edge_row.addWidget(QLabel("Target:"))
        self.target_combo = QComboBox()
        edge_row.addWidget(self.target_combo, stretch=1)
        self.connect_button = QPushButton("Connect")
        edge_row.addWidget(self.connect_button)
        root.addLayout(edge_row)

        action_row = QHBoxLayout()
        self.validate_button = QPushButton("Validate")
        self.execute_button = QPushButton("Execute")
        self.export_button = QPushButton("Refresh JSON")
        action_row.addWidget(self.validate_button)
        action_row.addWidget(self.execute_button)
        action_row.addWidget(self.export_button)
        root.addLayout(action_row)

        splitter = QSplitter()
        self.nodes_table = QTableWidget(0, 5)
        self.nodes_table.setHorizontalHeaderLabels(
            ["Node ID", "Lab", "Inputs", "Outputs", "Status"]
        )
        self.nodes_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.nodes_table.setEditTriggers(QTableWidget.NoEditTriggers)
        splitter.addWidget(self.nodes_table)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("Activity"))
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.log_text, stretch=1)
        right_layout.addWidget(QLabel("Workflow JSON"))
        self.json_text = QPlainTextEdit()
        self.json_text.setReadOnly(True)
        right_layout.addWidget(self.json_text, stretch=1)
        splitter.addWidget(right_panel)

        splitter.setSizes([650, 550])
        root.addWidget(splitter, stretch=1)

        self.create_workflow_button.clicked.connect(self.create_workflow)
        self.add_material_node_button.clicked.connect(self.add_material_node)
        self.add_node_button.clicked.connect(self.add_node)
        self.connect_button.clicked.connect(self.connect_nodes)
        self.validate_button.clicked.connect(self.validate_workflow)
        self.execute_button.clicked.connect(self.execute_workflow)
        self.export_button.clicked.connect(self.refresh_json_view)
        self.generate_from_prompt_button.clicked.connect(self.generate_workflow_from_prompt)
        self.generate_and_execute_button.clicked.connect(self.generate_and_execute_from_prompt)

    def set_material_context(self, material_name: str):
        self.selected_material = material_name
        self.material_context_label.setText(f"Material Context: {material_name}")
        self.params_input.setText(json.dumps({"material": material_name}))
        self.lab_input.setText("materials_selector")
        self.status_message.emit(f"Workflow context updated with material: {material_name}", 5000)

    def create_workflow(self):
        name = self.workflow_name_input.text().strip() or "Untitled Workflow"
        self.current_workflow_id = self.engine.create_workflow(name)
        self.nodes_table.setRowCount(0)
        self._refresh_node_selectors([])
        self._append_output(f"Created workflow '{name}' ({self.current_workflow_id}).")
        self.status_message.emit(f"Created workflow: {name}", 5000)
        self.refresh_json_view()
        self.mode_label.setText("Mode: manual")

    def _require_workflow(self) -> bool:
        if self.current_workflow_id:
            return True
        self.status_message.emit("Create a workflow first.", 5000)
        return False

    def _parse_params(self) -> Optional[Dict[str, Any]]:
        raw = self.params_input.text().strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.status_message.emit(f"Invalid JSON parameters: {exc}", 7000)
            return None
        if not isinstance(parsed, dict):
            self.status_message.emit("Parameters JSON must be an object.", 7000)
            return None
        return parsed

    def add_material_node(self):
        if not self._require_workflow():
            return
        if not self.selected_material:
            self.status_message.emit("No material selected from Materials Lab.", 6000)
            return

        try:
            node_id = self.engine.add_node(
                self.current_workflow_id,
                "materials_selector",
                {"material": self.selected_material},
            )
        except Exception as exc:
            self.status_message.emit(f"Failed to add material node: {exc}", 7000)
            return

        self._append_output(f"Added material node: {node_id}")
        self._refresh_nodes_view()

    def add_node(self):
        if not self._require_workflow():
            return

        lab_name = self.lab_input.text().strip()
        if not lab_name:
            self.status_message.emit("Lab name is required.", 5000)
            return

        params = self._parse_params()
        if params is None:
            return

        try:
            node_id = self.engine.add_node(self.current_workflow_id, lab_name, params)
        except Exception as exc:
            self.status_message.emit(f"Failed to add node: {exc}", 7000)
            return

        self._append_output(f"Added node: {node_id}")
        self.status_message.emit(f"Added node {node_id}", 5000)
        self._refresh_nodes_view()

    def connect_nodes(self):
        if not self._require_workflow():
            return

        source = self.source_combo.currentData()
        target = self.target_combo.currentData()
        if not source or not target:
            self.status_message.emit("Select source and target nodes.", 5000)
            return
        if source == target:
            self.status_message.emit("Cannot connect a node to itself.", 5000)
            return

        try:
            self.engine.connect_nodes(self.current_workflow_id, source, target)
        except Exception as exc:
            self.status_message.emit(f"Connect failed: {exc}", 7000)
            return

        self._append_output(f"Connected {source} -> {target}")
        self.status_message.emit(f"Connected {source} -> {target}", 5000)
        self._refresh_nodes_view()

    def validate_workflow(self):
        if not self._require_workflow():
            return

        validation = self.engine.validate_workflow(self.current_workflow_id)
        self._append_output("Validation:\n" + json.dumps(validation, indent=2))
        if validation.get("valid"):
            self.status_message.emit("Workflow validation passed.", 5000)
        else:
            self.status_message.emit("Workflow validation failed.", 6000)
        self._refresh_nodes_view()

    def execute_workflow(self):
        if not self._require_workflow():
            return

        try:
            result = asyncio.run(self.engine.execute_workflow(self.current_workflow_id))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    self.engine.execute_workflow(self.current_workflow_id)
                )
            finally:
                loop.close()
        except Exception as exc:
            self.status_message.emit(f"Execution failed: {exc}", 7000)
            return

        self._append_output("Execution:\n" + json.dumps(result, indent=2, default=str))
        status = result.get("status", "unknown")
        self.status_message.emit(f"Workflow execution status: {status}", 6000)
        self._refresh_nodes_view()

    def generate_workflow_from_prompt(self):
        prompt = self.nl_prompt_input.toPlainText().strip()
        if not prompt:
            self.status_message.emit("Enter an experiment request first.", 5000)
            return

        plan = self._build_plan_from_prompt(prompt)
        self._instantiate_plan(plan)

    def generate_and_execute_from_prompt(self):
        self.generate_workflow_from_prompt()
        if self.current_workflow_id:
            self.execute_workflow()

    def refresh_json_view(self):
        if not self._require_workflow():
            return
        payload = self.engine.export_workflow(self.current_workflow_id)
        self.json_text.setPlainText(json.dumps(payload, indent=2, default=str))

    def _refresh_nodes_view(self):
        if not self.current_workflow_id:
            self.nodes_table.setRowCount(0)
            self._refresh_node_selectors([])
            return

        workflow = self.engine.workflows[self.current_workflow_id]
        nodes: List[Any] = sorted(workflow["nodes"].values(), key=lambda n: n.id)
        self.nodes_table.setRowCount(len(nodes))
        for row, node in enumerate(nodes):
            values = [
                node.id,
                node.lab_name,
                ", ".join(node.inputs or []),
                ", ".join(node.outputs or []),
                node.status,
            ]
            for col, value in enumerate(values):
                self.nodes_table.setItem(row, col, QTableWidgetItem(value))

        self._refresh_node_selectors([node.id for node in nodes])
        self.refresh_json_view()

    def _refresh_node_selectors(self, node_ids: List[str]):
        self.source_combo.clear()
        self.target_combo.clear()
        for node_id in node_ids:
            self.source_combo.addItem(node_id, node_id)
            self.target_combo.addItem(node_id, node_id)

    def _append_output(self, text: str):
        existing = self.log_text.toPlainText().strip()
        if existing:
            self.log_text.setPlainText(f"{existing}\n\n{text}")
        else:
            self.log_text.setPlainText(text)

    def _build_plan_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Build a workflow plan from natural language.

        Strategy:
        - deterministic template when keywords match
        - freeform fallback when unmatched, so requests never feel blocked
        """
        p = prompt.lower()
        template_specs = [
            (
                "materials_pipeline",
                {"material", "alloy", "ceramic", "polymer", "metal", "corrosion"},
                [
                    "materials_selector",
                    "material_preparation",
                    "thermal_characterization",
                    "mechanical_validation",
                    "compliance_reporter",
                ],
            ),
            (
                "chemistry_pipeline",
                {"chemistry", "reaction", "synthesis", "compound", "beaker", "burner"},
                [
                    "reagent_planning",
                    "wet_lab_protocol_runner",
                    "reaction_monitoring",
                    "spectroscopy_analysis",
                    "results_summarizer",
                ],
            ),
            (
                "biomedical_pipeline",
                {"genomics", "dna", "rna", "cell", "oncology", "diagnostic", "pharmacokinetics"},
                [
                    "sample_preparation",
                    "assay_protocol_runner",
                    "omics_analysis",
                    "efficacy_or_risk_modeling",
                    "clinical_summary_writer",
                ],
            ),
            (
                "physics_pipeline",
                {"physics", "fluid", "pressure", "thermal", "stress", "simulation"},
                [
                    "experiment_design",
                    "apparatus_calibration",
                    "physics_simulation",
                    "result_validation",
                    "report_generation",
                ],
            ),
        ]

        selected = None
        selected_hits = 0
        for template_name, keywords, labs in template_specs:
            hits = sum(1 for kw in keywords if kw in p)
            if hits > selected_hits:
                selected = (template_name, labs)
                selected_hits = hits

        apparatus = self._extract_apparatus(prompt)
        material = self.selected_material

        # Require at least 2 keyword hits before forcing a deterministic template.
        if selected and selected_hits >= 2:
            template_name, labs = selected
            # Always include apparatus simulation so NL requests feel like real lab operation.
            if "lab_apparatus_simulator" not in labs:
                labs = [labs[0], "lab_apparatus_simulator", *labs[1:]]
            return {
                "mode": "deterministic-template",
                "template": template_name,
                "workflow_name": f"{template_name.replace('_', ' ').title()}",
                "labs": labs,
                "prompt": prompt,
                "apparatus": apparatus,
                "material": material,
            }

        # Open-ended fallback for arbitrary experiment requests.
        return {
            "mode": "open-ended",
            "template": "custom_experiment_fallback",
            "workflow_name": "Custom Experiment Request",
            "labs": [
                "experiment_intent_parser",
                "lab_apparatus_simulator",
                "custom_experiment_designer",
                "freeform_experiment_executor",
                "results_summarizer",
            ],
            "prompt": prompt,
            "apparatus": apparatus,
            "material": material,
        }

    def _extract_apparatus(self, prompt: str) -> List[str]:
        p = prompt.lower()
        known = [
            "beaker",
            "beakers",
            "burner",
            "burners",
            "flask",
            "flasks",
            "pipette",
            "pipettes",
            "centrifuge",
            "microscope",
            "spectrometer",
            "hot plate",
        ]
        found = [item for item in known if item in p]
        if found:
            return sorted(set(found))
        return ["beakers", "burners", "pipettes", "flasks"]

    def _instantiate_plan(self, plan: Dict[str, Any]):
        self.workflow_name_input.setText(plan["workflow_name"])
        self.create_workflow()

        if not self.current_workflow_id:
            self.status_message.emit("Unable to create workflow.", 7000)
            return

        self.mode_label.setText(f"Mode: {plan['mode']}")

        node_ids: List[str] = []
        for idx, lab_name in enumerate(plan["labs"]):
            params: Dict[str, Any] = {
                "natural_language_request": plan["prompt"],
                "step_index": idx + 1,
                "step_count": len(plan["labs"]),
            }
            if plan.get("material"):
                params["material"] = plan["material"]
            if lab_name == "lab_apparatus_simulator":
                params["apparatus"] = plan["apparatus"]
                params["simulation_mode"] = "natural_language_lab_protocol"
            if idx == len(plan["labs"]) - 1:
                params["expectation"] = "return clear experiment conclusions"

            node_id = self.engine.add_node(self.current_workflow_id, lab_name, params)
            node_ids.append(node_id)

        for source, target in zip(node_ids, node_ids[1:]):
            self.engine.connect_nodes(self.current_workflow_id, source, target)

        self._refresh_nodes_view()
        self._append_output(
            "NL Plan:\n"
            + json.dumps(
                {
                    "mode": plan["mode"],
                    "template": plan["template"],
                    "labs": plan["labs"],
                    "apparatus": plan["apparatus"],
                    "material": plan.get("material"),
                },
                indent=2,
            )
        )
        self.status_message.emit(
            f"Drafted {plan['mode']} workflow with {len(plan['labs'])} steps.",
            6000,
        )

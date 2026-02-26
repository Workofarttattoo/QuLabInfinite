"""
Control widget for the Physics Engine GUI.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QGroupBox,
    QFormLayout, QDoubleSpinBox
)

class PhysicsControls(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Simulation selection
        sim_group = QGroupBox("Physics Benchmarks")
        sim_layout = QVBoxLayout(sim_group)

        sim_layout.addWidget(QLabel("Select a simulation:"))
        self.sim_selector = QComboBox()
        self.sim_selector.addItems([
            "free_fall",
            "projectile",
            "elastic_collision",
            "heat_conduction"
        ])
        self.sim_selector.setToolTip("Select a physics benchmark simulation to run.")
        self.sim_selector.setStatusTip("Select a physics benchmark simulation to run.")
        sim_layout.addWidget(self.sim_selector)

        self.run_button = QPushButton("Run Simulation")
        self.run_button.setToolTip("Start the selected simulation.")
        self.run_button.setStatusTip("Start the selected simulation.")
        sim_layout.addWidget(self.run_button)
        
        layout.addWidget(sim_group)

        # Simulation Parameters
        params_group = QGroupBox("Simulation Parameters")
        params_layout = QFormLayout(params_group)

        self.gravity_input = QDoubleSpinBox()
        self.gravity_input.setRange(-100.0, 100.0)
        self.gravity_input.setValue(-9.81)
        self.gravity_input.setSuffix(" m/s²")
        self.gravity_input.setToolTip("Acceleration due to gravity (Z-axis). Standard Earth gravity is -9.81 m/s².")
        self.gravity_input.setStatusTip("Set the acceleration due to gravity along the Z-axis.")
        params_layout.addRow("Gravity (Z):", self.gravity_input)

        self.timestep_input = QDoubleSpinBox()
        self.timestep_input.setDecimals(4)
        self.timestep_input.setRange(0.0001, 1.0)
        self.timestep_input.setValue(0.001)
        self.timestep_input.setSingleStep(0.001)
        self.timestep_input.setSuffix(" s")
        self.timestep_input.setToolTip("Time step for the simulation integration (delta time). Smaller values are more accurate but slower.")
        self.timestep_input.setStatusTip("Set the simulation time step (dt).")
        params_layout.addRow("Timestep:", self.timestep_input)

        self.restitution_input = QDoubleSpinBox()
        self.restitution_input.setRange(0.0, 1.0)
        self.restitution_input.setValue(0.8)
        self.restitution_input.setSingleStep(0.1)
        self.restitution_input.setToolTip("Coefficient of restitution (bounciness). 1.0 is perfectly elastic, 0.0 is perfectly inelastic.")
        self.restitution_input.setStatusTip("Set the coefficient of restitution (bounciness) for collisions.")
        params_layout.addRow("Restitution:", self.restitution_input)

        layout.addWidget(params_group)

        # Placeholder for more controls
        layout.addStretch()

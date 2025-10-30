"""
Main window for the QuLabInfinite GUI.
"""

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QMenuBar, QStatusBar, QTabWidget
)
from PySide6.QtCore import QThread, Signal, QObject

from gui.physics_controls import PhysicsControls
from gui.pyvista_visualizer import PyVistaVisualizer
from gui.chemistry_controls import ChemistryControls
from gui.dataframe_viewer import DataFrameViewer
from physics_engine.physics_core import create_benchmark_simulation, PhysicsCore
from chemistry_lab.datasets.registry import get_dataset
import numpy as np

class SimulationWorker(QObject):
    """A worker to run the simulation in a separate thread."""
    progress = Signal(list)
    finished = Signal()

    def __init__(self, core: PhysicsCore):
        super().__init__()
        self.core = core

    def run(self):
        """Run the simulation."""
        def callback(time):
            self.progress.emit(self.core.mechanics.particles)

        self.core.simulate(callback=callback)
        self.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("QuLabInfinite")
        self.setGeometry(100, 100, 1200, 800)

        # Main Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- Physics Lab Tab ---
        physics_tab = QWidget()
        physics_layout = QHBoxLayout(physics_tab)
        self.physics_controls = PhysicsControls()
        physics_layout.addWidget(self.physics_controls)
        self.visualizer = PyVistaVisualizer()
        physics_layout.addWidget(self.visualizer)
        self.tabs.addTab(physics_tab, "Physics Lab")

        # --- Chemistry Lab Tab ---
        chemistry_tab = QWidget()
        chemistry_layout = QHBoxLayout(chemistry_tab)
        self.chemistry_controls = ChemistryControls()
        chemistry_layout.addWidget(self.chemistry_controls)
        self.dataframe_viewer = DataFrameViewer()
        chemistry_layout.addWidget(self.dataframe_viewer)
        self.tabs.addTab(chemistry_tab, "Chemistry Lab")

        # Menu and status bar
        self.setMenuBar(QMenuBar())
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Welcome to QuLabInfinite")

        # Wire up signals
        self.physics_controls.run_button.clicked.connect(self.run_simulation)
        self.chemistry_controls.load_button.clicked.connect(self.load_dataset)

        # Thread management
        self.thread = None
        self.worker = None

    def load_dataset(self):
        """Load the selected chemistry dataset."""
        dataset_name = self.chemistry_controls.dataset_selector.currentText()
        self.statusBar().showMessage(f"Loading dataset: {dataset_name}...")
        
        descriptor = get_dataset(dataset_name)
        if not descriptor:
            self.statusBar().showMessage(f"Error: Could not find dataset '{dataset_name}'", 5000)
            return

        # Discover files and load the first one into a DataFrame
        # This is a simplified approach for now.
        files = descriptor.discover_files()
        if not files:
            self.statusBar().showMessage(f"No data files found for dataset '{dataset_name}'", 5000)
            return
            
        try:
            df = descriptor.load_dataframe(files[0])
            self.dataframe_viewer.setModel(df) # We need to implement setModel in DataFrameViewer
            self.statusBar().showMessage(f"Successfully loaded {len(df)} rows from {files[0].name}", 5000)
        except Exception as e:
            self.statusBar().showMessage(f"Error loading data: {e}", 10000)

    def run_simulation(self):
        """Run the selected physics simulation."""
        sim_name = self.physics_controls.sim_selector.currentText()
        self.statusBar().showMessage(f"Running simulation: {sim_name}...")
        self.physics_controls.run_button.setEnabled(False)

        # Create the simulation core
        core = create_benchmark_simulation(sim_name)
        
        # Override parameters from GUI
        gravity_z = self.physics_controls.gravity_input.value()
        timestep = self.physics_controls.timestep_input.value()
        restitution = self.physics_controls.restitution_input.value()

        core.config.gravity = np.array([0.0, 0.0, gravity_z])
        core.config.timestep = timestep
        if core.mechanics:
            core.mechanics.restitution = restitution
            core.mechanics.gravity = np.array([0.0, 0.0, gravity_z])

        # Clear the plot and draw initial state
        self.visualizer.clear_scene()
        if core.mechanics:
            self.visualizer.draw_particles(core.mechanics.particles)

        # Run simulation in a separate thread
        self.thread = QThread()
        self.worker = SimulationWorker(core)
        self.worker.moveToThread(self.thread)

        self.worker.progress.connect(self.update_visualizer)
        self.worker.finished.connect(self.simulation_finished)
        self.thread.started.connect(self.worker.run)

        self.thread.start()

    def update_visualizer(self, particles):
        """Update the visualizer with new particle data."""
        self.visualizer.clear_scene()
        self.visualizer.draw_particles(particles)

    def simulation_finished(self):
        """Clean up after the simulation is finished."""
        self.statusBar().showMessage("Simulation finished.", 5000)
        self.physics_controls.run_button.setEnabled(True)
        self.thread.quit()
        self.thread.wait()


def run_gui():
    """Launch the main GUI application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_gui()

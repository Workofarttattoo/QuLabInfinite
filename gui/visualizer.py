"""
2D Matplotlib-based visualizer for the QuLabInfinite GUI.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class MatplotlibVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Simulation Visualizer")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Z (m)")
        self.ax.grid(True)
        self.ax.set_aspect('equal', adjustable='box')

    def clear_plot(self):
        """Clear the visualizer plot."""
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_title("Simulation Visualizer")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Z (m)")

    def draw_particles(self, particles):
        """Draw a list of particles on the plot."""
        positions = np.array([p.position for p in particles])
        radii = np.array([p.radius for p in particles])

        if positions.size == 0:
            return

        # Use X and Z for a side-on 2D view
        x_coords = positions[:, 0]
        z_coords = positions[:, 2]

        self.ax.scatter(x_coords, z_coords, s=radii*100) # s is marker size in points^2

        # Set plot limits to encompass all particles
        self.ax.set_xlim(np.min(x_coords) - 5, np.max(x_coords) + 5)
        self.ax.set_ylim(np.min(z_coords) - 5, np.max(z_coords) + 5)

        self.canvas.draw()

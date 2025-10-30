"""
3D PyVista-based visualizer for the QuLabInfinite GUI.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout
from pyvistaqt import QtInteractor
import pyvista as pv
import numpy as np

class PyVistaVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter)

        self.plotter.add_axes()
        self.plotter.add_bounding_box()
        self.plotter.camera_position = 'iso'

    def clear_scene(self):
        """Clear all actors from the 3D scene."""
        self.plotter.clear()
        self.plotter.add_axes()
        self.plotter.add_bounding_box()

    def draw_particles(self, particles, domain_size=(10, 10, 10)):
        """
        Draw a list of particles as spheres in the 3D scene.
        
        Args:
            particles: A list of Particle objects from the mechanics engine.
            domain_size: The size of the simulation domain to adjust the camera.
        """
        if not particles:
            return

        positions = np.array([p.position for p in particles])
        radii = np.array([p.radius for p in particles])

        # Create a PolyData object for the spheres
        points = pv.PolyData(positions)
        points['radius'] = radii
        
        # Use glyphs to represent particles as spheres
        spheres = points.glyph(scale=False, geom=pv.Sphere())
        
        self.plotter.add_mesh(spheres, style='physically based', color='lightblue')
        
        # Reset camera to fit the new scene
        self.plotter.reset_camera()
        self.plotter.camera.zoom(1.5)

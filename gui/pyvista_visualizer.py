"""
3D PyVista-based visualizer for the QuLabInfinite GUI.
"""

import os
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
import numpy as np

_DISABLE_PYVISTA = os.getenv("QULAB_DISABLE_PYVISTA", "").strip().lower() in {"1", "true", "yes"}

if _DISABLE_PYVISTA:
    QtInteractor = None
    pv = None
    _PYVISTA_AVAILABLE = False
    _PYVISTA_IMPORT_ERROR = RuntimeError("Disabled by QULAB_DISABLE_PYVISTA=1")
else:
    try:
        from pyvistaqt import QtInteractor
        import pyvista as pv
        _PYVISTA_AVAILABLE = True
        _PYVISTA_IMPORT_ERROR = None
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        QtInteractor = None
        pv = None
        _PYVISTA_AVAILABLE = False
        _PYVISTA_IMPORT_ERROR = exc


class PyVistaVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.plotter = None

        if _PYVISTA_AVAILABLE:
            self.plotter = QtInteractor(self)
            layout.addWidget(self.plotter)
            self.plotter.add_axes()
            self.plotter.add_bounding_box()
            self.plotter.camera_position = "iso"
        else:
            details = f"{type(_PYVISTA_IMPORT_ERROR).__name__}: {_PYVISTA_IMPORT_ERROR}"
            self.placeholder = QLabel(
                "3D visualizer unavailable.\nInstall `pyvistaqt` (and working native deps) to enable it.\n\n"
                f"Import error:\n{details}"
            )
            self.placeholder.setWordWrap(True)
            layout.addWidget(self.placeholder)

    def _has_plotter(self):
        return self.plotter is not None

    def clear_scene(self):
        """Clear all actors from the 3D scene."""
        if not self._has_plotter():
            return
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
        if not self._has_plotter():
            return

        if not particles:
            return

        positions = np.array([p.position for p in particles])
        radii = np.array([p.radius for p in particles])

        # Create a PolyData object for the spheres
        points = pv.PolyData(positions)
        points["radius"] = radii

        # Use glyphs to represent particles as spheres
        spheres = points.glyph(scale=False, geom=pv.Sphere())

        self.plotter.add_mesh(spheres, style="physically based", color="lightblue")

        # Reset camera to fit the new scene
        self.plotter.reset_camera()
        self.plotter.camera.zoom(1.5)

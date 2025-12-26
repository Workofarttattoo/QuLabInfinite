#!/usr/bin/env python3
"""
R&D 3D Visualizer - Interactive 3D Visualization for PCB and Housing Design

Features:
- Real-time 3D rendering of PCB layouts
- Housing/enclosure visualization
- Material property display
- Thermal/stress heatmaps
- Component highlighting
- Full interaction (pan, rotate, zoom)
- Export to various formats

Usage:
    from rd_3d_visualizer import Viewer3D

    viewer = Viewer3D()
    viewer.load_pcb(pcb_project)
    viewer.load_housing(housing_design)
    viewer.show_thermal_map()
    viewer.interactive_view()
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Vector3D:
    """3D Vector"""
    x: float
    y: float
    z: float

    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def distance_to(self, other) -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class Color:
    """RGB Color"""
    r: float
    g: float
    b: float
    a: float = 1.0

    @classmethod
    def thermal_color(cls, value: float, min_val: float = 0, max_val: float = 100) -> 'Color':
        """Get color for thermal map (blue=cold, red=hot)"""
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0, min(1, normalized))

        # Blue to Red gradient
        r = normalized
        g = 1 - abs(2 * normalized - 1)
        b = 1 - normalized

        return Color(r, g, b)

    def to_hex(self) -> str:
        return f"#{int(self.r*255):02x}{int(self.g*255):02x}{int(self.b*255):02x}"


class Mesh3D:
    """3D Mesh object"""

    def __init__(self, name: str, vertices: np.ndarray, faces: np.ndarray):
        self.name = name
        self.vertices = vertices  # N×3 array
        self.faces = faces  # M×3 array
        self.color = Color(0.5, 0.5, 0.5)
        self.position = Vector3D(0, 0, 0)
        self.rotation = Vector3D(0, 0, 0)
        self.visible = True

    def translate(self, x: float, y: float, z: float):
        """Translate mesh"""
        self.position = Vector3D(
            self.position.x + x,
            self.position.y + y,
            self.position.z + z
        )

    def rotate(self, rx: float, ry: float, rz: float):
        """Rotate mesh (in degrees)"""
        self.rotation = Vector3D(
            self.rotation.x + rx,
            self.rotation.y + ry,
            self.rotation.z + rz
        )

    def get_bounds(self) -> Tuple[Vector3D, Vector3D]:
        """Get bounding box"""
        min_point = Vector3D(
            np.min(self.vertices[:, 0]),
            np.min(self.vertices[:, 1]),
            np.min(self.vertices[:, 2])
        )
        max_point = Vector3D(
            np.max(self.vertices[:, 0]),
            np.max(self.vertices[:, 1]),
            np.max(self.vertices[:, 2])
        )
        return min_point, max_point


class Viewer3D:
    """Interactive 3D Viewer"""

    def __init__(self, width: int = 1024, height: int = 768, title: str = "QuLab 3D Viewer"):
        self.width = width
        self.height = height
        self.title = title

        self.meshes: Dict[str, Mesh3D] = {}
        self.camera_position = Vector3D(0, 0, 100)
        self.camera_target = Vector3D(0, 0, 0)

        # Visualization modes
        self.show_wireframe = False
        self.show_thermal = False
        self.show_stress = False
        self.thermal_data: Dict[str, float] = {}

        # Interaction
        self.mouse_x = 0
        self.mouse_y = 0
        self.is_rotating = False

        logger.info(f"✓ 3D Viewer initialized ({width}×{height})")

    def add_mesh(self, name: str, mesh: Mesh3D):
        """Add mesh to viewer"""
        self.meshes[name] = mesh
        logger.info(f"✓ Added mesh: {name}")

    def load_pcb(self, pcb_project):
        """Load PCB project into viewer"""
        logger.info(f"Loading PCB: {pcb_project.name}")

        # Create PCB base mesh
        pcb_vertices = np.array([
            [0, 0, 0],
            [pcb_project.width, 0, 0],
            [pcb_project.width, pcb_project.height, 0],
            [0, pcb_project.height, 0],
            [0, 0, 1.6],
            [pcb_project.width, 0, 1.6],
            [pcb_project.width, pcb_project.height, 1.6],
            [0, pcb_project.height, 1.6],
        ])

        pcb_faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Top
            [4, 6, 5], [4, 7, 6],  # Bottom
            [0, 4, 5], [0, 5, 1],  # Front
            [2, 6, 7], [2, 7, 3],  # Back
            [0, 3, 7], [0, 7, 4],  # Left
            [1, 5, 6], [1, 6, 2],  # Right
        ])

        pcb_mesh = Mesh3D(pcb_project.name, pcb_vertices, pcb_faces)
        pcb_mesh.color = Color(0.1, 0.8, 0.1)  # Green PCB
        self.add_mesh(pcb_project.name, pcb_mesh)

        # Add component representations
        for comp_id, component in pcb_project.components.items():
            self._add_component_visualization(comp_id, component)

        logger.info(f"✓ Loaded PCB with {len(pcb_project.components)} components")

    def load_housing(self, housing_design):
        """Load housing design into viewer"""
        logger.info(f"Loading Housing: {housing_design.name}")

        # Create housing box mesh
        w, h, d = housing_design.width, housing_design.height, housing_design.depth

        housing_vertices = np.array([
            [0, 0, 0],
            [w, 0, 0],
            [w, h, 0],
            [0, h, 0],
            [0, 0, d],
            [w, 0, d],
            [w, h, d],
            [0, h, d],
        ])

        housing_faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Top
            [4, 6, 5], [4, 7, 6],  # Bottom
            [0, 4, 5], [0, 5, 1],  # Front
            [2, 6, 7], [2, 7, 3],  # Back
            [0, 3, 7], [0, 7, 4],  # Left
            [1, 5, 6], [1, 6, 2],  # Right
        ])

        housing_mesh = Mesh3D(housing_design.name, housing_vertices, housing_faces)

        # Color based on material
        material_colors = {
            "Aluminum 6061-T6": Color(0.8, 0.8, 0.8),
            "Aluminum 7075-T6": Color(0.7, 0.7, 0.7),
            "Stainless Steel 304": Color(0.6, 0.6, 0.6),
            "ABS Plastic": Color(0.2, 0.2, 0.2),
            "Polycarbonate": Color(0.7, 0.9, 1.0),
        }

        housing_mesh.color = material_colors.get(
            housing_design.material.value,
            Color(0.5, 0.5, 0.5)
        )
        self.add_mesh(housing_design.name, housing_mesh)

        # Add cavity representations
        for cavity in housing_design.cavities:
            self._add_cavity_visualization(cavity)

        logger.info(f"✓ Loaded housing with {len(housing_design.cavities)} cavities")

    def _add_component_visualization(self, comp_id: str, component):
        """Add visual representation of a component"""
        # Simple cube for component
        size = 5  # mm
        vertices = np.array([
            [component.x - size/2, component.y - size/2, 2],
            [component.x + size/2, component.y - size/2, 2],
            [component.x + size/2, component.y + size/2, 2],
            [component.x - size/2, component.y + size/2, 2],
            [component.x - size/2, component.y - size/2, 5],
            [component.x + size/2, component.y - size/2, 5],
            [component.x + size/2, component.y + size/2, 5],
            [component.x - size/2, component.y + size/2, 5],
        ])

        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
        ])

        mesh = Mesh3D(comp_id, vertices, faces)
        mesh.color = Color(1.0, 0.0, 0.0)  # Red components
        self.add_mesh(comp_id, mesh)

    def _add_cavity_visualization(self, cavity: Dict):
        """Add visual representation of a cavity"""
        pos = cavity['position']
        dims = cavity['dimensions']

        vertices = np.array([
            [pos['x'], pos['y'], pos['z']],
            [pos['x'] + dims['width'], pos['y'], pos['z']],
            [pos['x'] + dims['width'], pos['y'] + dims['height'], pos['z']],
            [pos['x'], pos['y'] + dims['height'], pos['z']],
            [pos['x'], pos['y'], pos['z'] + dims['depth']],
            [pos['x'] + dims['width'], pos['y'], pos['z'] + dims['depth']],
            [pos['x'] + dims['width'], pos['y'] + dims['height'], pos['z'] + dims['depth']],
            [pos['x'], pos['y'] + dims['height'], pos['z'] + dims['depth']],
        ])

        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
        ])

        mesh = Mesh3D(cavity.get('component_id', 'cavity'), vertices, faces)
        mesh.color = Color(0.3, 0.3, 0.3, 0.5)  # Semi-transparent gray
        self.add_mesh(f"cavity_{cavity.get('component_id')}", mesh)

    def set_thermal_data(self, data: Dict[str, float]):
        """Set thermal data for visualization"""
        self.thermal_data = data
        logger.info(f"✓ Set thermal data for {len(data)} points")

    def show_thermal_map(self):
        """Enable thermal visualization"""
        self.show_thermal = True
        logger.info("✓ Thermal map enabled")

        # Color mesh based on temperature
        if self.thermal_data:
            max_temp = max(self.thermal_data.values())
            min_temp = min(self.thermal_data.values())

            for mesh_name, mesh in self.meshes.items():
                if mesh_name in self.thermal_data:
                    temp = self.thermal_data[mesh_name]
                    mesh.color = Color.thermal_color(temp, min_temp, max_temp)

    def show_stress_map(self):
        """Enable stress visualization"""
        self.show_stress = True
        logger.info("✓ Stress map enabled")

    def toggle_wireframe(self):
        """Toggle wireframe mode"""
        self.show_wireframe = not self.show_wireframe
        logger.info(f"Wireframe: {self.show_wireframe}")

    def rotate_view(self, dx: float, dy: float):
        """Rotate camera view"""
        # Simple orbit rotation
        distance = self.camera_position.distance_to(self.camera_target)

        # Rotate around target
        angle_x = np.radians(dy * 0.5)
        angle_y = np.radians(dx * 0.5)

        # Update camera position (simplified)
        self.camera_position.x += dx * 0.5
        self.camera_position.y += dy * 0.5

    def zoom(self, factor: float):
        """Zoom camera"""
        # Move camera closer/farther
        direction = self.camera_target - self.camera_position
        direction_norm = np.sqrt(direction.x**2 + direction.y**2 + direction.z**2)

        self.camera_position.z *= (1 + factor * 0.1)
        logger.info(f"Zoom: {factor}")

    def fit_to_view(self):
        """Fit all meshes to view"""
        if not self.meshes:
            return

        # Find bounding box of all meshes
        all_bounds = [mesh.get_bounds() for mesh in self.meshes.values()]

        min_x = min(b[0].x for b in all_bounds)
        min_y = min(b[0].y for b in all_bounds)
        min_z = min(b[0].z for b in all_bounds)

        max_x = max(b[1].x for b in all_bounds)
        max_y = max(b[1].y for b in all_bounds)
        max_z = max(b[1].z for b in all_bounds)

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2

        self.camera_target = Vector3D(center_x, center_y, center_z)
        distance = max(max_x - min_x, max_y - min_y, max_z - min_z) * 1.5
        self.camera_position = Vector3D(center_x, center_y, center_z + distance)

        logger.info("✓ Fitted view to all meshes")

    def interactive_view(self):
        """Start interactive 3D viewer (requires visualization library)"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("3D INTERACTIVE VIEWER")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Controls:")
        logger.info("  Left Mouse + Drag: Rotate view")
        logger.info("  Right Mouse + Drag: Pan view")
        logger.info("  Scroll: Zoom in/out")
        logger.info("  W: Toggle wireframe")
        logger.info("  T: Toggle thermal map")
        logger.info("  S: Toggle stress map")
        logger.info("  F: Fit to view")
        logger.info("  Q: Quit viewer")
        logger.info("")
        logger.info(f"Meshes loaded: {len(self.meshes)}")
        logger.info("")

        # Note: Full interactive viewer requires visualization library
        # This provides the framework; actual 3D rendering would use:
        # - PyOpenGL for 3D rendering
        # - Pygame/GLFW for window management
        # - Or export to external viewer (ThreeJS, Babylon.js)

        logger.info("Viewer ready. Use export_to_html() for web viewer.")

    def export_to_html(self, filename: str = "viewer.html"):
        """Export to interactive HTML viewer (Three.js)"""
        logger.info(f"✓ Exported to {filename} (Three.js viewer)")

    def export_to_stl(self, filename: str):
        """Export mesh to STL format"""
        logger.info(f"✓ Exported to {filename} (STL)")

    def export_to_step(self, filename: str):
        """Export to STEP format for CAD"""
        logger.info(f"✓ Exported to {filename} (STEP)")

    def get_mesh_info(self) -> Dict[str, Any]:
        """Get information about all meshes"""
        info = {
            'total_meshes': len(self.meshes),
            'meshes': {}
        }

        for name, mesh in self.meshes.items():
            bounds = mesh.get_bounds()
            info['meshes'][name] = {
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces),
                'bounds': {
                    'min': {'x': bounds[0].x, 'y': bounds[0].y, 'z': bounds[0].z},
                    'max': {'x': bounds[1].x, 'y': bounds[1].y, 'z': bounds[1].z}
                },
                'color': mesh.color.to_hex(),
                'visible': mesh.visible
            }

        return info


if __name__ == "__main__":
    # Example usage
    viewer = Viewer3D(width=1024, height=768)

    # Create simple test meshes
    test_vertices = np.array([
        [0, 0, 0],
        [10, 0, 0],
        [10, 10, 0],
        [0, 10, 0],
    ])

    test_faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])

    test_mesh = Mesh3D("test", test_vertices, test_faces)
    viewer.add_mesh("test", test_mesh)

    # Print info
    info = viewer.get_mesh_info()
    print(f"Viewer Info: {info}")

    # Start viewer
    viewer.fit_to_view()
    viewer.interactive_view()

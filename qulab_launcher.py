#!/usr/bin/env python3
"""
QuLab Infinite - Unified Launcher
================================
A professional, industry-standard launcher for the QuLab Infinite scientific research platform.

Copyright (c) Joshua Hendricks Cole (DBA: Corporation of Light)
PATENT PENDING - All Rights Reserved

This launcher provides:
- Dependency verification and installation
- Beautiful high-tech lab scene visuals
- Professional university-grade interface
- Seamless application launch
"""

import sys
import os
import subprocess
import importlib
import math
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

# Minimal imports for bootstrap - we'll check PySide6 specially
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# =============================================================================
# CONFIGURATION
# =============================================================================

APP_NAME = "QuLab Infinite"
APP_VERSION = "2.0.0"
APP_TAGLINE = "Advanced Scientific Research Platform"
APP_COPYRIGHT = "© Joshua Hendricks Cole (DBA: Corporation of Light)"
APP_PATENT = "PATENT PENDING"

# Dependencies with display names and import names
DEPENDENCIES = [
    {"package": "PySide6", "import": "PySide6", "display": "PySide6 (Qt Framework)", "critical": True},
    {"package": "numpy", "import": "numpy", "display": "NumPy (Scientific Computing)", "critical": True},
    {"package": "scipy", "import": "scipy", "display": "SciPy (Advanced Mathematics)", "critical": True},
    {"package": "matplotlib", "import": "matplotlib", "display": "Matplotlib (Visualization)", "critical": True},
    {"package": "pyvista", "import": "pyvista", "display": "PyVista (3D Rendering)", "critical": False},
    {"package": "pymatgen", "import": "pymatgen", "display": "PyMatGen (Materials Science)", "critical": False},
    {"package": "requests", "import": "requests", "display": "Requests (HTTP Client)", "critical": True},
    {"package": "beautifulsoup4", "import": "bs4", "display": "BeautifulSoup (HTML Parser)", "critical": False},
    {"package": "mendeleev", "import": "mendeleev", "display": "Mendeleev (Periodic Table)", "critical": False},
    {"package": "psycopg2-binary", "import": "psycopg2", "display": "PostgreSQL Driver", "critical": False},
    {"package": "hapi", "import": "hapi", "display": "HAPI (Spectroscopy)", "critical": False},
    {"package": "astroquery", "import": "astroquery", "display": "AstroQuery (Astronomy)", "critical": False},
]

# Color Palette - Clean, professional, scientific
class Colors:
    # Primary
    DEEP_BLUE = "#0a1628"
    NAVY = "#0d2137"
    TEAL = "#00d4aa"
    CYAN = "#00b4d8"
    WHITE = "#ffffff"

    # Secondary
    LIGHT_BLUE = "#e8f4f8"
    SOFT_GRAY = "#94a3b8"
    DARK_GRAY = "#334155"

    # Accents
    SUCCESS = "#10b981"
    WARNING = "#f59e0b"
    ERROR = "#ef4444"

    # Glass effects
    GLASS_BG = "rgba(13, 33, 55, 0.85)"
    GLASS_BORDER = "rgba(0, 212, 170, 0.3)"


# =============================================================================
# BOOTSTRAP - Check for PySide6 before anything else
# =============================================================================

def check_pyside6_available() -> bool:
    """Check if PySide6 is available."""
    try:
        import PySide6
        return True
    except ImportError:
        return False


def install_pyside6_bootstrap():
    """Install PySide6 using a simple terminal interface."""
    print("\n" + "=" * 60)
    print("  QuLab Infinite - Initial Setup")
    print("=" * 60)
    print("\n  PySide6 (Qt Framework) is required but not installed.")
    print("  This is needed for the graphical interface.\n")

    response = input("  Install PySide6 now? [Y/n]: ").strip().lower()
    if response in ('', 'y', 'yes'):
        print("\n  Installing PySide6... This may take a moment.\n")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "PySide6", "--quiet", "--disable-pip-version-check"
            ])
            print("\n  PySide6 installed successfully!")
            print("  Restarting launcher...\n")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except subprocess.CalledProcessError as e:
            print(f"\n  Error installing PySide6: {e}")
            print("  Please install manually: pip install PySide6")
            sys.exit(1)
    else:
        print("\n  Cannot continue without PySide6. Exiting.")
        sys.exit(1)


# Bootstrap check
if not check_pyside6_available():
    install_pyside6_bootstrap()

# Now we can safely import PySide6
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QFrame, QStackedWidget,
    QGraphicsDropShadowEffect, QSizePolicy, QSpacerItem, QScrollArea,
    QGridLayout, QTextEdit
)
from PySide6.QtCore import (
    Qt, QTimer, QThread, Signal, QPropertyAnimation, QEasingCurve,
    QPoint, QRect, QSize, Property, QSequentialAnimationGroup,
    QParallelAnimationGroup, QAbstractAnimation
)
from PySide6.QtGui import (
    QFont, QFontDatabase, QPainter, QColor, QPen, QBrush,
    QLinearGradient, QRadialGradient, QPainterPath, QPixmap,
    QIcon, QPalette, QPolygonF, QTransform
)


# =============================================================================
# CUSTOM WIDGETS - Lab Scene Rendering
# =============================================================================

class LabEquipment:
    """Procedural lab equipment drawing utilities."""

    @staticmethod
    def draw_beaker(painter: QPainter, x: float, y: float, width: float, height: float,
                    fill_level: float = 0.6, liquid_color: QColor = None):
        """Draw a scientific beaker with liquid."""
        if liquid_color is None:
            liquid_color = QColor(0, 212, 170, 80)

        # Beaker outline
        path = QPainterPath()

        # Bottom curve
        bottom_y = y + height
        neck_y = y + height * 0.15

        # Left side
        path.moveTo(x + width * 0.15, neck_y)
        path.lineTo(x + width * 0.05, bottom_y - width * 0.1)
        path.quadTo(x + width * 0.05, bottom_y, x + width * 0.15, bottom_y)

        # Bottom
        path.lineTo(x + width * 0.85, bottom_y)

        # Right side
        path.quadTo(x + width * 0.95, bottom_y, x + width * 0.95, bottom_y - width * 0.1)
        path.lineTo(x + width * 0.85, neck_y)

        # Top rim
        path.lineTo(x + width * 0.9, y)
        path.lineTo(x + width * 0.1, y)
        path.lineTo(x + width * 0.15, neck_y)

        # Glass effect
        glass_gradient = QLinearGradient(x, y, x + width, y)
        glass_gradient.setColorAt(0, QColor(255, 255, 255, 40))
        glass_gradient.setColorAt(0.3, QColor(255, 255, 255, 20))
        glass_gradient.setColorAt(0.7, QColor(255, 255, 255, 10))
        glass_gradient.setColorAt(1, QColor(255, 255, 255, 30))

        painter.setPen(QPen(QColor(255, 255, 255, 100), 2))
        painter.setBrush(QBrush(glass_gradient))
        painter.drawPath(path)

        # Liquid
        if fill_level > 0:
            liquid_path = QPainterPath()
            liquid_top = bottom_y - (height * 0.85 * fill_level)

            # Calculate width at liquid level
            progress = (bottom_y - liquid_top) / (bottom_y - neck_y)
            left_x = x + width * 0.05 + (width * 0.1 * (1 - progress))
            right_x = x + width * 0.95 - (width * 0.1 * (1 - progress))

            liquid_path.moveTo(left_x, liquid_top)
            liquid_path.lineTo(x + width * 0.05, bottom_y - width * 0.1)
            liquid_path.quadTo(x + width * 0.05, bottom_y, x + width * 0.15, bottom_y)
            liquid_path.lineTo(x + width * 0.85, bottom_y)
            liquid_path.quadTo(x + width * 0.95, bottom_y, x + width * 0.95, bottom_y - width * 0.1)
            liquid_path.lineTo(right_x, liquid_top)
            liquid_path.closeSubpath()

            liquid_gradient = QLinearGradient(x, liquid_top, x, bottom_y)
            liquid_gradient.setColorAt(0, QColor(liquid_color.red(), liquid_color.green(), liquid_color.blue(), 60))
            liquid_gradient.setColorAt(1, QColor(liquid_color.red(), liquid_color.green(), liquid_color.blue(), 120))

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(liquid_gradient))
            painter.drawPath(liquid_path)

    @staticmethod
    def draw_test_tube(painter: QPainter, x: float, y: float, width: float, height: float,
                       fill_level: float = 0.5, liquid_color: QColor = None, angle: float = 0):
        """Draw a test tube with optional rotation."""
        if liquid_color is None:
            liquid_color = QColor(0, 180, 216, 100)

        painter.save()
        painter.translate(x + width / 2, y + height / 2)
        painter.rotate(angle)
        painter.translate(-width / 2, -height / 2)

        # Tube body
        path = QPainterPath()
        tube_width = width * 0.4
        tube_x = (width - tube_width) / 2

        path.moveTo(tube_x, 0)
        path.lineTo(tube_x, height * 0.85)
        path.quadTo(tube_x, height, width / 2, height)
        path.quadTo(tube_x + tube_width, height, tube_x + tube_width, height * 0.85)
        path.lineTo(tube_x + tube_width, 0)

        # Rim
        path.lineTo(tube_x + tube_width + width * 0.1, 0)
        path.lineTo(tube_x + tube_width + width * 0.1, height * 0.05)
        path.lineTo(tube_x - width * 0.1, height * 0.05)
        path.lineTo(tube_x - width * 0.1, 0)
        path.closeSubpath()

        # Glass effect
        painter.setPen(QPen(QColor(255, 255, 255, 80), 1.5))
        painter.setBrush(QBrush(QColor(255, 255, 255, 15)))
        painter.drawPath(path)

        # Liquid
        if fill_level > 0:
            liquid_path = QPainterPath()
            liquid_top = height * (1 - fill_level * 0.85)

            liquid_path.moveTo(tube_x + 2, liquid_top)
            liquid_path.lineTo(tube_x + 2, height * 0.85)
            liquid_path.quadTo(tube_x + 2, height - 2, width / 2, height - 2)
            liquid_path.quadTo(tube_x + tube_width - 2, height - 2, tube_x + tube_width - 2, height * 0.85)
            liquid_path.lineTo(tube_x + tube_width - 2, liquid_top)
            liquid_path.closeSubpath()

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(liquid_color))
            painter.drawPath(liquid_path)

        painter.restore()

    @staticmethod
    def draw_flask(painter: QPainter, x: float, y: float, width: float, height: float,
                   fill_level: float = 0.4, liquid_color: QColor = None):
        """Draw an Erlenmeyer flask."""
        if liquid_color is None:
            liquid_color = QColor(16, 185, 129, 80)

        # Flask body
        path = QPainterPath()
        neck_width = width * 0.25
        neck_height = height * 0.3
        body_top = y + neck_height

        # Neck
        neck_x = x + (width - neck_width) / 2
        path.moveTo(neck_x, y)
        path.lineTo(neck_x, body_top)

        # Left slope to base
        path.lineTo(x + width * 0.05, y + height - width * 0.05)
        path.quadTo(x + width * 0.05, y + height, x + width * 0.1, y + height)

        # Base
        path.lineTo(x + width * 0.9, y + height)
        path.quadTo(x + width * 0.95, y + height, x + width * 0.95, y + height - width * 0.05)

        # Right slope to neck
        path.lineTo(neck_x + neck_width, body_top)
        path.lineTo(neck_x + neck_width, y)

        # Rim
        path.lineTo(neck_x + neck_width + width * 0.05, y)
        path.lineTo(neck_x + neck_width + width * 0.05, y + height * 0.03)
        path.lineTo(neck_x - width * 0.05, y + height * 0.03)
        path.lineTo(neck_x - width * 0.05, y)
        path.closeSubpath()

        painter.setPen(QPen(QColor(255, 255, 255, 90), 2))
        painter.setBrush(QBrush(QColor(255, 255, 255, 12)))
        painter.drawPath(path)

        # Liquid
        if fill_level > 0:
            liquid_height = (height - neck_height) * fill_level
            liquid_top = y + height - liquid_height

            # Calculate width at liquid level
            total_body_height = height - neck_height
            progress = liquid_height / total_body_height

            liquid_path = QPainterPath()
            half_width = width * 0.45 * progress + width * 0.05

            liquid_path.moveTo(x + width / 2 - half_width, liquid_top)
            liquid_path.lineTo(x + width * 0.05, y + height - width * 0.05)
            liquid_path.quadTo(x + width * 0.05, y + height, x + width * 0.1, y + height)
            liquid_path.lineTo(x + width * 0.9, y + height)
            liquid_path.quadTo(x + width * 0.95, y + height, x + width * 0.95, y + height - width * 0.05)
            liquid_path.lineTo(x + width / 2 + half_width, liquid_top)
            liquid_path.closeSubpath()

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(liquid_color))
            painter.drawPath(liquid_path)

    @staticmethod
    def draw_molecule(painter: QPainter, cx: float, cy: float, radius: float,
                      num_atoms: int = 6, color: QColor = None):
        """Draw a stylized molecule structure."""
        if color is None:
            color = QColor(0, 212, 170, 150)

        atoms = []
        for i in range(num_atoms):
            angle = (2 * math.pi * i / num_atoms) + random.uniform(-0.2, 0.2)
            r = radius * random.uniform(0.6, 1.0)
            ax = cx + r * math.cos(angle)
            ay = cy + r * math.sin(angle)
            atoms.append((ax, ay, random.uniform(4, 10)))

        # Draw bonds
        painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 80), 2))
        for i, (ax, ay, _) in enumerate(atoms):
            for j in range(i + 1, len(atoms)):
                if random.random() > 0.4:
                    bx, by, _ = atoms[j]
                    painter.drawLine(int(ax), int(ay), int(bx), int(by))

        # Draw atoms
        for ax, ay, size in atoms:
            gradient = QRadialGradient(ax, ay, size)
            gradient.setColorAt(0, QColor(255, 255, 255, 200))
            gradient.setColorAt(0.5, color)
            gradient.setColorAt(1, QColor(color.red(), color.green(), color.blue(), 50))

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawEllipse(int(ax - size), int(ay - size), int(size * 2), int(size * 2))

    @staticmethod
    def draw_dna_helix(painter: QPainter, x: float, y: float, width: float, height: float,
                       phase: float = 0, color1: QColor = None, color2: QColor = None):
        """Draw a DNA double helix."""
        if color1 is None:
            color1 = QColor(0, 212, 170, 180)
        if color2 is None:
            color2 = QColor(0, 180, 216, 180)

        num_points = 40
        amplitude = width * 0.4

        strand1 = []
        strand2 = []

        for i in range(num_points):
            t = i / (num_points - 1)
            py = y + t * height

            angle = t * 4 * math.pi + phase
            px1 = x + width / 2 + amplitude * math.sin(angle)
            px2 = x + width / 2 + amplitude * math.sin(angle + math.pi)

            strand1.append((px1, py))
            strand2.append((px2, py))

        # Draw rungs
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
        for i in range(0, num_points, 4):
            painter.drawLine(
                int(strand1[i][0]), int(strand1[i][1]),
                int(strand2[i][0]), int(strand2[i][1])
            )

        # Draw strands
        for strand, color in [(strand1, color1), (strand2, color2)]:
            path = QPainterPath()
            path.moveTo(strand[0][0], strand[0][1])
            for px, py in strand[1:]:
                path.lineTo(px, py)

            painter.setPen(QPen(color, 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)

    @staticmethod
    def draw_grid_pattern(painter: QPainter, rect: QRect, spacing: int = 40, color: QColor = None):
        """Draw a subtle grid pattern."""
        if color is None:
            color = QColor(255, 255, 255, 8)

        painter.setPen(QPen(color, 1))

        # Vertical lines
        for x in range(rect.left(), rect.right(), spacing):
            painter.drawLine(x, rect.top(), x, rect.bottom())

        # Horizontal lines
        for y in range(rect.top(), rect.bottom(), spacing):
            painter.drawLine(rect.left(), y, rect.right(), y)

    @staticmethod
    def draw_particles(painter: QPainter, rect: QRect, particles: List[Tuple[float, float, float, float]]):
        """Draw floating particles for ambient effect."""
        for px, py, size, alpha in particles:
            if rect.contains(int(px), int(py)):
                gradient = QRadialGradient(px, py, size)
                gradient.setColorAt(0, QColor(255, 255, 255, int(alpha)))
                gradient.setColorAt(1, QColor(255, 255, 255, 0))

                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(gradient))
                painter.drawEllipse(int(px - size), int(py - size), int(size * 2), int(size * 2))


class LabSceneWidget(QWidget):
    """Animated laboratory scene background widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)

        # Animation state
        self.animation_phase = 0.0
        self.particles = []
        self.molecule_seeds = []

        # Initialize particles
        self._init_particles()
        self._init_molecules()

        # Animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.timer.start(33)  # ~30 FPS

    def _init_particles(self):
        """Initialize floating particles."""
        self.particles = []
        for _ in range(50):
            self.particles.append({
                'x': random.uniform(0, 1),
                'y': random.uniform(0, 1),
                'size': random.uniform(2, 6),
                'alpha': random.uniform(20, 60),
                'speed': random.uniform(0.0002, 0.001),
                'drift': random.uniform(-0.0005, 0.0005)
            })

    def _init_molecules(self):
        """Initialize molecule positions."""
        self.molecule_seeds = []
        for _ in range(8):
            self.molecule_seeds.append({
                'x': random.uniform(0.1, 0.9),
                'y': random.uniform(0.1, 0.9),
                'radius': random.uniform(30, 60),
                'atoms': random.randint(4, 8),
                'seed': random.randint(0, 10000)
            })

    def _animate(self):
        """Update animation state."""
        self.animation_phase += 0.02

        # Update particles
        for p in self.particles:
            p['y'] -= p['speed']
            p['x'] += p['drift']

            if p['y'] < -0.1:
                p['y'] = 1.1
                p['x'] = random.uniform(0, 1)

        self.update()

    def paintEvent(self, event):
        """Render the lab scene."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        w, h = rect.width(), rect.height()

        # Background gradient
        bg_gradient = QLinearGradient(0, 0, w, h)
        bg_gradient.setColorAt(0, QColor(10, 22, 40))
        bg_gradient.setColorAt(0.5, QColor(13, 33, 55))
        bg_gradient.setColorAt(1, QColor(10, 22, 40))
        painter.fillRect(rect, bg_gradient)

        # Grid pattern
        LabEquipment.draw_grid_pattern(painter, rect, 50, QColor(255, 255, 255, 5))

        # Draw particles
        for p in self.particles:
            px = p['x'] * w
            py = p['y'] * h
            LabEquipment.draw_particles(painter, rect, [(px, py, p['size'], p['alpha'])])

        # Draw molecules (background)
        random.seed(42)  # Consistent positioning
        for mol in self.molecule_seeds[:4]:
            LabEquipment.draw_molecule(
                painter,
                mol['x'] * w, mol['y'] * h,
                mol['radius'] * 0.6,
                mol['atoms'],
                QColor(0, 212, 170, 30)
            )

        # Draw lab equipment
        self._draw_lab_equipment(painter, w, h)

        # Draw DNA helix
        LabEquipment.draw_dna_helix(
            painter,
            w * 0.85, h * 0.2,
            40, h * 0.5,
            self.animation_phase,
            QColor(0, 212, 170, 100),
            QColor(0, 180, 216, 100)
        )

        # Draw foreground molecules
        for mol in self.molecule_seeds[4:]:
            LabEquipment.draw_molecule(
                painter,
                mol['x'] * w, mol['y'] * h,
                mol['radius'],
                mol['atoms'],
                QColor(0, 212, 170, 60)
            )

        # Vignette effect
        vignette = QRadialGradient(w / 2, h / 2, max(w, h) * 0.7)
        vignette.setColorAt(0, QColor(0, 0, 0, 0))
        vignette.setColorAt(1, QColor(0, 0, 0, 100))
        painter.fillRect(rect, vignette)

        painter.end()

    def _draw_lab_equipment(self, painter: QPainter, w: float, h: float):
        """Draw laboratory equipment arrangement."""
        # Lab bench line
        bench_y = h * 0.75
        painter.setPen(QPen(QColor(255, 255, 255, 30), 2))
        painter.drawLine(int(w * 0.05), int(bench_y), int(w * 0.7), int(bench_y))

        # Beakers
        LabEquipment.draw_beaker(
            painter, w * 0.08, bench_y - h * 0.22, w * 0.08, h * 0.22,
            0.5 + 0.1 * math.sin(self.animation_phase),
            QColor(0, 212, 170, 80)
        )

        LabEquipment.draw_beaker(
            painter, w * 0.20, bench_y - h * 0.18, w * 0.06, h * 0.18,
            0.7,
            QColor(0, 180, 216, 80)
        )

        # Flask
        LabEquipment.draw_flask(
            painter, w * 0.30, bench_y - h * 0.28, w * 0.1, h * 0.28,
            0.35 + 0.05 * math.sin(self.animation_phase * 0.8),
            QColor(16, 185, 129, 70)
        )

        # Test tubes in rack
        for i in range(5):
            fill = 0.3 + 0.4 * (i / 4)
            color = QColor(
                int(0 + 16 * i),
                int(212 - 30 * i),
                int(170 + 15 * i),
                90
            )
            angle = -5 + 2 * i
            LabEquipment.draw_test_tube(
                painter,
                w * (0.45 + i * 0.04), bench_y - h * 0.2,
                w * 0.03, h * 0.18,
                fill, color, angle
            )

        # Large beaker (right side)
        LabEquipment.draw_beaker(
            painter, w * 0.55, bench_y - h * 0.3, w * 0.12, h * 0.3,
            0.6,
            QColor(0, 212, 170, 60)
        )


class GlassPanel(QFrame):
    """A frosted glass effect panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("glassPanel")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Rounded rectangle with glass effect
        path = QPainterPath()
        path.addRoundedRect(self.rect().adjusted(1, 1, -1, -1), 16, 16)

        # Glass background
        painter.fillPath(path, QColor(13, 33, 55, 220))

        # Border
        painter.setPen(QPen(QColor(0, 212, 170, 60), 1))
        painter.drawPath(path)

        # Top highlight
        highlight = QPainterPath()
        highlight.addRoundedRect(self.rect().adjusted(2, 2, -2, -self.height() + 40), 15, 15)

        highlight_gradient = QLinearGradient(0, 0, 0, 40)
        highlight_gradient.setColorAt(0, QColor(255, 255, 255, 15))
        highlight_gradient.setColorAt(1, QColor(255, 255, 255, 0))
        painter.fillPath(highlight, highlight_gradient)

        painter.end()


# =============================================================================
# DEPENDENCY MANAGEMENT
# =============================================================================

class DependencyChecker(QThread):
    """Background thread for checking and installing dependencies."""

    progress = Signal(int, str, str)  # percent, package_name, status
    finished = Signal(bool, list)  # success, missing_critical
    log = Signal(str)  # log message

    def __init__(self, dependencies: List[Dict]):
        super().__init__()
        self.dependencies = dependencies
        self.should_install = True

    def run(self):
        """Check all dependencies and install missing ones."""
        total = len(self.dependencies)
        missing_critical = []

        for i, dep in enumerate(self.dependencies):
            package = dep["package"]
            import_name = dep["import"]
            display = dep["display"]
            critical = dep["critical"]

            percent = int((i / total) * 100)
            self.progress.emit(percent, display, "Checking...")
            self.log.emit(f"Checking {display}...")

            # Check if installed
            try:
                importlib.import_module(import_name)
                self.progress.emit(percent, display, "Installed")
                self.log.emit(f"  ✓ {display} is installed")
            except ImportError:
                self.progress.emit(percent, display, "Installing...")
                self.log.emit(f"  ○ {display} not found, installing...")

                if self.should_install:
                    try:
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install",
                            package, "--quiet", "--disable-pip-version-check"
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                        # Verify installation
                        importlib.import_module(import_name)
                        self.progress.emit(percent, display, "Installed")
                        self.log.emit(f"  ✓ {display} installed successfully")
                    except Exception as e:
                        if critical:
                            missing_critical.append(display)
                            self.progress.emit(percent, display, "Failed")
                            self.log.emit(f"  ✗ Failed to install {display}: {e}")
                        else:
                            self.progress.emit(percent, display, "Skipped")
                            self.log.emit(f"  - {display} skipped (non-critical)")
                else:
                    if critical:
                        missing_critical.append(display)

            # Small delay for visual feedback
            self.msleep(100)

        self.progress.emit(100, "Complete", "Done")
        self.finished.emit(len(missing_critical) == 0, missing_critical)


# =============================================================================
# SPLASH SCREEN
# =============================================================================

class SplashScreen(QWidget):
    """Professional splash screen with dependency checking."""

    finished = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(700, 500)

        self._setup_ui()
        self._center_on_screen()

    def _setup_ui(self):
        """Setup the splash screen UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Main container
        container = QFrame()
        container.setObjectName("splashContainer")
        container.setStyleSheet("""
            #splashContainer {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0a1628, stop:0.5 #0d2137, stop:1 #0a1628);
                border-radius: 20px;
                border: 1px solid rgba(0, 212, 170, 0.3);
            }
        """)

        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(40, 40, 40, 40)
        container_layout.setSpacing(20)

        # Logo/Title area
        title_label = QLabel(APP_NAME)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 42px;
            font-weight: bold;
            color: #00d4aa;
            letter-spacing: 4px;
        """)

        tagline_label = QLabel(APP_TAGLINE)
        tagline_label.setAlignment(Qt.AlignCenter)
        tagline_label.setStyleSheet("""
            font-size: 14px;
            color: #94a3b8;
            letter-spacing: 2px;
        """)

        # Version
        version_label = QLabel(f"Version {APP_VERSION}")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet("""
            font-size: 11px;
            color: #64748b;
        """)

        # Spacer
        container_layout.addWidget(title_label)
        container_layout.addWidget(tagline_label)
        container_layout.addWidget(version_label)
        container_layout.addSpacing(30)

        # Status area
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 13px;
            color: #e8f4f8;
        """)

        self.package_label = QLabel("")
        self.package_label.setAlignment(Qt.AlignCenter)
        self.package_label.setStyleSheet("""
            font-size: 11px;
            color: #94a3b8;
        """)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
                border: none;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00d4aa, stop:1 #00b4d8);
                border-radius: 3px;
            }
        """)

        container_layout.addWidget(self.status_label)
        container_layout.addWidget(self.package_label)
        container_layout.addWidget(self.progress_bar)
        container_layout.addStretch()

        # Copyright
        copyright_label = QLabel(f"{APP_COPYRIGHT}\n{APP_PATENT}")
        copyright_label.setAlignment(Qt.AlignCenter)
        copyright_label.setStyleSheet("""
            font-size: 10px;
            color: #475569;
        """)
        container_layout.addWidget(copyright_label)

        layout.addWidget(container)

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 10)
        container.setGraphicsEffect(shadow)

    def _center_on_screen(self):
        """Center the splash screen on the primary display."""
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def start_dependency_check(self):
        """Start checking dependencies."""
        self.checker = DependencyChecker(DEPENDENCIES)
        self.checker.progress.connect(self._on_progress)
        self.checker.finished.connect(self._on_finished)
        self.checker.start()

    def _on_progress(self, percent: int, package: str, status: str):
        """Handle progress updates."""
        self.progress_bar.setValue(percent)
        self.status_label.setText(f"Verifying Dependencies...")
        self.package_label.setText(f"{package} - {status}")

    def _on_finished(self, success: bool, missing: List[str]):
        """Handle completion."""
        if success:
            self.status_label.setText("Ready to Launch")
            self.package_label.setText("All dependencies verified")
            self.progress_bar.setValue(100)

            # Delay before proceeding
            QTimer.singleShot(800, self.finished.emit)
        else:
            self.status_label.setText("Missing Critical Dependencies")
            self.status_label.setStyleSheet("font-size: 13px; color: #ef4444;")
            self.package_label.setText(", ".join(missing))


# =============================================================================
# MAIN LAUNCHER WINDOW
# =============================================================================

class LauncherWindow(QMainWindow):
    """Main launcher window with lab scene background."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} Launcher")
        self.setMinimumSize(1100, 750)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._drag_position = None
        self._setup_ui()
        self._center_on_screen()

    def _setup_ui(self):
        """Setup the main UI."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Container with border
        self.container = QFrame()
        self.container.setObjectName("mainContainer")
        self.container.setStyleSheet("""
            #mainContainer {
                background: transparent;
                border-radius: 16px;
                border: 1px solid rgba(0, 212, 170, 0.2);
            }
        """)

        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Lab scene background
        self.scene_widget = LabSceneWidget()
        self.scene_widget.setStyleSheet("border-radius: 16px;")

        # Overlay layout on scene
        overlay_layout = QVBoxLayout(self.scene_widget)
        overlay_layout.setContentsMargins(40, 20, 40, 40)
        overlay_layout.setSpacing(20)

        # Title bar
        title_bar = self._create_title_bar()
        overlay_layout.addWidget(title_bar)

        # Content area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(30)

        # Left panel - Info
        left_panel = self._create_info_panel()
        content_layout.addWidget(left_panel, stretch=2)

        # Right panel - Launch options
        right_panel = self._create_launch_panel()
        content_layout.addWidget(right_panel, stretch=1)

        overlay_layout.addLayout(content_layout)

        # Footer
        footer = self._create_footer()
        overlay_layout.addWidget(footer)

        container_layout.addWidget(self.scene_widget)
        main_layout.addWidget(self.container)

        # Shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(50)
        shadow.setColor(QColor(0, 0, 0, 200))
        shadow.setOffset(0, 15)
        self.container.setGraphicsEffect(shadow)

    def _create_title_bar(self) -> QWidget:
        """Create custom title bar."""
        title_bar = QWidget()
        title_bar.setFixedHeight(50)

        layout = QHBoxLayout(title_bar)
        layout.setContentsMargins(0, 0, 0, 0)

        # App title
        title = QLabel(APP_NAME)
        title.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            color: #00d4aa;
            letter-spacing: 3px;
        """)

        # Window controls
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        for btn_type, color, hover in [
            ("minimize", "#f59e0b", "#fbbf24"),
            ("maximize", "#10b981", "#34d399"),
            ("close", "#ef4444", "#f87171")
        ]:
            btn = QPushButton()
            btn.setFixedSize(14, 14)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {color};
                    border-radius: 7px;
                    border: none;
                }}
                QPushButton:hover {{
                    background: {hover};
                }}
            """)

            if btn_type == "minimize":
                btn.clicked.connect(self.showMinimized)
            elif btn_type == "maximize":
                btn.clicked.connect(self._toggle_maximize)
            else:
                btn.clicked.connect(self.close)

            controls_layout.addWidget(btn)

        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(controls)

        return title_bar

    def _create_info_panel(self) -> QWidget:
        """Create the information panel."""
        panel = GlassPanel()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Welcome header
        welcome = QLabel("Welcome to QuLab Infinite")
        welcome.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #ffffff;
        """)

        tagline = QLabel(APP_TAGLINE)
        tagline.setStyleSheet("""
            font-size: 14px;
            color: #00d4aa;
            letter-spacing: 1px;
        """)

        # Description
        desc = QLabel(
            "QuLab Infinite is a comprehensive scientific simulation platform providing "
            "calibrated, empirically-grounded accuracy for materials testing, quantum computing, "
            "chemistry, and physics experiments.\n\n"
            "With 80+ specialized laboratories and a database of 6.6 million materials, "
            "QuLab enables virtual experiments that produce dependable preliminary results "
            "before physical prototyping."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("""
            font-size: 13px;
            color: #94a3b8;
            line-height: 1.6;
        """)

        # Features grid
        features_label = QLabel("Key Capabilities")
        features_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #e8f4f8;
            margin-top: 10px;
        """)

        features_grid = QGridLayout()
        features_grid.setSpacing(15)

        features = [
            ("⚛️", "Quantum Computing", "30-qubit simulator"),
            ("🧪", "Chemistry Lab", "Molecular dynamics"),
            ("🔬", "Materials Science", "6.6M materials"),
            ("⚡", "Physics Engine", "Multi-domain simulation"),
            ("🏥", "Medical Labs", "Clinical-grade APIs"),
            ("🤖", "AI Integration", "ECH0 research system"),
            ("📐", "Virtual Product Dev", "PLM & collaboration"),
            ("🔧", "Design Optimizer", "Multi-objective optimization"),
            ("📊", "BOM Management", "Cost & variant analysis"),
        ]

        for i, (icon, title, subtitle) in enumerate(features):
            feature_widget = self._create_feature_item(icon, title, subtitle)
            features_grid.addWidget(feature_widget, i // 3, i % 3)

        layout.addWidget(welcome)
        layout.addWidget(tagline)
        layout.addSpacing(10)
        layout.addWidget(desc)
        layout.addSpacing(10)
        layout.addWidget(features_label)
        layout.addLayout(features_grid)
        layout.addStretch()

        return panel

    def _create_feature_item(self, icon: str, title: str, subtitle: str) -> QWidget:
        """Create a feature item widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)

        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 24px;")

        title_label = QLabel(title)
        title_label.setStyleSheet("""
            font-size: 12px;
            font-weight: bold;
            color: #e8f4f8;
        """)

        subtitle_label = QLabel(subtitle)
        subtitle_label.setStyleSheet("""
            font-size: 10px;
            color: #64748b;
        """)

        layout.addWidget(icon_label)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)

        return widget

    def _create_launch_panel(self) -> QWidget:
        """Create the launch options panel."""
        panel = GlassPanel()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(25, 30, 25, 30)
        layout.setSpacing(15)

        # Header
        header = QLabel("Launch Options")
        header.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #ffffff;
        """)

        layout.addWidget(header)
        layout.addSpacing(10)

        # Launch buttons
        buttons_data = [
            ("Launch GUI Application", "Full desktop interface with all labs", self._launch_gui, True),
            ("Run VPD Lab Demo", "Virtual Product Development showcase", self._launch_vpd_demo, False),
            ("Launch API Server", "REST API on localhost:8000", self._launch_api, False),
            ("Launch All Services", "GUI + API + Medical Labs", self._launch_all, False),
            ("Open Documentation", "View guides and API reference", self._open_docs, False),
        ]

        for text, desc, callback, primary in buttons_data:
            btn = self._create_launch_button(text, desc, callback, primary)
            layout.addWidget(btn)

        layout.addStretch()

        # System info
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            padding: 10px;
        """)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(12, 12, 12, 12)
        info_layout.setSpacing(6)

        info_header = QLabel("System Information")
        info_header.setStyleSheet("""
            font-size: 11px;
            font-weight: bold;
            color: #94a3b8;
        """)

        python_version = f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        info_text = QLabel(f"{python_version}\nPySide6 (Qt Framework)\nAll dependencies verified")
        info_text.setStyleSheet("""
            font-size: 10px;
            color: #64748b;
        """)

        info_layout.addWidget(info_header)
        info_layout.addWidget(info_text)

        layout.addWidget(info_frame)

        return panel

    def _create_launch_button(self, text: str, description: str,
                               callback, primary: bool = False) -> QPushButton:
        """Create a styled launch button."""
        btn = QPushButton()
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(70)

        # Custom layout inside button
        btn_layout = QVBoxLayout(btn)
        btn_layout.setContentsMargins(15, 10, 15, 10)
        btn_layout.setSpacing(4)

        title = QLabel(text)
        title.setStyleSheet(f"""
            font-size: 13px;
            font-weight: bold;
            color: {'#0a1628' if primary else '#ffffff'};
        """)
        title.setAttribute(Qt.WA_TransparentForMouseEvents)

        desc = QLabel(description)
        desc.setStyleSheet(f"""
            font-size: 10px;
            color: {'#334155' if primary else '#94a3b8'};
        """)
        desc.setAttribute(Qt.WA_TransparentForMouseEvents)

        btn_layout.addWidget(title)
        btn_layout.addWidget(desc)

        if primary:
            btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #00d4aa, stop:1 #00b4d8);
                    border-radius: 10px;
                    border: none;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #00e4ba, stop:1 #00c4e8);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #00c49a, stop:1 #00a4c8);
                }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 10px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
                QPushButton:hover {
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(0, 212, 170, 0.3);
                }
                QPushButton:pressed {
                    background: rgba(255, 255, 255, 0.05);
                }
            """)

        btn.clicked.connect(callback)
        return btn

    def _create_footer(self) -> QWidget:
        """Create the footer."""
        footer = QWidget()
        footer.setFixedHeight(40)

        layout = QHBoxLayout(footer)
        layout.setContentsMargins(0, 0, 0, 0)

        copyright_label = QLabel(f"{APP_COPYRIGHT}  •  {APP_PATENT}")
        copyright_label.setStyleSheet("""
            font-size: 10px;
            color: #475569;
        """)

        version_label = QLabel(f"Version {APP_VERSION}")
        version_label.setStyleSheet("""
            font-size: 10px;
            color: #475569;
        """)

        layout.addWidget(copyright_label)
        layout.addStretch()
        layout.addWidget(version_label)

        return footer

    def _center_on_screen(self):
        """Center window on screen."""
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def _toggle_maximize(self):
        """Toggle maximize state."""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    # Window dragging
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag_position is not None:
            self.move(event.globalPosition().toPoint() - self._drag_position)

    def mouseReleaseEvent(self, event):
        self._drag_position = None

    # Launch actions
    def _launch_gui(self):
        """Launch the main GUI application."""
        self.hide()
        try:
            from gui.main_window import run_gui
            run_gui()
        except Exception as e:
            self._show_error(f"Failed to launch GUI: {e}")
            self.show()

    def _launch_api(self):
        """Launch the API server."""
        try:
            subprocess.Popen([
                sys.executable,
                str(SCRIPT_DIR / "api" / "main.py")
            ], cwd=str(SCRIPT_DIR))
            self._show_info("API Server", "API server starting on http://localhost:8000")
        except Exception as e:
            self._show_error(f"Failed to launch API: {e}")

    def _launch_all(self):
        """Launch all services."""
        self._launch_api()
        QTimer.singleShot(1000, self._launch_gui)

    def _launch_vpd_demo(self):
        """Launch the Virtual Product Development Lab demo."""
        try:
            from virtual_product_lab import VirtualProductLab
            lab = VirtualProductLab({
                'project_name': 'QuLab Demo',
                'organization': 'Corporation of Light'
            })
            result = lab.run_demo()
            if result.get('success'):
                self._show_info(
                    "VPD Lab Demo Complete",
                    f"Successfully demonstrated Virtual Product Development Lab:\n\n"
                    f"• Product: {result['product']['name']}\n"
                    f"• Components: {result['product']['summary']['total_components']}\n"
                    f"• Total Cost: ${result['bom']['cost_rollup']['total_material_cost']:.2f}\n"
                    f"• Optimization: {result['optimization']['iterations']} iterations\n\n"
                    f"Check console for detailed output."
                )
            else:
                self._show_error("VPD Lab demo encountered an issue")
        except Exception as e:
            self._show_error(f"Failed to run VPD demo: {e}")

    def _open_docs(self):
        """Open documentation."""
        import webbrowser
        readme_path = SCRIPT_DIR / "README.md"
        if readme_path.exists():
            webbrowser.open(f"file://{readme_path}")
        else:
            self._show_info("Documentation", "Documentation available at LAUNCH_QULAB.md")

    def _show_error(self, message: str):
        """Show error message."""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Error", message)

    def _show_info(self, title: str, message: str):
        """Show info message."""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, title, message)


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

class QuLabLauncher:
    """Main application controller."""

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setApplicationName(APP_NAME)
        self.app.setApplicationVersion(APP_VERSION)

        # Set global font
        font = QFont("Segoe UI", 10)
        if not QFontDatabase.hasFamily("Segoe UI"):
            font = QFont("Arial", 10)
        self.app.setFont(font)

        # Global stylesheet
        self.app.setStyleSheet("""
            QToolTip {
                background: #0d2137;
                color: #e8f4f8;
                border: 1px solid rgba(0, 212, 170, 0.3);
                border-radius: 4px;
                padding: 5px;
            }
        """)

        self.splash = None
        self.launcher = None

    def run(self):
        """Run the launcher application."""
        # Show splash screen
        self.splash = SplashScreen()
        self.splash.finished.connect(self._on_splash_finished)
        self.splash.show()

        # Start dependency check
        QTimer.singleShot(500, self.splash.start_dependency_check)

        return self.app.exec()

    def _on_splash_finished(self):
        """Handle splash screen completion."""
        self.splash.close()

        # Show main launcher
        self.launcher = LauncherWindow()
        self.launcher.show()


def main():
    """Main entry point."""
    launcher = QuLabLauncher()
    sys.exit(launcher.run())


if __name__ == "__main__":
    main()

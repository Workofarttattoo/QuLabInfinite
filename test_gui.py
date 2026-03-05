#!/usr/bin/env python3
"""
Test script to isolate GUI loading issues.
"""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel

class SimpleMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuLabInfinite - Test")
        self.setGeometry(100, 100, 800, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        label = QLabel("Testing GUI components...")
        layout.addWidget(label)

        # Test basic imports
        try:
            from gui.physics_controls import PhysicsControls
            layout.addWidget(QLabel("✓ PhysicsControls imported"))
        except Exception as e:
            layout.addWidget(QLabel(f"✗ PhysicsControls failed: {e}"))

        try:
            from gui.pyvista_visualizer import PyVistaVisualizer
            layout.addWidget(QLabel("✓ PyVistaVisualizer imported"))
        except Exception as e:
            layout.addWidget(QLabel(f"✗ PyVistaVisualizer failed: {e}"))

        try:
            from gui.chemistry_controls import ChemistryControls
            layout.addWidget(QLabel("✓ ChemistryControls imported"))
        except Exception as e:
            layout.addWidget(QLabel(f"✗ ChemistryControls failed: {e}"))

def run_test():
    app = QApplication(sys.argv)
    window = SimpleMainWindow()
    window.show()
    print("GUI test window created successfully")
    sys.exit(app.exec())

if __name__ == "__main__":
    run_test()
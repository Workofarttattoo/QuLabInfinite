"""
Control widget for the Chemistry Lab GUI.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QGroupBox, QTextEdit
)
from chemistry_lab.datasets.registry import list_datasets, get_dataset

class ChemistryControls(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Dataset selection
        dataset_group = QGroupBox("Chemistry Datasets")
        dataset_layout = QVBoxLayout(dataset_group)

        dataset_layout.addWidget(QLabel("Select a dataset:"))
        self.dataset_selector = QComboBox()
        self.dataset_selector.addItems(list_datasets())
        dataset_layout.addWidget(self.dataset_selector)
        
        self.dataset_info = QTextEdit()
        self.dataset_info.setReadOnly(True)
        self.dataset_info.setFixedHeight(150)
        dataset_layout.addWidget(self.dataset_info)

        self.load_button = QPushButton("Load Dataset")
        dataset_layout.addWidget(self.load_button)
        
        layout.addWidget(dataset_group)

        # Connect signals
        self.dataset_selector.currentTextChanged.connect(self.update_dataset_info)
        
        # Initial info update
        self.update_dataset_info(self.dataset_selector.currentText())
        
        layout.addStretch()

    def update_dataset_info(self, dataset_name: str):
        """Update the info box with details of the selected dataset."""
        descriptor = get_dataset(dataset_name)
        if descriptor:
            info_text = f"Description:\n{descriptor.description}\n\n"
            info_text += f"Citation:\n{descriptor.citation}\n\n"
            info_text += f"Notes:\n{descriptor.notes}"
            self.dataset_info.setText(info_text)
        else:
            self.dataset_info.setText("")

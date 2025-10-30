"""
A widget to display a pandas DataFrame in a QTableView.
"""

from PySide6.QtWidgets import QTableView, QAbstractItemView
from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex
import pandas as pd

class DataFrameModel(QAbstractTableModel):
    """A model to interface a pandas DataFrame with a QTableView."""
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self._data = data

    def rowCount(self, parent=QModelIndex()):
        return self._data.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Vertical:
                return str(self._data.index[section])
        return None

class DataFrameViewer(QTableView):
    """A QTableView specialized for displaying pandas DataFrames."""
    def __init__(self, df: pd.DataFrame = pd.DataFrame()):
        super().__init__()
        self.setModel(DataFrameModel(df))
        self.setSortingEnabled(True)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)

    def setModel(self, df: pd.DataFrame):
        """Set a new DataFrame to be displayed."""
        model = DataFrameModel(df)
        super().setModel(model)

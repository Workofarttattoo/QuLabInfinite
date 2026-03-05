"""
Viewer for Stitch QuLab reference screens and exported HTML snippets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QComboBox,
    QSplitter,
    QPlainTextEdit,
    QScrollArea,
)


@dataclass
class ScreenAsset:
    pack: str
    name: str
    image_path: Path
    html_path: Optional[Path]


class StitchScreensWidget(QWidget):
    """Browse reference screens copied from stitch_qulabinfinite packs."""

    def __init__(self):
        super().__init__()
        self.assets_root = Path(__file__).resolve().parent / "assets" / "stitch_qulab"
        self.assets: List[ScreenAsset] = []
        self.current_asset: Optional[ScreenAsset] = None

        self._build_ui()
        self._load_assets()

    def _build_ui(self):
        root = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Pack:"))
        self.pack_filter = QComboBox()
        self.pack_filter.addItem("All Packs")
        top_bar.addWidget(self.pack_filter, stretch=1)
        self.count_label = QLabel("0 screens")
        top_bar.addWidget(self.count_label)
        root.addLayout(top_bar)

        splitter = QSplitter()

        self.screen_list = QListWidget()
        splitter.addWidget(self.screen_list)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_label = QLabel("No screen selected.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 320)
        self.image_label.setWordWrap(True)
        self.image_scroll.setWidget(self.image_label)
        right_layout.addWidget(self.image_scroll, stretch=2)

        right_layout.addWidget(QLabel("code.html"))
        self.code_view = QPlainTextEdit()
        self.code_view.setReadOnly(True)
        right_layout.addWidget(self.code_view, stretch=1)

        splitter.addWidget(right_panel)
        splitter.setSizes([380, 920])
        root.addWidget(splitter, stretch=1)

        self.pack_filter.currentTextChanged.connect(self._refresh_list)
        self.screen_list.currentItemChanged.connect(self._on_selection_changed)

    def _load_assets(self):
        if not self.assets_root.exists():
            self.count_label.setText("assets missing")
            self.image_label.setText(f"Assets directory not found:\n{self.assets_root}")
            return

        discovered: List[ScreenAsset] = []
        for image in sorted(self.assets_root.glob("**/screen.png")):
            relative = image.relative_to(self.assets_root)
            parts = relative.parts
            if len(parts) < 2:
                continue
            pack = parts[0]
            name = parts[-2].replace("_", " ")
            html_path = image.with_name("code.html")
            discovered.append(
                ScreenAsset(
                    pack=pack,
                    name=name,
                    image_path=image,
                    html_path=html_path if html_path.exists() else None,
                )
            )

        self.assets = discovered
        self._load_pack_filter()
        self._refresh_list()

    def _load_pack_filter(self):
        self.pack_filter.blockSignals(True)
        self.pack_filter.clear()
        self.pack_filter.addItem("All Packs")
        packs = sorted({asset.pack for asset in self.assets})
        for pack in packs:
            self.pack_filter.addItem(pack)
        self.pack_filter.blockSignals(False)

    def _refresh_list(self):
        selected_pack = self.pack_filter.currentText()
        self.screen_list.clear()

        filtered = [
            asset
            for asset in self.assets
            if selected_pack == "All Packs" or asset.pack == selected_pack
        ]

        for asset in filtered:
            item = QListWidgetItem(f"{asset.name} [{asset.pack}]")
            item.setData(Qt.UserRole, asset)
            self.screen_list.addItem(item)

        self.count_label.setText(f"{len(filtered)} screens")
        if filtered:
            self.screen_list.setCurrentRow(0)
        else:
            self.current_asset = None
            self.image_label.setText("No screens in this selection.")
            self.code_view.setPlainText("")

    def _on_selection_changed(self, current: Optional[QListWidgetItem], _previous):
        if current is None:
            self.current_asset = None
            return

        asset = current.data(Qt.UserRole)
        if not isinstance(asset, ScreenAsset):
            self.current_asset = None
            return

        self.current_asset = asset
        self._render_image(asset.image_path)
        self._render_code(asset)

    def _render_image(self, image_path: Path):
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.image_label.setText(f"Failed to load image:\n{image_path}")
            return

        viewport_size = self.image_scroll.viewport().size()
        target_width = max(320, viewport_size.width() - 24)
        scaled = pixmap.scaledToWidth(target_width, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())

    def _render_code(self, asset: ScreenAsset):
        if asset.html_path and asset.html_path.exists():
            try:
                code_text = asset.html_path.read_text(encoding="utf-8")
            except Exception as exc:
                self.code_view.setPlainText(f"Failed to read code.html:\n{exc}")
                return
            self.code_view.setPlainText(code_text)
            return
        self.code_view.setPlainText("No code.html file found for this screen.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_asset:
            self._render_image(self.current_asset.image_path)


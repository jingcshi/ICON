"""Left panel: searchable tree view of the taxonomy."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QTreeWidget, QTreeWidgetItem
)
from PyQt5.QtCore import pyqtSignal, Qt


class TreePanel(QWidget):
    node_selected = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._search = QLineEdit()
        self._search.setPlaceholderText('Search nodes…')
        self._search.textChanged.connect(self._filter)
        layout.addWidget(self._search)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.itemClicked.connect(self._on_click)
        layout.addWidget(self._tree)

        self._adapter = None
        self._items: dict = {}   # node_id -> QTreeWidgetItem

    def load(self, adapter) -> None:
        self._adapter = adapter
        self._rebuild()

    def _rebuild(self) -> None:
        self._tree.clear()
        self._items.clear()
        if self._adapter is None or not self._adapter.loaded:
            return
        taxo = self._adapter.taxo
        # Find roots (nodes with no parents in the graph)
        roots = [n for n in taxo.nodes if taxo.in_degree(n) == 0]
        for root in roots:
            self._add_subtree(root, None)
        self._tree.expandToDepth(1)

    def _add_subtree(self, node_id, parent_item) -> QTreeWidgetItem:
        taxo = self._adapter.taxo
        label = taxo.get_label(node_id)
        item = QTreeWidgetItem([f"{label}  [{node_id}]"])
        item.setData(0, Qt.UserRole, node_id)
        if parent_item is None:
            self._tree.addTopLevelItem(item)
        else:
            parent_item.addChild(item)
        self._items[node_id] = item
        for child in taxo.get_children(node_id):
            self._add_subtree(child, item)
        return item

    def _filter(self, text: str) -> None:
        text = text.strip().lower()
        for node_id, item in self._items.items():
            label = item.text(0).lower()
            visible = (not text) or text in label
            item.setHidden(not visible)
            if visible and text:
                parent = item.parent()
                while parent:
                    parent.setHidden(False)
                    parent = parent.parent()

    def _on_click(self, item: QTreeWidgetItem, _col: int) -> None:
        node_id = item.data(0, Qt.UserRole)
        if node_id is not None:
            self.node_selected.emit(node_id)

    def select_node(self, node_id) -> None:
        """Highlight the given node without re-emitting the signal."""
        item = self._items.get(node_id)
        if item:
            self._tree.blockSignals(True)
            self._tree.setCurrentItem(item)
            self._tree.scrollToItem(item)
            self._tree.blockSignals(False)

    def refresh(self) -> None:
        """Rebuild after structural edits."""
        self._rebuild()

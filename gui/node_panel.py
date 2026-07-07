"""Right panel: node properties and edit actions."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QHBoxLayout
)
from PyQt5.QtCore import pyqtSignal, Qt


class NodePanel(QWidget):
    rename_requested   = pyqtSignal(object)          # node_id
    add_child_requested = pyqtSignal(object)         # parent node_id
    delete_requested   = pyqtSignal(object)          # node_id
    add_edge_requested = pyqtSignal(object)          # child node_id
    remove_edge_requested = pyqtSignal(object, object)  # (child, parent)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(200)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # Properties form
        self._form = QFormLayout()
        self._lbl_id       = QLabel('—')
        self._lbl_label    = QLabel('—')
        self._lbl_depth    = QLabel('—')
        self._lbl_parents  = QLabel('—')
        self._lbl_parents.setWordWrap(True)
        self._lbl_children = QLabel('—')
        self._lbl_children.setWordWrap(True)
        for row, (key, widget) in enumerate([
            ('ID',       self._lbl_id),
            ('Label',    self._lbl_label),
            ('Depth',    self._lbl_depth),
            ('Parents',  self._lbl_parents),
            ('Children', self._lbl_children),
        ]):
            self._form.addRow(f'<b>{key}</b>', widget)
        layout.addLayout(self._form)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        # Action buttons
        self._btn_rename    = QPushButton('Rename')
        self._btn_add_child = QPushButton('Add Child')
        self._btn_delete    = QPushButton('Delete Node')
        self._btn_add_edge  = QPushButton('Add Parent Edge')
        self._btn_rm_edge   = QPushButton('Remove Parent Edge')

        for btn in (self._btn_rename, self._btn_add_child, self._btn_delete,
                    self._btn_add_edge, self._btn_rm_edge):
            btn.setEnabled(False)
            layout.addWidget(btn)

        layout.addStretch()

        self._node_id = None
        self._adapter = None

        self._btn_rename.clicked.connect(lambda: self.rename_requested.emit(self._node_id))
        self._btn_add_child.clicked.connect(lambda: self.add_child_requested.emit(self._node_id))
        self._btn_delete.clicked.connect(lambda: self.delete_requested.emit(self._node_id))
        self._btn_add_edge.clicked.connect(lambda: self.add_edge_requested.emit(self._node_id))
        self._btn_rm_edge.clicked.connect(self._request_remove_edge)

    def load(self, adapter) -> None:
        self._adapter = adapter

    def show_node(self, node_id) -> None:
        self._node_id = node_id
        if self._adapter is None or not self._adapter.loaded or node_id is None:
            self._clear()
            return

        taxo = self._adapter.taxo
        label    = taxo.get_label(node_id)
        depth    = taxo.get_depth(node_id)
        parents  = taxo.get_parents(node_id)
        children = taxo.get_children(node_id)

        self._lbl_id.setText(str(node_id))
        self._lbl_label.setText(label)
        self._lbl_depth.setText(str(depth))
        self._lbl_parents.setText(
            ', '.join(f"{taxo.get_label(p)} [{p}]" for p in parents) or '—'
        )
        self._lbl_children.setText(
            ', '.join(f"{taxo.get_label(c)} [{c}]" for c in children) or '—'
        )

        for btn in (self._btn_rename, self._btn_add_child, self._btn_delete,
                    self._btn_add_edge, self._btn_rm_edge):
            btn.setEnabled(True)

    def _clear(self) -> None:
        for lbl in (self._lbl_id, self._lbl_label, self._lbl_depth,
                    self._lbl_parents, self._lbl_children):
            lbl.setText('—')
        for btn in (self._btn_rename, self._btn_add_child, self._btn_delete,
                    self._btn_add_edge, self._btn_rm_edge):
            btn.setEnabled(False)

    def _request_remove_edge(self) -> None:
        if self._node_id is None or self._adapter is None:
            return
        parents = list(self._adapter.taxo.get_parents(self._node_id))
        if len(parents) == 1:
            self.remove_edge_requested.emit(self._node_id, parents[0])
        elif len(parents) > 1:
            # Let the main window pick via dialog
            self.remove_edge_requested.emit(self._node_id, None)

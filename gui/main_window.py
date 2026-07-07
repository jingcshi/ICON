"""Main application window."""

import os
from PyQt5.QtWidgets import (
    QMainWindow, QSplitter, QStatusBar, QAction, QFileDialog,
    QMessageBox, QApplication
)
from PyQt5.QtCore import Qt

from taxonomy_adapter import TaxonomyAdapter
from graph_canvas import GraphCanvas
from tree_panel import TreePanel
from node_panel import NodePanel
from dialogs import RenameDialog, AddChildDialog, ChooseNodeDialog

_RADIUS = 2   # neighbourhood radius for subgraph view


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ICON Taxonomy Viewer')
        self.resize(1200, 750)

        self._adapter = TaxonomyAdapter()
        self._selected_id = None

        # Panels
        self._tree   = TreePanel()
        self._canvas = GraphCanvas()
        self._panel  = NodePanel()

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._tree)
        splitter.addWidget(self._canvas)
        splitter.addWidget(self._panel)
        splitter.setSizes([220, 700, 250])
        self.setCentralWidget(splitter)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage('No taxonomy loaded')

        # Menu
        self._build_menu()

        # Wiring
        self._tree.node_selected.connect(self._on_node_selected)
        self._canvas.node_selected.connect(self._on_node_selected)
        self._panel.rename_requested.connect(self._do_rename)
        self._panel.add_child_requested.connect(self._do_add_child)
        self._panel.delete_requested.connect(self._do_delete)
        self._panel.add_edge_requested.connect(self._do_add_edge)
        self._panel.remove_edge_requested.connect(self._do_remove_edge)

    # ── Menu ──────────────────────────────────────────────────────────────────

    def _build_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu('&File')
        self._add_action(file_menu, '&Open…',    'Ctrl+O', self._open)
        self._add_action(file_menu, '&Save',      'Ctrl+S', self._save)
        self._add_action(file_menu, 'Save &As…', 'Ctrl+Shift+S', self._save_as)
        file_menu.addSeparator()
        self._add_action(file_menu, '&Quit', 'Ctrl+Q', self.close)

        edit_menu = mb.addMenu('&Edit')
        self._add_action(edit_menu, '&Undo', 'Ctrl+Z', self._undo)

        view_menu = mb.addMenu('&View')
        self._add_action(view_menu, 'Fit graph', 'Ctrl+F', self._fit_graph)

    def _add_action(self, menu, label, shortcut, slot):
        act = QAction(label, self)
        act.setShortcut(shortcut)
        act.triggered.connect(slot)
        menu.addAction(act)

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _open(self):
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Taxonomy', '',
            'Taxonomy files (*.json *.owl *.rdf *.ttl *.n3);;All files (*)'
        )
        if not path:
            return
        try:
            self._adapter.load(path)
        except Exception as e:
            QMessageBox.critical(self, 'Load error', str(e))
            return
        self._selected_id = None
        self._tree.load(self._adapter)
        self._panel.load(self._adapter)
        self._canvas.render(None)
        self._update_status()
        self._update_title()

    def _save(self):
        if not self._adapter.loaded:
            return
        if self._adapter.path is None:
            self._save_as()
            return
        try:
            self._adapter.save()
            self._update_title()
        except Exception as e:
            QMessageBox.critical(self, 'Save error', str(e))

    def _save_as(self):
        if not self._adapter.loaded:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Taxonomy As', '',
            'JSON (*.json);;All files (*)'
        )
        if not path:
            return
        try:
            self._adapter.save(path)
            self._update_title()
        except Exception as e:
            QMessageBox.critical(self, 'Save error', str(e))

    def _confirm_discard(self) -> bool:
        if not self._adapter.dirty:
            return True
        reply = QMessageBox.question(
            self, 'Unsaved changes',
            'There are unsaved changes. Discard and continue?',
            QMessageBox.Discard | QMessageBox.Cancel
        )
        return reply == QMessageBox.Discard

    # ── Selection ─────────────────────────────────────────────────────────────

    def _on_node_selected(self, node_id):
        if not self._adapter.loaded:
            return
        self._selected_id = node_id
        self._panel.show_node(node_id)
        self._tree.select_node(node_id)
        sub = self._adapter.neighborhood(node_id, radius=_RADIUS)
        self._canvas.render(sub, selected_id=node_id, adapter=self._adapter)
        self._update_status()

    def _fit_graph(self):
        self._canvas._ax.autoscale()
        self._canvas._canvas.draw()

    # ── Edit actions ──────────────────────────────────────────────────────────

    def _do_rename(self, node_id):
        current = self._adapter.taxo.get_label(node_id)
        dlg = RenameDialog(current, self)
        if dlg.exec_() != RenameDialog.Accepted or not dlg.label():
            return
        self._adapter.rename_node(node_id, dlg.label())
        self._refresh_after_edit(node_id)

    def _do_add_child(self, parent_id):
        taxo = self._adapter.taxo
        next_id = max(taxo.nodes) + 1 if taxo.nodes else 1
        dlg = AddChildDialog(next_id, self)
        if dlg.exec_() != AddChildDialog.Accepted or not dlg.label():
            return
        try:
            self._adapter.add_node(dlg.node_id(), dlg.label(), parent_id)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))
            return
        self._refresh_after_edit(parent_id)

    def _do_delete(self, node_id):
        reply = QMessageBox.warning(
            self, 'Delete node',
            f'Delete node [{node_id}] and all its edges?',
            QMessageBox.Yes | QMessageBox.Cancel
        )
        if reply != QMessageBox.Yes:
            return
        self._adapter.delete_node(node_id)
        self._selected_id = None
        self._tree.refresh()
        self._panel.show_node(None)
        self._canvas.render(None)
        self._update_status()
        self._update_title()

    def _do_add_edge(self, child_id):
        taxo = self._adapter.taxo
        candidates = [n for n in taxo.nodes if n != child_id]
        dlg = ChooseNodeDialog(
            'Add Parent Edge', 'Choose new parent:',
            candidates, taxo.get_label, self
        )
        if dlg.exec_() != ChooseNodeDialog.Accepted:
            return
        parent_id = dlg.chosen()
        try:
            self._adapter.add_edge(child_id, parent_id)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))
            return
        self._refresh_after_edit(child_id)

    def _do_remove_edge(self, child_id, parent_id):
        taxo = self._adapter.taxo
        if parent_id is None:
            # Multiple parents — let user pick
            parents = list(taxo.get_parents(child_id))
            dlg = ChooseNodeDialog(
                'Remove Parent Edge', 'Which parent edge to remove?',
                parents, taxo.get_label, self
            )
            if dlg.exec_() != ChooseNodeDialog.Accepted:
                return
            parent_id = dlg.chosen()
        try:
            self._adapter.remove_edge(child_id, parent_id)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))
            return
        self._refresh_after_edit(child_id)

    def _undo(self):
        desc = self._adapter.undo()
        if desc is None:
            self._status.showMessage('Nothing to undo', 2000)
            return
        self._tree.refresh()
        if self._selected_id is not None and self._selected_id in self._adapter.taxo.nodes:
            self._on_node_selected(self._selected_id)
        else:
            self._selected_id = None
            self._panel.show_node(None)
            self._canvas.render(None)
        self._update_status()
        self._update_title()
        self._status.showMessage(f'Undone: {desc}', 2000)

    def _refresh_after_edit(self, focus_id=None):
        self._tree.refresh()
        if focus_id is not None and focus_id in self._adapter.taxo.nodes:
            self._on_node_selected(focus_id)
        self._update_status()
        self._update_title()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_status(self):
        if not self._adapter.loaded:
            self._status.showMessage('No taxonomy loaded')
            return
        taxo = self._adapter.taxo
        n = taxo.number_of_nodes()
        e = taxo.number_of_edges()
        sel = f'  |  Selected: {self._selected_id}' if self._selected_id is not None else ''
        self._status.showMessage(f'Nodes: {n}   Edges: {e}{sel}')

    def _update_title(self):
        base = 'ICON Taxonomy Viewer'
        if self._adapter.loaded:
            fname = os.path.basename(self._adapter.path or 'untitled.json')
            dirty = '*' if self._adapter.dirty else ''
            self.setWindowTitle(f'{base} — {dirty}{fname}')
        else:
            self.setWindowTitle(base)

    def closeEvent(self, event):
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()

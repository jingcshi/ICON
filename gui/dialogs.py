"""Simple input dialogs for edit actions."""

from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox,
    QComboBox, QVBoxLayout, QLabel
)


class RenameDialog(QDialog):
    def __init__(self, current_label: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Rename Node')
        layout = QFormLayout(self)
        self._edit = QLineEdit(current_label)
        layout.addRow('New label:', self._edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def label(self) -> str:
        return self._edit.text().strip()


class AddChildDialog(QDialog):
    def __init__(self, next_id: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Add Child Node')
        layout = QFormLayout(self)
        self._id_edit    = QLineEdit(str(next_id))
        self._label_edit = QLineEdit()
        layout.addRow('Node ID:', self._id_edit)
        layout.addRow('Label:',   self._label_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def node_id(self):
        raw = self._id_edit.text().strip()
        try:
            return int(raw)
        except ValueError:
            return raw

    def label(self) -> str:
        return self._label_edit.text().strip()


class ChooseNodeDialog(QDialog):
    """Pick one node from a list (for edge source/target selection)."""

    def __init__(self, title: str, prompt: str, nodes: list, label_fn, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(prompt))
        self._combo = QComboBox()
        for n in nodes:
            self._combo.addItem(f"{label_fn(n)}  [{n}]", n)
        layout.addWidget(self._combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def chosen(self):
        return self._combo.currentData()

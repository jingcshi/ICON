#!/usr/bin/env python3
"""Entry point for the ICON Taxonomy Viewer GUI."""

import sys
import os

# Ensure src/ is on the path when running directly from gui/
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import QApplication
from main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('ICON Taxonomy Viewer')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

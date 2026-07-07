"""Dash application factory."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import dash
import dash_bootstrap_components as dbc

from layout import build_layout
import callbacks  # registers callbacks as a side-effect


def create_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title='ICON Taxonomy Viewer',
        suppress_callback_exceptions=True,
    )
    app.layout = build_layout()
    callbacks.register(app)
    return app

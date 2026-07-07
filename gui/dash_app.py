"""Dash application factory."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import dash

from layout import build_layout
import callbacks  # registers callbacks as a side-effect


def create_app() -> dash.Dash:
    # assets/ folder is served automatically by Dash (bootstrap.min.css lives there).
    # external_stylesheets is empty so no CDN requests are made — required because
    # the VM only has intranet access and CDN fetches block indefinitely.
    app = dash.Dash(
        __name__,
        external_stylesheets=[],
        assets_folder=os.path.join(os.path.dirname(__file__), 'assets'),
        title='ICON Taxonomy Viewer',
        suppress_callback_exceptions=True,
        serve_locally=True,
    )
    app.layout = build_layout()
    callbacks.register(app)
    return app

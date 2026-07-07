#!/usr/bin/env python3
"""Entry point for the ICON Taxonomy Viewer (Dash/web)."""

import sys
import os

# The VM injects /opt/clients/pykrylov/... into PYTHONPATH which ships a
# broken pydantic that shadows miniforge's. Remove those entries from sys.path
# before any third-party imports so miniforge's packages take precedence.
sys.path = [p for p in sys.path if '/opt/clients' not in p]
os.environ.pop('PYTHONPATH', None)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dash_app import create_app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('ICON_GUI_PORT', 8050))
    print(f"ICON Taxonomy Viewer running at http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)

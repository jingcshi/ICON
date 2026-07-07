"""Dash component layout for the ICON Taxonomy Viewer."""

import dash_cytoscape as cyto
import dash_bootstrap_components as dbc
from dash import html, dcc

cyto.load_extra_layouts()   # enables 'dagre' hierarchical layout


# ── Cytoscape stylesheet ──────────────────────────────────────────────────────

CYTO_STYLESHEET = [
    {
        'selector': 'node',
        'style': {
            'label': 'data(label)',
            'font-size': '11px',
            'text-valign': 'center',
            'text-halign': 'center',
            'background-color': '#5B9BD5',
            'color': '#fff',
            'width': 'label',
            'height': 'label',
            'padding': '6px',
            'shape': 'roundrectangle',
            'text-wrap': 'wrap',
            'text-max-width': '120px',
        },
    },
    {
        'selector': 'node[role = "selected"]',
        'style': {'background-color': '#E05C3A'},
    },
    {
        'selector': 'node[role = "ancestor"]',
        'style': {'background-color': '#A8D08D', 'color': '#000'},
    },
    {
        'selector': 'node[role = "child"]',
        'style': {'background-color': '#FFD966', 'color': '#000'},
    },
    {
        'selector': 'edge',
        'style': {
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#999',
            'line-color': '#999',
            'width': 1.5,
        },
    },
]


# ── Helper: sidebar section ───────────────────────────────────────────────────

def _section(title: str, *children) -> html.Div:
    return html.Div([
        html.H6(title, className='text-muted fw-bold mb-1 mt-2'),
        *children,
    ])


# ── Layout builder ────────────────────────────────────────────────────────────

def build_layout() -> html.Div:
    sidebar = dbc.Col([
        # ── File I/O ──
        _section('File'),
        dbc.ButtonGroup([
            dbc.Button('Open', id='btn-open', size='sm', color='primary', outline=True),
            dbc.Button('Save', id='btn-save', size='sm', color='primary', outline=True),
            dbc.Button('Save As', id='btn-save-as', size='sm', color='primary', outline=True),
        ], className='mb-2 w-100'),

        # hidden Upload component for file picking
        dcc.Upload(
            id='upload-taxonomy',
            children=html.Div(id='upload-hint', style={'display': 'none'}),
            style={'display': 'none'},
            accept='.json,.owl,.rdf,.ttl,.n3',
        ),
        # Download component for saving
        dcc.Download(id='download-taxonomy'),

        html.Hr(className='my-2'),

        # ── Search ──
        _section('Search'),
        dbc.Input(id='search-input', placeholder='Search nodes…', size='sm', debounce=True, className='mb-2'),

        # ── Tree ──
        _section('Taxonomy Tree'),
        html.Div(
            id='tree-container',
            style={
                'overflowY': 'auto',
                'maxHeight': '50vh',
                'fontSize': '13px',
                'border': '1px solid #dee2e6',
                'borderRadius': '4px',
                'padding': '4px',
            },
        ),
    ], width=3, style={'borderRight': '1px solid #dee2e6', 'paddingRight': '12px'})

    canvas = dbc.Col([
        cyto.Cytoscape(
            id='cyto-graph',
            elements=[],
            layout={'name': 'dagre', 'rankDir': 'TB', 'spacingFactor': 1.4},
            stylesheet=CYTO_STYLESHEET,
            style={'width': '100%', 'height': '75vh', 'border': '1px solid #dee2e6'},
            minZoom=0.1,
            maxZoom=4.0,
            responsive=True,
        ),
        # Legend
        html.Div([
            _legend_dot('#E05C3A', 'Selected'),
            _legend_dot('#A8D08D', 'Parent'),
            _legend_dot('#FFD966', 'Child'),
            _legend_dot('#5B9BD5', 'Context'),
        ], className='d-flex gap-3 mt-1', style={'fontSize': '12px'}),
    ], width=6)

    node_panel = dbc.Col([
        _section('Node Properties'),
        html.Div(id='node-info', children=_empty_info()),
        html.Hr(className='my-2'),
        _section('Edit'),
        dbc.Button('Rename', id='btn-rename', size='sm', color='secondary', outline=True, className='mb-1 w-100', disabled=True),
        dbc.Button('Add Child', id='btn-add-child', size='sm', color='success', outline=True, className='mb-1 w-100', disabled=True),
        dbc.Button('Delete Node', id='btn-delete', size='sm', color='danger', outline=True, className='mb-1 w-100', disabled=True),
        dbc.Button('Add Parent Edge', id='btn-add-edge', size='sm', color='secondary', outline=True, className='mb-1 w-100', disabled=True),
        dbc.Button('Remove Parent Edge', id='btn-rm-edge', size='sm', color='secondary', outline=True, className='mb-1 w-100', disabled=True),
        html.Hr(className='my-2'),
        dbc.Button('Undo', id='btn-undo', size='sm', color='warning', outline=True, className='w-100', disabled=True),
    ], width=3, style={'borderLeft': '1px solid #dee2e6', 'paddingLeft': '12px'})

    # ── Modals ────────────────────────────────────────────────────────────────
    modals = html.Div([
        _modal('modal-rename', 'Rename Node', [
            dbc.Label('New label'),
            dbc.Input(id='input-rename', type='text'),
        ]),
        _modal('modal-add-child', 'Add Child Node', [
            dbc.Label('Node ID'),
            dbc.Input(id='input-child-id', type='text', className='mb-2'),
            dbc.Label('Label'),
            dbc.Input(id='input-child-label', type='text'),
        ]),
        _modal('modal-add-edge', 'Add Parent Edge', [
            dbc.Label('Choose parent node'),
            dcc.Dropdown(id='dd-parent-choice', placeholder='Search…', clearable=False),
        ]),
        _modal('modal-rm-edge', 'Remove Parent Edge', [
            dbc.Label('Choose parent edge to remove'),
            dcc.Dropdown(id='dd-rm-parent-choice', placeholder='Search…', clearable=False),
        ]),
        _modal('modal-save-as', 'Save As', [
            dbc.Label('Filename (will be saved to the source directory)'),
            dbc.Input(id='input-save-as-name', type='text', placeholder='taxonomy.json'),
        ]),
        _confirm('confirm-delete', 'Delete this node and all its edges?'),
    ])

    # ── Status bar ────────────────────────────────────────────────────────────
    status_bar = dbc.Alert(
        id='status-bar',
        children='No taxonomy loaded',
        color='light',
        className='mb-0 py-1 px-3',
        style={'fontSize': '12px', 'borderRadius': '0'},
    )

    # ── Hidden stores ─────────────────────────────────────────────────────────
    stores = html.Div([
        dcc.Store(id='store-loaded', data=False),       # bool: taxonomy loaded
        dcc.Store(id='store-selected', data=None),      # currently selected node_id
        dcc.Store(id='store-dirty', data=False),        # unsaved changes
        dcc.Store(id='store-file-path', data=None),     # current file path
        dcc.Store(id='store-action-trigger', data=0),   # bump to re-render after edit
        dcc.Store(id='store-tree-expanded', data=[]),   # list of expanded node_ids in tree
        # Notification toast
        dbc.Toast(
            id='toast-notify',
            header='',
            children='',
            is_open=False,
            duration=3000,
            style={'position': 'fixed', 'bottom': '40px', 'right': '20px', 'zIndex': 9999},
        ),
    ])

    return html.Div([
        # Navbar / title
        dbc.Navbar(
            dbc.Container([
                dbc.NavbarBrand('ICON Taxonomy Viewer', className='fw-bold'),
                html.Span(id='nav-dirty', className='text-warning ms-2', style={'fontSize': '20px'}),
                html.Span(id='nav-filename', className='text-muted ms-2', style={'fontSize': '13px'}),
            ], fluid=True),
            color='dark', dark=True, className='py-1 mb-2',
        ),
        dbc.Container([
            dbc.Row([sidebar, canvas, node_panel], className='g-2'),
        ], fluid=True),
        status_bar,
        modals,
        stores,
    ])


# ── Small helpers ─────────────────────────────────────────────────────────────

def _legend_dot(color: str, label: str) -> html.Span:
    return html.Span([
        html.Span('●', style={'color': color, 'marginRight': '3px'}),
        label,
    ])


def _empty_info() -> list:
    return [html.P('No node selected', className='text-muted small')]


def _modal(modal_id: str, title: str, body_children: list) -> dbc.Modal:
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(title)),
        dbc.ModalBody(body_children),
        dbc.ModalFooter([
            dbc.Button('OK', id=f'{modal_id}-ok', color='primary', className='me-2'),
            dbc.Button('Cancel', id=f'{modal_id}-cancel', color='secondary'),
        ]),
    ], id=modal_id, is_open=False)


def _confirm(confirm_id: str, message: str) -> dbc.Modal:
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle('Confirm')),
        dbc.ModalBody(message),
        dbc.ModalFooter([
            dbc.Button('Yes', id=f'{confirm_id}-yes', color='danger', className='me-2'),
            dbc.Button('Cancel', id=f'{confirm_id}-cancel', color='secondary'),
        ]),
    ], id=confirm_id, is_open=False)

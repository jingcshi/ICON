"""All Dash callbacks for the ICON Taxonomy Viewer.

State management: TaxonomyAdapter is a module-level singleton (single-user local dev server).
Dash Stores carry lightweight signalling state (selected node id, dirty flag, etc.).
The adapter itself is never serialised into a Store — callbacks read/write it directly.
"""

import os
import base64
import tempfile
import json

import dash
from dash import Input, Output, State, callback_context, no_update, html, dcc
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from taxonomy_adapter import TaxonomyAdapter

_ADAPTER = TaxonomyAdapter()
_RADIUS = 2   # neighbourhood hops


# ── Registration entry point ──────────────────────────────────────────────────

def register(app):
    _register_file_callbacks(app)
    _register_selection_callbacks(app)
    _register_tree_callback(app)
    _register_graph_callback(app)
    _register_node_panel_callback(app)
    _register_edit_callbacks(app)
    _register_status_callback(app)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _node_info_children(node_id) -> list:
    taxo = _ADAPTER.taxo
    label    = taxo.get_label(node_id)
    depth    = taxo.get_depth(node_id)
    parents  = list(taxo.get_parents(node_id))
    children = list(taxo.get_children(node_id))

    def fmt(ids):
        if not ids:
            return '—'
        return ', '.join(f"{taxo.get_label(n)} [{n}]" for n in ids)

    return [
        html.Table([
            html.Tr([html.Th('ID'),       html.Td(str(node_id))]),
            html.Tr([html.Th('Label'),    html.Td(label)]),
            html.Tr([html.Th('Depth'),    html.Td(str(depth))]),
            html.Tr([html.Th('Parents'),  html.Td(fmt(parents),  style={'wordBreak': 'break-word'})]),
            html.Tr([html.Th('Children'), html.Td(fmt(children), style={'wordBreak': 'break-word'})]),
        ], className='table table-sm table-borderless', style={'fontSize': '12px'}),
    ]


def _build_cyto_elements(node_id):
    """Build Cytoscape elements for the neighbourhood subgraph."""
    sub = _ADAPTER.neighborhood(node_id, radius=_RADIUS)
    taxo = _ADAPTER.taxo

    ancestors = set(sub.predecessors(node_id)) if node_id in sub else set()
    children  = set(sub.successors(node_id))   if node_id in sub else set()

    nodes = []
    for n in sub.nodes:
        if n == node_id:
            role = 'selected'
        elif n in ancestors:
            role = 'ancestor'
        elif n in children:
            role = 'child'
        else:
            role = 'context'
        lbl = taxo.get_label(n)
        if len(lbl) > 20:
            lbl = lbl[:18] + '…'
        nodes.append({'data': {'id': str(n), 'label': lbl, 'role': role, 'node_id': n}})

    edges = [
        {'data': {'source': str(u), 'target': str(v)}}
        for u, v in sub.edges
    ]
    return nodes + edges


def _build_tree_items(adapter) -> list:
    taxo = adapter.taxo
    roots = [n for n in taxo.nodes if taxo.in_degree(n) == 0]

    def render_node(nid, depth=0):
        lbl = taxo.get_label(nid)
        children_ids = list(taxo.get_children(nid))
        toggle = html.Details([
            html.Summary(
                f"{'  ' * depth}{lbl} [{nid}]",
                id={'type': 'tree-node', 'index': nid},
                style={'cursor': 'pointer', 'userSelect': 'none',
                       'padding': '1px 2px', 'whiteSpace': 'nowrap'},
            ),
            *[render_node(c, depth + 1) for c in children_ids],
        ] if children_ids else [
            html.Summary(
                f"{'  ' * depth}{lbl} [{nid}]",
                id={'type': 'tree-node', 'index': nid},
                style={'cursor': 'pointer', 'userSelect': 'none',
                       'listStyle': 'none', 'padding': '1px 2px', 'whiteSpace': 'nowrap'},
            ),
        ], open=(depth == 0))
        return toggle

    return [render_node(r) for r in roots]


def _dropdown_options(node_ids, exclude=None):
    taxo = _ADAPTER.taxo
    opts = []
    for n in sorted(node_ids, key=lambda x: taxo.get_label(x)):
        if n == exclude:
            continue
        opts.append({'label': f"{taxo.get_label(n)}  [{n}]", 'value': n})
    return opts


# ── File I/O callbacks ────────────────────────────────────────────────────────

def _register_file_callbacks(app):

    # Open button → trigger the hidden Upload click via JS (handled in clientside)
    # Instead, we use the Upload component directly for file picking.
    # The Open button is wired to open a "use the Upload component" hint.
    # In practice users just drag-drop or click the Upload component area.
    # We expose Open as a button that opens an invisible Upload. We achieve this
    # with a clientside callback that simulates a click on the upload input.
    app.clientside_callback(
        """
        function(n) {
            if (n) {
                document.querySelector('#upload-taxonomy input[type=file]').click();
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('upload-taxonomy', 'style'),
        Input('btn-open', 'n_clicks'),
        prevent_initial_call=True,
    )

    @app.callback(
        Output('store-loaded', 'data'),
        Output('store-file-path', 'data'),
        Output('store-dirty', 'data'),
        Output('store-action-trigger', 'data'),
        Output('toast-notify', 'children'),
        Output('toast-notify', 'header'),
        Output('toast-notify', 'is_open'),
        Input('upload-taxonomy', 'contents'),
        State('upload-taxonomy', 'filename'),
        State('store-action-trigger', 'data'),
        prevent_initial_call=True,
    )
    def on_upload(contents, filename, trigger):
        if contents is None:
            raise PreventUpdate
        # Decode base64 upload content, write to a temp file, then load
        content_type, content_string = contents.split(',', 1)
        decoded = base64.b64decode(content_string)
        ext = os.path.splitext(filename)[1].lower() if filename else '.json'
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(decoded)
            tmp_path = f.name
        try:
            _ADAPTER.load(tmp_path)
            # Store original filename hint (we don't know the real path after upload)
            _ADAPTER._path = None   # will require Save As for first save
            _ADAPTER._upload_name = filename or 'taxonomy'
            return True, None, False, (trigger or 0) + 1, f"Loaded {filename}", 'Open', True
        except Exception as e:
            return no_update, no_update, no_update, no_update, str(e), 'Load Error', True
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @app.callback(
        Output('download-taxonomy', 'data'),
        Output('store-dirty', 'data', allow_duplicate=True),
        Output('toast-notify', 'children', allow_duplicate=True),
        Output('toast-notify', 'header', allow_duplicate=True),
        Output('toast-notify', 'is_open', allow_duplicate=True),
        Input('btn-save', 'n_clicks'),
        State('store-loaded', 'data'),
        prevent_initial_call=True,
    )
    def on_save(n, loaded):
        if not n or not loaded or not _ADAPTER.loaded:
            raise PreventUpdate
        try:
            content = _ADAPTER.to_json_string()
            fname = getattr(_ADAPTER, '_upload_name', None) or 'taxonomy'
            if not fname.endswith('.json'):
                fname += '.json'
            return dcc.send_string(content, fname), False, f"Saved {fname}", 'Save', True
        except Exception as e:
            return no_update, no_update, str(e), 'Save Error', True

    @app.callback(
        Output('modal-save-as', 'is_open'),
        Input('btn-save-as', 'n_clicks'),
        Input('modal-save-as-cancel', 'n_clicks'),
        Input('modal-save-as-ok', 'n_clicks'),
        State('store-loaded', 'data'),
        prevent_initial_call=True,
    )
    def toggle_save_as(open_n, cancel_n, ok_n, loaded):
        triggered = callback_context.triggered_id
        if triggered == 'btn-save-as' and loaded:
            return True
        return False

    @app.callback(
        Output('download-taxonomy', 'data', allow_duplicate=True),
        Output('store-dirty', 'data', allow_duplicate=True),
        Output('toast-notify', 'children', allow_duplicate=True),
        Output('toast-notify', 'header', allow_duplicate=True),
        Output('toast-notify', 'is_open', allow_duplicate=True),
        Input('modal-save-as-ok', 'n_clicks'),
        State('input-save-as-name', 'value'),
        State('store-loaded', 'data'),
        prevent_initial_call=True,
    )
    def do_save_as(n, fname, loaded):
        if not n or not loaded or not _ADAPTER.loaded:
            raise PreventUpdate
        fname = (fname or 'taxonomy').strip()
        if not fname.endswith('.json'):
            fname += '.json'
        try:
            content = _ADAPTER.to_json_string()
            _ADAPTER._upload_name = fname
            return dcc.send_string(content, fname), False, f"Downloaded {fname}", 'Save As', True
        except Exception as e:
            return no_update, no_update, str(e), 'Save Error', True


# ── Selection: tree click or cyto tap ────────────────────────────────────────

def _register_selection_callbacks(app):

    @app.callback(
        Output('store-selected', 'data'),
        Input({'type': 'tree-node', 'index': dash.ALL}, 'n_clicks'),
        Input('cyto-graph', 'tapNodeData'),
        State('store-loaded', 'data'),
        prevent_initial_call=True,
    )
    def on_select(tree_clicks, tap_data, loaded):
        if not loaded:
            raise PreventUpdate
        triggered = callback_context.triggered_id
        if isinstance(triggered, dict) and triggered.get('type') == 'tree-node':
            return triggered['index']
        if tap_data:
            return tap_data.get('node_id')
        raise PreventUpdate


# ── Tree panel ────────────────────────────────────────────────────────────────

def _register_tree_callback(app):

    @app.callback(
        Output('tree-container', 'children'),
        Input('store-action-trigger', 'data'),
        Input('store-loaded', 'data'),
        Input('search-input', 'value'),
        prevent_initial_call=False,
    )
    def update_tree(trigger, loaded, search):
        if not loaded or not _ADAPTER.loaded:
            return html.P('No taxonomy loaded', className='text-muted small p-2')

        if search and search.strip():
            return _filtered_tree(search.strip().lower())

        return _build_tree_items(_ADAPTER)

    def _filtered_tree(text: str) -> list:
        taxo = _ADAPTER.taxo
        matching = [
            n for n in taxo.nodes
            if text in taxo.get_label(n).lower() or text in str(n).lower()
        ]
        if not matching:
            return [html.P('No matches', className='text-muted small p-2')]
        return [
            html.Div(
                f"{taxo.get_label(n)}  [{n}]",
                id={'type': 'tree-node', 'index': n},
                style={'cursor': 'pointer', 'padding': '2px 4px', 'fontSize': '12px'},
            )
            for n in matching
        ]


# ── Graph canvas ──────────────────────────────────────────────────────────────

def _register_graph_callback(app):

    @app.callback(
        Output('cyto-graph', 'elements'),
        Output('cyto-graph', 'layout'),
        Input('store-selected', 'data'),
        Input('store-action-trigger', 'data'),
        State('store-loaded', 'data'),
        prevent_initial_call=False,
    )
    def update_graph(selected_id, trigger, loaded):
        if not loaded or not _ADAPTER.loaded or selected_id is None:
            return [], {'name': 'dagre', 'rankDir': 'TB'}
        if selected_id not in _ADAPTER.taxo.nodes:
            return [], {'name': 'dagre', 'rankDir': 'TB'}
        elements = _build_cyto_elements(selected_id)
        layout = {'name': 'dagre', 'rankDir': 'TB', 'spacingFactor': 1.4, 'animate': False}
        return elements, layout


# ── Node panel ────────────────────────────────────────────────────────────────

def _register_node_panel_callback(app):

    @app.callback(
        Output('node-info', 'children'),
        Output('btn-rename', 'disabled'),
        Output('btn-add-child', 'disabled'),
        Output('btn-delete', 'disabled'),
        Output('btn-add-edge', 'disabled'),
        Output('btn-rm-edge', 'disabled'),
        Output('btn-undo', 'disabled'),
        Input('store-selected', 'data'),
        Input('store-action-trigger', 'data'),
        State('store-loaded', 'data'),
        prevent_initial_call=False,
    )
    def update_panel(selected_id, trigger, loaded):
        from layout import _empty_info
        no_node = [_empty_info(), True, True, True, True, True, True]

        if not loaded or not _ADAPTER.loaded:
            return no_node

        has_undo = bool(_ADAPTER._undo_stack)
        if selected_id is None or selected_id not in _ADAPTER.taxo.nodes:
            return [_empty_info(), True, True, True, True, True, not has_undo]

        info = _node_info_children(selected_id)
        return [info, False, False, False, False, False, not has_undo]


# ── Edit callbacks ────────────────────────────────────────────────────────────

def _register_edit_callbacks(app):

    # ── Rename ──
    @app.callback(
        Output('modal-rename', 'is_open'),
        Output('input-rename', 'value'),
        Input('btn-rename', 'n_clicks'),
        Input('modal-rename-cancel', 'n_clicks'),
        Input('modal-rename-ok', 'n_clicks'),
        State('store-selected', 'data'),
        prevent_initial_call=True,
    )
    def toggle_rename(open_n, cancel_n, ok_n, selected_id):
        triggered = callback_context.triggered_id
        if triggered == 'btn-rename' and selected_id is not None and _ADAPTER.loaded:
            current = _ADAPTER.taxo.get_label(selected_id)
            return True, current
        return False, no_update

    @app.callback(
        Output('store-action-trigger', 'data', allow_duplicate=True),
        Output('store-dirty', 'data', allow_duplicate=True),
        Input('modal-rename-ok', 'n_clicks'),
        State('input-rename', 'value'),
        State('store-selected', 'data'),
        State('store-action-trigger', 'data'),
        prevent_initial_call=True,
    )
    def do_rename(n, label, selected_id, trigger):
        if not n or not label or selected_id is None:
            raise PreventUpdate
        _ADAPTER.rename_node(selected_id, label.strip())
        return (trigger or 0) + 1, True

    # ── Add child ──
    @app.callback(
        Output('modal-add-child', 'is_open'),
        Output('input-child-id', 'value'),
        Output('input-child-label', 'value'),
        Input('btn-add-child', 'n_clicks'),
        Input('modal-add-child-cancel', 'n_clicks'),
        Input('modal-add-child-ok', 'n_clicks'),
        State('store-selected', 'data'),
        prevent_initial_call=True,
    )
    def toggle_add_child(open_n, cancel_n, ok_n, selected_id):
        triggered = callback_context.triggered_id
        if triggered == 'btn-add-child' and selected_id is not None and _ADAPTER.loaded:
            next_id = max(_ADAPTER.taxo.nodes) + 1 if _ADAPTER.taxo.nodes else 1
            return True, str(next_id), ''
        return False, no_update, no_update

    @app.callback(
        Output('store-action-trigger', 'data', allow_duplicate=True),
        Output('store-dirty', 'data', allow_duplicate=True),
        Output('toast-notify', 'children', allow_duplicate=True),
        Output('toast-notify', 'header', allow_duplicate=True),
        Output('toast-notify', 'is_open', allow_duplicate=True),
        Input('modal-add-child-ok', 'n_clicks'),
        State('input-child-id', 'value'),
        State('input-child-label', 'value'),
        State('store-selected', 'data'),
        State('store-action-trigger', 'data'),
        prevent_initial_call=True,
    )
    def do_add_child(n, child_id_str, label, parent_id, trigger):
        if not n or not label or parent_id is None:
            raise PreventUpdate
        try:
            child_id = int(child_id_str) if child_id_str else max(_ADAPTER.taxo.nodes) + 1
        except ValueError:
            child_id = child_id_str
        try:
            _ADAPTER.add_node(child_id, label.strip(), parent_id)
            return (trigger or 0) + 1, True, no_update, no_update, False
        except Exception as e:
            return no_update, no_update, str(e), 'Error', True

    # ── Delete ──
    @app.callback(
        Output('confirm-delete', 'is_open'),
        Input('btn-delete', 'n_clicks'),
        Input('confirm-delete-cancel', 'n_clicks'),
        Input('confirm-delete-yes', 'n_clicks'),
        prevent_initial_call=True,
    )
    def toggle_delete(open_n, cancel_n, yes_n):
        triggered = callback_context.triggered_id
        if triggered == 'btn-delete':
            return True
        return False

    @app.callback(
        Output('store-action-trigger', 'data', allow_duplicate=True),
        Output('store-selected', 'data', allow_duplicate=True),
        Output('store-dirty', 'data', allow_duplicate=True),
        Input('confirm-delete-yes', 'n_clicks'),
        State('store-selected', 'data'),
        State('store-action-trigger', 'data'),
        prevent_initial_call=True,
    )
    def do_delete(n, selected_id, trigger):
        if not n or selected_id is None:
            raise PreventUpdate
        _ADAPTER.delete_node(selected_id)
        return (trigger or 0) + 1, None, True

    # ── Add parent edge ──
    @app.callback(
        Output('modal-add-edge', 'is_open'),
        Output('dd-parent-choice', 'options'),
        Input('btn-add-edge', 'n_clicks'),
        Input('modal-add-edge-cancel', 'n_clicks'),
        Input('modal-add-edge-ok', 'n_clicks'),
        State('store-selected', 'data'),
        prevent_initial_call=True,
    )
    def toggle_add_edge(open_n, cancel_n, ok_n, selected_id):
        triggered = callback_context.triggered_id
        if triggered == 'btn-add-edge' and selected_id is not None and _ADAPTER.loaded:
            opts = _dropdown_options(_ADAPTER.taxo.nodes, exclude=selected_id)
            return True, opts
        return False, no_update

    @app.callback(
        Output('store-action-trigger', 'data', allow_duplicate=True),
        Output('store-dirty', 'data', allow_duplicate=True),
        Output('toast-notify', 'children', allow_duplicate=True),
        Output('toast-notify', 'header', allow_duplicate=True),
        Output('toast-notify', 'is_open', allow_duplicate=True),
        Input('modal-add-edge-ok', 'n_clicks'),
        State('dd-parent-choice', 'value'),
        State('store-selected', 'data'),
        State('store-action-trigger', 'data'),
        prevent_initial_call=True,
    )
    def do_add_edge(n, parent_id, child_id, trigger):
        if not n or parent_id is None or child_id is None:
            raise PreventUpdate
        try:
            _ADAPTER.add_edge(child_id, parent_id)
            return (trigger or 0) + 1, True, no_update, no_update, False
        except Exception as e:
            return no_update, no_update, str(e), 'Error', True

    # ── Remove parent edge ──
    @app.callback(
        Output('modal-rm-edge', 'is_open'),
        Output('dd-rm-parent-choice', 'options'),
        Input('btn-rm-edge', 'n_clicks'),
        Input('modal-rm-edge-cancel', 'n_clicks'),
        Input('modal-rm-edge-ok', 'n_clicks'),
        State('store-selected', 'data'),
        prevent_initial_call=True,
    )
    def toggle_rm_edge(open_n, cancel_n, ok_n, selected_id):
        triggered = callback_context.triggered_id
        if triggered == 'btn-rm-edge' and selected_id is not None and _ADAPTER.loaded:
            parents = list(_ADAPTER.taxo.get_parents(selected_id))
            opts = _dropdown_options(parents)
            return True, opts
        return False, no_update

    @app.callback(
        Output('store-action-trigger', 'data', allow_duplicate=True),
        Output('store-dirty', 'data', allow_duplicate=True),
        Output('toast-notify', 'children', allow_duplicate=True),
        Output('toast-notify', 'header', allow_duplicate=True),
        Output('toast-notify', 'is_open', allow_duplicate=True),
        Input('modal-rm-edge-ok', 'n_clicks'),
        State('dd-rm-parent-choice', 'value'),
        State('store-selected', 'data'),
        State('store-action-trigger', 'data'),
        prevent_initial_call=True,
    )
    def do_rm_edge(n, parent_id, child_id, trigger):
        if not n or parent_id is None or child_id is None:
            raise PreventUpdate
        try:
            _ADAPTER.remove_edge(child_id, parent_id)
            return (trigger or 0) + 1, True, no_update, no_update, False
        except Exception as e:
            return no_update, no_update, str(e), 'Error', True

    # ── Undo ──
    @app.callback(
        Output('store-action-trigger', 'data', allow_duplicate=True),
        Output('store-dirty', 'data', allow_duplicate=True),
        Output('toast-notify', 'children', allow_duplicate=True),
        Output('toast-notify', 'header', allow_duplicate=True),
        Output('toast-notify', 'is_open', allow_duplicate=True),
        Input('btn-undo', 'n_clicks'),
        State('store-action-trigger', 'data'),
        prevent_initial_call=True,
    )
    def do_undo(n, trigger):
        if not n:
            raise PreventUpdate
        desc = _ADAPTER.undo()
        if desc is None:
            return no_update, no_update, 'Nothing to undo', 'Undo', True
        return (trigger or 0) + 1, bool(_ADAPTER._undo_stack), f"Undone: {desc}", 'Undo', True


# ── Status bar + navbar ───────────────────────────────────────────────────────

def _register_status_callback(app):

    @app.callback(
        Output('status-bar', 'children'),
        Output('nav-dirty', 'children'),
        Output('nav-filename', 'children'),
        Input('store-loaded', 'data'),
        Input('store-selected', 'data'),
        Input('store-dirty', 'data'),
        Input('store-action-trigger', 'data'),
        prevent_initial_call=False,
    )
    def update_status(loaded, selected_id, dirty, trigger):
        if not loaded or not _ADAPTER.loaded:
            return 'No taxonomy loaded', '', ''
        taxo = _ADAPTER.taxo
        n = taxo.number_of_nodes()
        e = taxo.number_of_edges()
        sel = f'   |   Selected: {selected_id}' if selected_id is not None else ''
        status = f'Nodes: {n}   Edges: {e}{sel}'
        dirty_marker = '●' if dirty else ''
        fname = getattr(_ADAPTER, '_upload_name', '') or ''
        return status, dirty_marker, fname



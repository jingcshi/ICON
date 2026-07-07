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
    _register_tree_callbacks(app)
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

    ancestors = set(sub.successors(node_id))   if node_id in sub else set()
    children  = set(sub.predecessors(node_id)) if node_id in sub else set()

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


def _path_key(path: tuple) -> str:
    """Stable string key for a root-to-node path (used in store-tree-expanded)."""
    return "|".join(str(n) for n in path)


def _build_tree_rows(adapter, expanded_keys: set, selected_id) -> list:
    """
    Render the tree as a flat list of rows.

    Each visible row corresponds to one occurrence of a node along a specific
    root-to-node path.  Expansion state is keyed by path so that a node
    appearing in multiple places (DAG) can be independently expanded.

    Highlight rules:
      - selected node: bold + red text at every occurrence
      - ancestors-of-selected (nodes above selected in any path): italic hint
        only at collapsed ancestors whose subtree contains selected
    """
    taxo = adapter.taxo
    roots = sorted(
        taxo.get_GCD([]),   # nodes with no parents (out_degree==0 in child→parent graph)
        key=lambda n: taxo.get_label(n),
    )

    # Pre-compute all ancestors of selected node for hinting
    ancestor_ids: set = set()
    if selected_id is not None and selected_id in taxo.nodes:
        frontier = {selected_id}
        while frontier:
            next_f = set()
            for n in frontier:
                for p in taxo.get_parents(n):
                    if p not in ancestor_ids:
                        ancestor_ids.add(p)
                        next_f.add(p)
            frontier = next_f

    rows = []

    def visit(nid, path: tuple, depth: int):
        path_k = _path_key(path)
        lbl = taxo.get_label(nid)
        children_ids = sorted(taxo.get_children(nid), key=lambda n: taxo.get_label(n))
        is_expanded = path_k in expanded_keys
        is_selected = (nid == selected_id)

        # Hint: this collapsed node is an ancestor of the selected node
        # and the selected node is not already visible through this subtree
        is_hint = (
            not is_selected
            and nid in ancestor_ids
            and not is_expanded
        )

        # Toggle button (▶ / ▼) or blank spacer for leaf
        if children_ids:
            toggle_btn = html.Span(
                '▼' if is_expanded else '▶',
                id={'type': 'tree-toggle', 'index': path_k},
                style={
                    'cursor': 'pointer',
                    'marginRight': '4px',
                    'fontSize': '9px',
                    'color': '#666',
                    'userSelect': 'none',
                    'display': 'inline-block',
                    'width': '12px',
                },
                title='Expand/collapse',
            )
        else:
            toggle_btn = html.Span(
                '·',
                style={
                    'marginRight': '4px',
                    'fontSize': '9px',
                    'color': '#ccc',
                    'display': 'inline-block',
                    'width': '12px',
                },
            )

        label_style = {
            'cursor': 'pointer',
            'userSelect': 'none',
            'whiteSpace': 'nowrap',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'maxWidth': '160px',
            'display': 'inline-block',
            'verticalAlign': 'middle',
        }
        if is_selected:
            label_style.update({'fontWeight': 'bold', 'color': '#E05C3A'})
        elif is_hint:
            label_style.update({'fontStyle': 'italic', 'color': '#888'})

        label_span = html.Span(
            f"{lbl}",
            id={'type': 'tree-node', 'index': nid},
            style=label_style,
            title=f"{lbl}  [{nid}]",
        )

        row = html.Div(
            [toggle_btn, label_span],
            style={
                'paddingLeft': f'{depth * 14}px',
                'paddingTop': '1px',
                'paddingBottom': '1px',
                'lineHeight': '1.4',
                'fontSize': '12px',
                'backgroundColor': '#fff3f0' if is_selected else 'transparent',
            },
        )
        rows.append(row)

        if is_expanded:
            for child in children_ids:
                visit(child, path + (child,), depth + 1)

    for root in roots:
        visit(root, (root,), 0)

    return rows


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
        Output('store-tree-expanded', 'data', allow_duplicate=True),
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
            _ADAPTER._path = None   # will require Save As for first save
            _ADAPTER._upload_name = filename or 'taxonomy'
            return True, None, False, (trigger or 0) + 1, [], f"Loaded {filename}", 'Open', True
        except Exception as e:
            return no_update, no_update, no_update, no_update, no_update, str(e), 'Load Error', True
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


# ── Tree panel + selection ────────────────────────────────────────────────────

def _register_tree_callbacks(app):

    # Cyto tap or tree-node label click → update selected store
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
            # n_clicks is None when component just appeared (tree re-render); not a real click
            if not callback_context.triggered or callback_context.triggered[0].get('value') is None:
                raise PreventUpdate
            return triggered['index']
        if tap_data:
            return tap_data.get('node_id')
        raise PreventUpdate

    # Toggle button click → flip path key in store-tree-expanded
    @app.callback(
        Output('store-tree-expanded', 'data'),
        Input({'type': 'tree-toggle', 'index': dash.ALL}, 'n_clicks'),
        State('store-tree-expanded', 'data'),
        prevent_initial_call=True,
    )
    def on_toggle(toggle_clicks, expanded):
        triggered = callback_context.triggered_id
        if not isinstance(triggered, dict) or triggered.get('type') != 'tree-toggle':
            raise PreventUpdate
        # n_clicks is None when the component just appeared (tree re-render); not a real click
        if not callback_context.triggered or callback_context.triggered[0].get('value') is None:
            raise PreventUpdate
        path_k = triggered['index']
        expanded = list(expanded or [])
        if path_k in expanded:
            expanded.remove(path_k)
        else:
            expanded.append(path_k)
        return expanded

    # Re-render tree rows whenever expansion, selection, or taxonomy changes
    @app.callback(
        Output('tree-container', 'children'),
        Input('store-tree-expanded', 'data'),
        Input('store-selected', 'data'),
        Input('store-action-trigger', 'data'),
        Input('store-loaded', 'data'),
        Input('search-input', 'value'),
        prevent_initial_call=False,
    )
    def update_tree(expanded, selected_id, trigger, loaded, search):
        if not loaded or not _ADAPTER.loaded:
            return html.P('No taxonomy loaded', className='text-muted small p-2')

        if search and search.strip():
            return _filtered_tree(search.strip().lower(), selected_id)

        expanded_keys = set(expanded or [])
        return _build_tree_rows(_ADAPTER, expanded_keys, selected_id)

    def _filtered_tree(text: str, selected_id) -> list:
        taxo = _ADAPTER.taxo
        matching = [
            n for n in taxo.nodes
            if text in taxo.get_label(n).lower() or text in str(n).lower()
        ]
        if not matching:
            return [html.P('No matches', className='text-muted small p-2')]
        rows = []
        for n in sorted(matching, key=lambda x: taxo.get_label(x)):
            lbl = taxo.get_label(n)
            is_sel = (n == selected_id)
            style = {
                'cursor': 'pointer', 'padding': '2px 4px', 'fontSize': '12px',
                'fontWeight': 'bold' if is_sel else 'normal',
                'color': '#E05C3A' if is_sel else 'inherit',
                'backgroundColor': '#fff3f0' if is_sel else 'transparent',
            }
            rows.append(html.Div(
                f"{lbl}  [{n}]",
                id={'type': 'tree-node', 'index': n},
                style=style,
            ))
        return rows


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



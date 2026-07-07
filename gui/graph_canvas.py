"""Matplotlib-based interactive graph canvas embedded in Qt."""

import networkx as nx
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSignal

# Colours
C_DEFAULT  = '#5B9BD5'
C_SELECTED = '#E05C3A'
C_ANCESTOR = '#A8D08D'
C_CHILD    = '#FFD966'


def _hierarchy_layout(G: nx.DiGraph, root=None):
    """Top-down hierarchical layout using graphviz if available, else Sugiyama-style."""
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        return graphviz_layout(G, prog='dot', args='-Grankdir=TB')
    except Exception:
        pass
    # Fallback: BFS layers from root
    if root is None:
        roots = [n for n in G.nodes if G.in_degree(n) == 0]
        root = roots[0] if roots else next(iter(G.nodes))
    layers = {}
    queue = [(root, 0)]
    visited = set()
    while queue:
        node, depth = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        layers[node] = depth
        for child in G.successors(node):
            if child not in visited:
                queue.append((child, depth + 1))
    for node in G.nodes:
        if node not in layers:
            layers[node] = 0
    layer_groups = {}
    for node, layer in layers.items():
        layer_groups.setdefault(layer, []).append(node)
    pos = {}
    for layer, nodes in layer_groups.items():
        for i, node in enumerate(nodes):
            pos[node] = (i - len(nodes) / 2, -layer)
    return pos


class GraphCanvas(QWidget):
    node_selected = pyqtSignal(object)  # emits node_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fig, self._ax = plt.subplots(figsize=(8, 6))
        self._fig.tight_layout(pad=1.0)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        self._node_positions = {}
        self._node_ids = []         # parallel to scatter offsets
        self._scatter = None
        self._selected_id = None
        self._canvas.mpl_connect('pick_event', self._on_pick)

    def render(self, subgraph: nx.DiGraph, selected_id=None, adapter=None):
        self._ax.clear()
        self._node_positions.clear()
        self._node_ids.clear()
        self._scatter = None
        self._selected_id = selected_id

        if subgraph is None or subgraph.number_of_nodes() == 0:
            self._canvas.draw()
            return

        pos = _hierarchy_layout(subgraph, root=selected_id)
        self._node_positions = pos

        # Classify nodes for colouring
        ancestors = set(subgraph.predecessors(selected_id)) if selected_id in subgraph else set()
        children  = set(subgraph.successors(selected_id))  if selected_id in subgraph else set()

        colours = []
        self._node_ids = list(subgraph.nodes)
        for n in self._node_ids:
            if n == selected_id:
                colours.append(C_SELECTED)
            elif n in ancestors:
                colours.append(C_ANCESTOR)
            elif n in children:
                colours.append(C_CHILD)
            else:
                colours.append(C_DEFAULT)

        # Draw edges
        nx.draw_networkx_edges(
            subgraph, pos, ax=self._ax,
            edge_color='#999999', arrows=True,
            arrowstyle='->', arrowsize=12,
            connectionstyle='arc3,rad=0.0',
            min_source_margin=15, min_target_margin=15,
        )

        # Draw nodes as a scatter (enables pick events)
        xs = [pos[n][0] for n in self._node_ids]
        ys = [pos[n][1] for n in self._node_ids]
        self._scatter = self._ax.scatter(
            xs, ys, s=300, c=colours, zorder=3,
            picker=True, pickradius=12,
        )

        # Labels
        labels = {}
        for n in self._node_ids:
            lbl = adapter.taxo.get_label(n) if adapter else str(n)
            labels[n] = lbl if len(lbl) <= 18 else lbl[:16] + '…'
        nx.draw_networkx_labels(subgraph, pos, labels=labels, ax=self._ax, font_size=7)

        # Legend
        legend = [
            mpatches.Patch(color=C_SELECTED, label='Selected'),
            mpatches.Patch(color=C_ANCESTOR, label='Parent'),
            mpatches.Patch(color=C_CHILD,    label='Child'),
            mpatches.Patch(color=C_DEFAULT,  label='Context'),
        ]
        self._ax.legend(handles=legend, loc='upper right', fontsize=7, framealpha=0.7)

        self._ax.axis('off')
        self._canvas.draw()

    def _on_pick(self, event):
        if event.artist is not self._scatter:
            return
        idx = event.ind[0]
        if idx < len(self._node_ids):
            self.node_selected.emit(self._node_ids[idx])

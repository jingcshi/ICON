import sys

import click

from icon.core.taxonomy import from_json, from_owl


@click.group()
def cli():
    """ICON — Implicit CONcept Insertion taxonomy enrichment toolkit."""
    pass

@cli.group()
def taxo():
    """Taxonomy management commands."""
    pass

@taxo.command("view")
@click.argument("file", type=click.Path(exists=True))
@click.option("--depth", default=3, show_default=True, help="Max depth for tree preview.")
def taxo_view(file, depth):
    """Print stats and a tree preview of a taxonomy file."""
    if file.endswith(".owl") or file.endswith(".rdf"):
        taxo = from_owl(file)
    else:
        taxo = from_json(file)

    n_nodes = taxo.number_of_nodes()
    n_edges = taxo.number_of_edges()
    leaves = len(taxo.get_LCA([]))
    roots = len(taxo.get_GCD([]))
    click.echo(f"Nodes : {n_nodes}")
    click.echo(f"Edges : {n_edges}")
    click.echo(f"Roots : {roots}")
    click.echo(f"Leaves: {leaves}")
    click.echo()

    def _print_tree(node, indent=0, visited=None):
        if visited is None:
            visited = set()
        if node in visited or indent > depth:
            return
        visited.add(node)
        label = taxo.get_label(node) or str(node)
        click.echo("  " * indent + label)
        for child in taxo.get_children(node):
            _print_tree(child, indent + 1, visited)

    click.echo(f"Tree preview (depth ≤ {depth}):")
    for root in taxo.get_GCD([]):
        _print_tree(root)

@taxo.command("convert")
@click.argument("src", type=click.Path(exists=True))
@click.argument("dst")
def taxo_convert(src, dst):
    """Convert a taxonomy between JSON and OWL formats."""
    if src.endswith(".owl") or src.endswith(".rdf"):
        taxo = from_owl(src)
    else:
        taxo = from_json(src)

    if dst.endswith(".json"):
        taxo.to_json(dst, indent=2)
        click.echo(f"Saved JSON taxonomy to {dst}")
    else:
        click.echo("Only JSON output is currently supported.", err=True)
        sys.exit(1)

@taxo.command("validate")
@click.argument("file", type=click.Path(exists=True))
def taxo_validate(file):
    """Validate taxonomy integrity: DAG, label completeness, orphans."""
    import networkx as nx

    if file.endswith(".owl") or file.endswith(".rdf"):
        taxo = from_owl(file)
    else:
        taxo = from_json(file)

    errors = []
    if not nx.is_directed_acyclic_graph(taxo):
        errors.append("FAIL: Taxonomy contains cycles.")

    missing_labels = [n for n, d in taxo.nodes(data=True) if not d.get("label")]
    if missing_labels:
        errors.append(f"FAIL: {len(missing_labels)} node(s) have no label: {missing_labels[:5]}...")

    orphans = [n for n in taxo.nodes() if n != 0 and taxo.in_degree(n) == 0 and taxo.out_degree(n) == 0]
    if orphans:
        errors.append(f"WARN: {len(orphans)} orphan node(s) found.")

    if errors:
        for e in errors:
            click.echo(e)
        sys.exit(1)
    else:
        click.echo("OK: Taxonomy is valid.")

@cli.command("enrich")
@click.argument("taxo_file", type=click.Path(exists=True))
@click.option("--mode", type=click.Choice(["auto", "semiauto", "manual"]), default="auto", show_default=True)
@click.option("--output", "-o", default=None, help="Output JSON path. Defaults to <taxo_file>.enriched.json")
def enrich(taxo_file, mode, output):
    """Run ICON enrichment on a taxonomy file.

    Models must be configured programmatically; this command is a scaffold.
    """
    click.echo("ICON enrich command requires model configuration via Python API.")
    click.echo("See demo.ipynb for a full example.")
    sys.exit(0)

if __name__ == "__main__":
    cli()

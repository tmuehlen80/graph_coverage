"""
Run the data-driven archetype discovery (SubgraphExtractor) over the
pre-built Argoverse component graphs and report the discovered patterns.

This is a faithful run of the existing src/subgraphs/SubgraphExtractor.py
algorithm; we only (a) feed it the already-built per-scene component graphs,
(b) disable the per-graph PNG dump, and (c) skip the decomposition step
(we only want the discovered pattern library + frequencies).
"""
import sys, glob, random, collections, argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import pickle
import networkx as nx
from subgraphs.SubgraphExtractor import SubgraphExtractor


def edge_type_multiset(g):
    c = collections.Counter(d.get("edge_type") for _, _, d in g.edges(data=True))
    return tuple(sorted(c.items()))


class GraphBundle:
    """Minimal stand-in for an ActorGraph: only needs `.actor_graphs` (dict t->graph)."""
    def __init__(self, graphs):
        self.actor_graphs = {i: g for i, g in enumerate(graphs)}
        self.actor_subgraphs = {}
    def __repr__(self):
        return f"<GraphBundle n={len(self.actor_graphs)}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="number of component graphs to sample")
    ap.add_argument("--min", type=int, default=2)
    ap.add_argument("--max", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    comp_dir = REPO / "argoverse_components_nx"
    files = sorted(glob.glob(str(comp_dir / "*.pkl")))
    print(f"Found {len(files)} component graphs in {comp_dir.name}")
    rng = random.Random(args.seed)
    sample = rng.sample(files, min(args.n, len(files)))
    graphs = [pickle.load(open(f, "rb")) for f in sample]
    # Convert MultiDiGraph -> DiGraph (collapse parallel edges; keep edge_type)
    di = []
    for g in graphs:
        h = nx.DiGraph()
        h.add_nodes_from(g.nodes(data=True))
        for u, v, d in g.edges(data=True):
            h.add_edge(u, v, **d)
        di.append(h)
    graphs = di

    size_dist = collections.Counter(g.number_of_nodes() for g in graphs)
    print("sampled node-count distribution:", dict(sorted(size_dist.items())))

    extractor = SubgraphExtractor(
        subgraph_selection_strategy="frequency",
        min_subgraph_size=args.min,
        max_subgraph_size=args.max,
    )
    # Disable the per-graph PNG dump and the (DiGraphMatcher) decomposition step.
    extractor._save_graph_visualization = lambda *a, **k: None
    extractor._decompose_graphs = lambda *a, **k: None

    bundle = GraphBundle(graphs)
    extractor.extract_subgraphs([bundle])

    lib = extractor.subgraph_library
    freq = extractor.subgraph_frequency
    print(f"\n=== Discovered {len(lib)} distinct archetype patterns "
          f"(size {args.min}-{args.max}, node attrs IGNORED by the algorithm) ===\n")
    ranked = sorted(lib.keys(), key=lambda i: freq.get(i, 0), reverse=True)
    total = sum(freq.values()) or 1
    print(f"{'rank':>4} {'freq':>8} {'share':>7}  {'n':>2} {'e':>2}  edge_type multiset")
    for r, sid in enumerate(ranked, 1):
        g = lib[sid]
        f = freq.get(sid, 0)
        ets = ", ".join(f"{et}×{cnt}" for et, cnt in edge_type_multiset(g))
        print(f"{r:>4} {f:>8} {100*f/total:>6.1f}% {g.number_of_nodes():>3} "
              f"{g.number_of_edges():>2}  {ets}")


if __name__ == "__main__":
    main()

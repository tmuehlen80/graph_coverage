#!/usr/bin/env python3
"""Render the after-discovery actor graph straight from the pickle, WITH axis
ticks, so a crop rectangle can be chosen by reading off coordinates.
No map/scenario data required."""
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt

script_dir = Path(__file__).parent.resolve()
repo_root = script_dir.parents[2]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))
SCENARIO_ID = "5758074f-be16-49da-8bb5-d43a0d8cd034"

with open(script_dir / f"{SCENARIO_ID}_graph_after_discovery.pkl", "rb") as f:
    G = pickle.load(f)

EDGE_STYLE = {
    "following_lead": ("blue", "following_lead"),
    "leading_vehicle": ("red", "leading_vehicle"),
    "neighbor_vehicle": ("forestgreen", "neighbor_vehicle"),
    "opposite_vehicle": ("orange", "opposite_vehicle"),
}

pos = {n: (G.nodes[n]["xyz"].x, G.nodes[n]["xyz"].y) for n in G.nodes()}

fig, ax = plt.subplots(figsize=(14, 14))

# edges
for u, v, d in G.edges(data=True):
    color = EDGE_STYLE.get(d.get("edge_type"), ("gray", "?"))[0]
    (x0, y0), (x1, y1) = pos[u], pos[v]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, linewidth=1.5, alpha=0.7))

# nodes + labels
for n, (x, y) in pos.items():
    ax.plot(x, y, "o", color="black", markersize=6, zorder=3)
    ax.text(x, y + 1.5, str(n), fontsize=8, ha="center", color="red")

handles = [plt.Line2D([0], [0], color=c, label=l) for c, l in EDGE_STYLE.values()]
ax.legend(handles=handles, loc="upper right", fontsize=10)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.5)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("After Relation Discovery (coordinates)")

out = script_dir / f"{SCENARIO_ID}_discovery_with_ticks.png"
fig.savefig(out, dpi=150, bbox_inches="tight")

xs = [p[0] for p in pos.values()]
ys = [p[1] for p in pos.values()]
print(f"actor x range: {min(xs):.1f} .. {max(xs):.1f}")
print(f"actor y range: {min(ys):.1f} .. {max(ys):.1f}")
print("saved:", out)

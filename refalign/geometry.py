"""Tree-space geometry — the claimed novel core of RefAlign.

Provides, over trees on a *shared leaf set*:

  * :func:`rf_distance` / :func:`weighted_rf_distance` — Robinson-Foulds;
  * :func:`patristic_distance_matrix` — path-length (tree-space) distance between
    leaves, used to pick the counterfactual "nearest successful neighbour";
  * :class:`TreePoint` — a tree embedded as a weighted split system, i.e. a point
    in Billera-Holmes-Vogtmann (BHV) tree space;
  * :func:`bhv_distance` — a BHV geodesic distance via the cone-path decomposed
    over connected components of the split-incompatibility graph (this is the
    exact Owen-Provan geodesic when each incompatibility component is resolved by
    a single cone segment, and a valid upper bound otherwise);
  * :func:`geodesic_interpolate` — a point at parameter ``t`` along that path;
  * :func:`frechet_mean` / :func:`frechet_variance` — the Sturm mean and variance
    of a set of trees, used to characterise the "healthy centroid" of a run
    population and to summarise failure-mode distributions.

The cone-path construction and the shared-leaf-set requirement are exactly the
methodological wrinkles flagged in the research grounding; they are documented
here so the approximation is explicit rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .trees import Tree, Node


# ---------------------------------------------------------------------------
# Robinson-Foulds
# ---------------------------------------------------------------------------
def rf_distance(t1: Tree, t2: Tree) -> int:
    """Unweighted (symmetric-difference) Robinson-Foulds distance."""
    return len(t1.bipartitions() ^ t2.bipartitions())


def weighted_rf_distance(t1: Tree, t2: Tree) -> float:
    """Branch-length-weighted RF (sum of |length differences| over all splits)."""
    e1 = dict(t1.edges())
    e2 = dict(t2.edges())
    keys = set(e1) | set(e2)
    return float(sum(abs(e1.get(k, 0.0) - e2.get(k, 0.0)) for k in keys))


# ---------------------------------------------------------------------------
# Patristic (path-length) distance on a single tree
# ---------------------------------------------------------------------------
def patristic_distance_matrix(tree: Tree) -> Tuple[List[str], np.ndarray]:
    """All-pairs leaf distances = summed branch lengths along the connecting path."""
    leaves = tree.leaves()
    idx = {lbl: i for i, lbl in enumerate(leaves)}
    n = len(leaves)
    D = np.zeros((n, n))

    # BFS from each leaf over the undirected tree (parent + children edges)
    def neighbours(node: Node):
        out = []
        if node.parent is not None:
            out.append((node.parent, node.length))
        for c in node.children:
            out.append((c, c.length))
        return out

    # locate leaf nodes
    leaf_nodes: Dict[str, Node] = {}
    stack = [tree.root]
    while stack:
        nd = stack.pop()
        if nd.is_leaf and nd.name is not None:
            leaf_nodes[nd.name] = nd
        stack.extend(nd.children)

    for lbl, src in leaf_nodes.items():
        # Dijkstra-free BFS: tree has unique paths, so a stack accumulating
        # distance suffices.
        seen = {id(src)}
        frontier = [(src, 0.0)]
        while frontier:
            node, dist = frontier.pop()
            if node.is_leaf and node.name is not None and node.name != lbl:
                D[idx[lbl], idx[node.name]] = dist
            for nb, w in neighbours(node):
                if id(nb) not in seen:
                    seen.add(id(nb))
                    frontier.append((nb, dist + max(0.0, w)))
    # symmetrise (numerical safety)
    D = 0.5 * (D + D.T)
    return leaves, D


# ---------------------------------------------------------------------------
# BHV tree space
# ---------------------------------------------------------------------------
@dataclass
class TreePoint:
    """A tree as a point in BHV space: canonical split -> positive branch length."""

    splits: Dict[FrozenSet[str], float]
    leaves: FrozenSet[str]

    @classmethod
    def from_tree(cls, tree: Tree) -> "TreePoint":
        leaves = frozenset(tree.leaves())
        splits = {s: max(0.0, L) for s, L in tree.edges() if L > 0.0}
        return cls(splits=splits, leaves=leaves)

    def norm(self) -> float:
        return float(np.sqrt(sum(v * v for v in self.splits.values())))


def _other_side(side: FrozenSet[str], leaves: FrozenSet[str]) -> FrozenSet[str]:
    return leaves - side


def splits_compatible(a: FrozenSet[str], b: FrozenSet[str],
                      leaves: FrozenSet[str]) -> bool:
    """Two bipartitions are compatible iff one of the four quadrant intersections
    is empty (equivalently, they can co-exist in a single tree)."""
    a2 = _other_side(a, leaves)
    b2 = _other_side(b, leaves)
    return not (a & b) or not (a & b2) or not (a2 & b) or not (a2 & b2)


def _incompatibility_components(p_only: List[FrozenSet[str]],
                                q_only: List[FrozenSet[str]],
                                leaves: FrozenSet[str]
                                ) -> List[Tuple[List[FrozenSet[str]], List[FrozenSet[str]]]]:
    """Connected components of the p/q split-incompatibility graph.

    Returns a list of (p_splits_in_component, q_splits_in_component).
    """
    # union-find over nodes = p_only (0..) and q_only (offset)
    nodes = [("p", s) for s in p_only] + [("q", s) for s in q_only]
    parent = list(range(len(nodes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    np_ = len(p_only)
    for i, ps in enumerate(p_only):
        for j, qs in enumerate(q_only):
            if not splits_compatible(ps, qs, leaves):
                union(i, np_ + j)

    comps: Dict[int, Tuple[List[FrozenSet[str]], List[FrozenSet[str]]]] = {}
    for i, (side, s) in enumerate(nodes):
        r = find(i)
        pc, qc = comps.setdefault(r, ([], []))
        (pc if side == "p" else qc).append(s)
    return list(comps.values())


def bhv_distance(p: TreePoint, q: TreePoint) -> float:
    """BHV geodesic distance (component-decomposed cone path)."""
    if p.leaves != q.leaves:
        raise ValueError("BHV distance requires a shared leaf set")
    leaves = p.leaves
    shared = set(p.splits) & set(q.splits)
    sq = 0.0
    for s in shared:
        sq += (p.splits[s] - q.splits[s]) ** 2

    p_only = [s for s in p.splits if s not in shared]
    q_only = [s for s in q.splits if s not in shared]
    for pc, qc in _incompatibility_components(p_only, q_only, leaves):
        np_norm = np.sqrt(sum(p.splits[s] ** 2 for s in pc))
        nq_norm = np.sqrt(sum(q.splits[s] ** 2 for s in qc))
        sq += (np_norm + nq_norm) ** 2
    return float(np.sqrt(sq))


def geodesic_interpolate(p: TreePoint, q: TreePoint, t: float) -> TreePoint:
    """Point at parameter ``t`` in [0,1] along the cone-path geodesic p->q."""
    if p.leaves != q.leaves:
        raise ValueError("geodesic requires a shared leaf set")
    t = float(min(1.0, max(0.0, t)))
    leaves = p.leaves
    out: Dict[FrozenSet[str], float] = {}
    shared = set(p.splits) & set(q.splits)
    for s in shared:
        out[s] = (1 - t) * p.splits[s] + t * q.splits[s]

    p_only = [s for s in p.splits if s not in shared]
    q_only = [s for s in q.splits if s not in shared]
    for pc, qc in _incompatibility_components(p_only, q_only, leaves):
        np_norm = np.sqrt(sum(p.splits[s] ** 2 for s in pc))
        nq_norm = np.sqrt(sum(q.splits[s] ** 2 for s in qc))
        denom = np_norm + nq_norm
        if denom <= 0:
            continue
        t_star = np_norm / denom
        if t < t_star:
            scale = (1.0 - t / t_star) if t_star > 0 else 0.0
            for s in pc:
                v = p.splits[s] * scale
                if v > 1e-12:
                    out[s] = v
        else:
            scale = (t - t_star) / (1.0 - t_star) if t_star < 1 else 1.0
            for s in qc:
                v = q.splits[s] * scale
                if v > 1e-12:
                    out[s] = v
    return TreePoint(splits=out, leaves=leaves)


def frechet_mean(points: List[TreePoint], passes: int = 40) -> TreePoint:
    """Sturm's proximal-point estimate of the Frechet (barycentre) mean.

    x_{k+1} = geodesic(x_k, sample_k, 1/(k+2)); deterministic cyclic schedule.
    """
    pts = [p for p in points if p.splits or True]
    if not pts:
        raise ValueError("no points")
    if len(pts) == 1:
        return pts[0]
    mean = pts[0]
    k = 0
    for _ in range(passes):
        for p in pts:
            step = 1.0 / (k + 2)
            mean = geodesic_interpolate(mean, p, step)
            k += 1
    return mean


def frechet_variance(points: List[TreePoint], mean: Optional[TreePoint] = None) -> float:
    """Frechet variance = mean squared BHV distance from the barycentre."""
    if not points:
        return 0.0
    if mean is None:
        mean = frechet_mean(points)
    return float(np.mean([bhv_distance(mean, p) ** 2 for p in points]))

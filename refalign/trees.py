"""Phylogenetic tree inference from a distance matrix.

Pure-Python neighbor-joining (Saitou & Nei) and UPGMA, a minimal ``Tree`` /
``Node`` model with Newick export, and bipartition (split) extraction used by
the tree-space geometry module for Robinson-Foulds / BHV distances.

Distance matrices are built either from an MSA (column-wise alignment distance)
or directly from pairwise Needleman-Wunsch costs (the "no-MSA" ablation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from .encode import Alphabet, GAP
from .msa import MultipleAlignment, pairwise_cost


# ---------------------------------------------------------------------------
class Node:
    """A tree node. Leaves carry a ``name``; internal nodes have ``children``."""

    __slots__ = ("name", "children", "parent", "length")

    def __init__(self, name: Optional[str] = None):
        self.name: Optional[str] = name
        self.children: List["Node"] = []
        self.parent: Optional["Node"] = None
        self.length: float = 0.0     # branch length to parent

    def add_child(self, child: "Node", length: float) -> None:
        child.parent = self
        child.length = float(length)
        self.children.append(child)

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass
class Tree:
    root: Node
    rooted: bool = False

    def leaves(self) -> List[str]:
        out: List[str] = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n.is_leaf and n.name is not None:
                out.append(n.name)
            stack.extend(n.children)
        return sorted(out)

    def _descendant_leaves(self) -> Dict[int, FrozenSet[str]]:
        """Map id(node) -> frozenset of leaf labels below it (post-order)."""
        memo: Dict[int, FrozenSet[str]] = {}

        def visit(n: Node) -> FrozenSet[str]:
            if n.is_leaf:
                s = frozenset([n.name]) if n.name is not None else frozenset()
            else:
                acc: Set[str] = set()
                for c in n.children:
                    acc |= visit(c)
                s = frozenset(acc)
            memo[id(n)] = s
            return s

        visit(self.root)
        return memo

    def bipartitions(self) -> Set[FrozenSet[str]]:
        """Non-trivial splits, canonicalised as the side excluding a fixed leaf.

        Two unrooted trees are compared by these split sets (Robinson-Foulds).
        """
        all_leaves = frozenset(self.leaves())
        n = len(all_leaves)
        if n < 4:
            return set()
        ref = min(all_leaves)  # fixed reference leaf for canonicalisation
        memo = self._descendant_leaves()
        splits: Set[FrozenSet[str]] = set()
        for node_id, clade in memo.items():
            k = len(clade)
            if k <= 1 or k >= n - 1:
                continue  # trivial split
            side = clade if ref not in clade else (all_leaves - clade)
            splits.add(frozenset(side))
        return splits

    def edges(self) -> List[Tuple[FrozenSet[str], float]]:
        """(canonical-split, branch-length) for every non-trivial internal edge.

        Used to embed the tree as a point in BHV split-space.
        """
        all_leaves = frozenset(self.leaves())
        n = len(all_leaves)
        ref = min(all_leaves) if all_leaves else None
        memo = self._descendant_leaves()
        out: List[Tuple[FrozenSet[str], float]] = []
        seen: Set[FrozenSet[str]] = set()

        def visit(node: Node) -> None:
            for c in node.children:
                clade = memo[id(c)]
                k = len(clade)
                if 1 < k < n - 1:
                    side = clade if ref not in clade else (all_leaves - clade)
                    fs = frozenset(side)
                    if fs not in seen:
                        seen.add(fs)
                        out.append((fs, max(0.0, c.length)))
                visit(c)

        visit(self.root)
        return out

    def newick(self) -> str:
        def fmt(n: Node) -> str:
            if n.is_leaf:
                return f"{n.name}:{n.length:.6g}"
            inner = ",".join(fmt(c) for c in n.children)
            return f"({inner}):{n.length:.6g}"

        # drop the root's own branch length
        def fmt_root(n: Node) -> str:
            if n.is_leaf:
                return n.name or ""
            return "(" + ",".join(fmt(c) for c in n.children) + ")"

        return fmt_root(self.root) + ";"


# ---------------------------------------------------------------------------
def distance_from_msa(msa: MultipleAlignment, metric: str = "align") -> Tuple[List[str], np.ndarray]:
    """Distance matrix from an alignment.

    metric='align' : per-column cost averaged over columns (indels count);
    metric='p'     : classic p-distance with pairwise gap deletion.
    """
    labels = list(msa.labels)
    M = msa.matrix
    n = len(labels)
    D = np.zeros((n, n))
    alpha = msa.alphabet
    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = M[i], M[j]
            if metric == "p":
                both = (ri != GAP) & (rj != GAP)
                comp = int(both.sum())
                if comp == 0:
                    d = 1.0
                else:
                    d = float(np.sum(ri[both] != rj[both])) / comp
            elif metric == "align":
                total = 0.0
                cols = 0
                for a, b in zip(ri.tolist(), rj.tolist()):
                    if a == GAP and b == GAP:
                        continue
                    cols += 1
                    if a == GAP or b == GAP:
                        total += 1.0
                    else:
                        total += alpha.substitution_cost(a, b)
                d = total / cols if cols else 0.0
            else:
                raise ValueError(f"unknown metric {metric!r}")
            D[i, j] = D[j, i] = d
    return labels, D


def distance_pairwise(labels: List[str], seqs: List[np.ndarray], alphabet: Alphabet,
                      gap: float = 1.0, normalize: bool = True
                      ) -> Tuple[List[str], np.ndarray]:
    """Distance matrix from raw pairwise NW costs (no MSA)."""
    n = len(seqs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            c = pairwise_cost(seqs[i], seqs[j], alphabet, gap=gap)
            if normalize:
                denom = max(len(seqs[i]), len(seqs[j]), 1)
                c = c / denom
            D[i, j] = D[j, i] = c
    return list(labels), D


# ---------------------------------------------------------------------------
def upgma(labels: List[str], D: np.ndarray) -> Tree:
    """UPGMA (rooted, ultrametric)."""
    n = len(labels)
    if n == 0:
        return Tree(Node(), rooted=True)
    if n == 1:
        return Tree(Node(labels[0]), rooted=True)

    dist = D.astype(float).copy()
    nodes: List[Optional[Node]] = [Node(lbl) for lbl in labels]
    sizes = [1] * n
    heights = [0.0] * n
    active = list(range(n))

    while len(active) > 1:
        # closest pair among active
        best = None
        bi = bj = -1
        for a_idx in range(len(active)):
            for b_idx in range(a_idx + 1, len(active)):
                i, j = active[a_idx], active[b_idx]
                if best is None or dist[i, j] < best:
                    best, bi, bj = dist[i, j], i, j
        h = best / 2.0
        parent = Node()
        parent.add_child(nodes[bi], max(0.0, h - heights[bi]))
        parent.add_child(nodes[bj], max(0.0, h - heights[bj]))

        # new merged cluster reuses index bi
        ni, nj = sizes[bi], sizes[bj]
        for k in active:
            if k in (bi, bj):
                continue
            dist[bi, k] = dist[k, bi] = (ni * dist[bi, k] + nj * dist[bj, k]) / (ni + nj)
        nodes[bi] = parent
        sizes[bi] = ni + nj
        heights[bi] = h
        active.remove(bj)

    return Tree(nodes[active[0]], rooted=True)


def neighbor_joining(labels: List[str], D: np.ndarray) -> Tree:
    """Neighbor-joining (Saitou & Nei, 1987); unrooted."""
    n = len(labels)
    if n == 0:
        return Tree(Node(), rooted=False)
    if n == 1:
        return Tree(Node(labels[0]), rooted=False)
    if n == 2:
        root = Node()
        a, b = Node(labels[0]), Node(labels[1])
        root.add_child(a, D[0, 1] / 2.0)
        root.add_child(b, D[0, 1] / 2.0)
        return Tree(root, rooted=False)

    dist: Dict[int, Dict[int, float]] = {}
    for i in range(n):
        dist[i] = {}
        for j in range(n):
            if i != j:
                dist[i][j] = float(D[i, j])
    nodes: Dict[int, Node] = {i: Node(labels[i]) for i in range(n)}
    active = list(range(n))
    next_id = n

    while len(active) > 2:
        r = len(active)
        s = {i: sum(dist[i][k] for k in active if k != i) for i in active}
        # find min Q
        best = None
        bi = bj = -1
        for a_idx in range(r):
            for b_idx in range(a_idx + 1, r):
                i, j = active[a_idx], active[b_idx]
                q = (r - 2) * dist[i][j] - s[i] - s[j]
                if best is None or q < best:
                    best, bi, bj = q, i, j

        dij = dist[bi][bj]
        li = 0.5 * dij + (s[bi] - s[bj]) / (2.0 * (r - 2))
        lj = dij - li
        u = Node()
        u.add_child(nodes[bi], max(0.0, li))
        u.add_child(nodes[bj], max(0.0, lj))
        uid = next_id
        next_id += 1
        nodes[uid] = u

        dist[uid] = {}
        for k in active:
            if k in (bi, bj):
                continue
            duk = 0.5 * (dist[bi][k] + dist[bj][k] - dij)
            dist[uid][k] = duk
            dist[k][uid] = duk
        active.remove(bi)
        active.remove(bj)
        active.append(uid)

    # join the final two
    i, j = active
    root = Node()
    root.add_child(nodes[i], max(0.0, dist[i][j] / 2.0))
    root.add_child(nodes[j], max(0.0, dist[i][j] / 2.0))
    return Tree(root, rooted=False)


def build_tree(labels: List[str], D: np.ndarray, method: str = "nj") -> Tree:
    if method == "nj":
        return neighbor_joining(labels, D)
    if method == "upgma":
        return upgma(labels, D)
    raise ValueError(f"unknown tree method {method!r}")

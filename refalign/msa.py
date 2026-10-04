"""Multiple sequence alignment over a population of trace-sequences.

Pure-Python, no MAFFT/Clustal binary required:
  * :func:`needleman_wunsch` — global pairwise alignment under the alphabet's
    soft substitution cost (minimisation form);
  * :func:`center_star_msa` — Gusfield center-star progressive MSA, which picks
    the medoid sequence and progressively merges every other sequence onto it,
    preserving the "once a gap, always a gap" invariant.

This is the process-mining "trace alignment" step, but applied to a *population*
of runs rather than a single log — the object the phylogeny is then built over.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .encode import Alphabet, GAP


# ---------------------------------------------------------------------------
def needleman_wunsch(a: np.ndarray, b: np.ndarray, alphabet: Alphabet,
                     gap: float = 1.0) -> Tuple[float, np.ndarray, np.ndarray]:
    """Global alignment of int-sequences ``a`` and ``b`` (cost minimisation).

    Returns (total_cost, aligned_a, aligned_b) where the aligned arrays contain
    :data:`GAP` in inserted positions and have equal length.
    """
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), dtype=np.float64)
    # 0 = diag (match/sub), 1 = up (gap in b), 2 = left (gap in a)
    ptr = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        D[i, 0] = i * gap
        ptr[i, 0] = 1
    for j in range(1, m + 1):
        D[0, j] = j * gap
        ptr[0, j] = 2
    for i in range(1, n + 1):
        ai = int(a[i - 1])
        for j in range(1, m + 1):
            sub = D[i - 1, j - 1] + alphabet.substitution_cost(ai, int(b[j - 1]))
            up = D[i - 1, j] + gap
            left = D[i, j - 1] + gap
            best = sub
            k = 0
            if up < best:
                best, k = up, 1
            if left < best:
                best, k = left, 2
            D[i, j] = best
            ptr[i, j] = k

    # traceback
    ra: List[int] = []
    rb: List[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        k = ptr[i, j]
        if k == 0:
            ra.append(int(a[i - 1]))
            rb.append(int(b[j - 1]))
            i -= 1
            j -= 1
        elif k == 1:
            ra.append(int(a[i - 1]))
            rb.append(GAP)
            i -= 1
        else:
            ra.append(GAP)
            rb.append(int(b[j - 1]))
            j -= 1
    ra.reverse()
    rb.reverse()
    return float(D[n, m]), np.asarray(ra, dtype=np.int64), np.asarray(rb, dtype=np.int64)


def pairwise_cost(a: np.ndarray, b: np.ndarray, alphabet: Alphabet,
                  gap: float = 1.0) -> float:
    """Edit-style alignment cost between two sequences (no alignment returned)."""
    return needleman_wunsch(a, b, alphabet, gap=gap)[0]


# ---------------------------------------------------------------------------
@dataclass
class MultipleAlignment:
    """Result of an MSA: a (n_rows x n_cols) matrix of symbol ids / GAP."""

    labels: List[str]                 # run_ids, one per row
    matrix: np.ndarray                # shape (n_rows, n_cols), dtype int64, GAP for gaps
    alphabet: Alphabet

    @property
    def n_rows(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_cols(self) -> int:
        return self.matrix.shape[1] if self.matrix.ndim == 2 else 0

    def row(self, label: str) -> np.ndarray:
        return self.matrix[self.labels.index(label)]

    def to_str(self, label: str) -> str:
        return " ".join(self.alphabet.symbol(int(x)) for x in self.row(label))


def _pick_center(seqs: List[np.ndarray], alphabet: Alphabet, gap: float) -> int:
    """Medoid: sequence minimising summed pairwise cost to all others."""
    n = len(seqs)
    if n == 1:
        return 0
    total = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = pairwise_cost(seqs[i], seqs[j], alphabet, gap)
            total[i] += c
            total[j] += c
    return int(np.argmin(total))


def center_star_msa(labels: List[str], seqs: List[np.ndarray], alphabet: Alphabet,
                    gap: float = 1.0, center: Optional[int] = None) -> MultipleAlignment:
    """Center-star multiple sequence alignment (Gusfield).

    ``center`` may be given to force the medoid (useful for determinism/tests);
    otherwise it is chosen as the sequence with minimum sum-of-pairs cost.
    """
    assert len(labels) == len(seqs)
    n = len(seqs)
    if n == 0:
        return MultipleAlignment([], np.zeros((0, 0), dtype=np.int64), alphabet)
    if n == 1:
        return MultipleAlignment(list(labels), seqs[0][None, :].copy(), alphabet)

    c = _pick_center(seqs, alphabet, gap) if center is None else center
    center_seq = seqs[c]

    # columns as a growing list; each column carries the ids for all rows so far,
    # plus which center residue index it corresponds to (None = center-gap column).
    cols: List[List[int]] = [[int(r)] for r in center_seq]
    center_pos: List[Optional[int]] = list(range(len(center_seq)))
    row_labels: List[str] = [labels[c]]

    order = [c] + [k for k in range(n) if k != c]
    for k in order[1:]:
        _, ca, sa = needleman_wunsch(center_seq, seqs[k], alphabet, gap=gap)
        n_rows = len(row_labels)
        p = 0            # pointer into existing columns
        t = 0            # pointer into center residues (center_pos values)
        for j in range(len(ca)):
            if ca[j] != GAP:
                # advance over existing center-gap columns, padding this row w/ GAP
                while p < len(cols) and center_pos[p] is None:
                    cols[p].append(GAP)
                    p += 1
                # cols[p] must be the center residue at index t
                cols[p].append(int(sa[j]))
                p += 1
                t += 1
            else:
                # center gap in the pairwise -> insert a fresh column here
                newcol = [GAP] * n_rows + [int(sa[j])]
                cols.insert(p, newcol)
                center_pos.insert(p, None)
                p += 1
        # any trailing columns not reached (all remaining are center-gap): pad GAP
        while p < len(cols):
            cols[p].append(GAP)
            p += 1
        row_labels.append(labels[k])

    matrix = np.asarray(cols, dtype=np.int64).T  # (rows, cols)
    # reorder rows back to the caller's original label order
    idx = [row_labels.index(lbl) for lbl in labels]
    matrix = matrix[idx]
    return MultipleAlignment(list(labels), matrix, alphabet)

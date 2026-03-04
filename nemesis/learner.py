"""
Online Learning — Prototype Refinement + Prompt Tuning for NEMESIS.

This module adds *actual learning* on top of the KNN memory store:

1. **Prototype Refinement**
   Maintains per-class centroid histograms (averaged token-frequency
   vectors).  After each classification pass, centroids are EMA-updated:
     - Correct → attract prototype toward the sample
     - Wrong   → repel prototype away from the sample
   During retrieval the prototype similarity is blended with the raw
   KNN similarity to re-rank candidates.

2. **Prompt Tuning**
   Each memory entry carries an *effectiveness score* (default 1.0).
   After classification, entries that appeared as few-shot neighbours
   get their scores updated:
     - Neighbour label == ground truth & prediction correct → boost
     - Neighbour label != ground truth & prediction wrong   → decay
   During retrieval the final score is:
     sim_final = sim_cosine * effectiveness^α + β * sim_prototype
"""

import json
import os
import sqlite3
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# =====================================================================
# Prototype Refinement
# =====================================================================

class PrototypeRefiner:
    """
    Maintains per-class prototype histograms (centroids) and updates
    them via exponential moving average after each classification.

    Prototypes are stored in a SQLite table for persistence across
    sessions.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        codebook_size: int = 1028,
        lr: float = 0.05,
        repel_lr: float = 0.02,
    ):
        self.conn = conn
        self.codebook_size = codebook_size
        self.lr = lr            # EMA rate for correct (attract)
        self.repel_lr = repel_lr  # EMA rate for wrong (repel)
        self._create_table()
        self._prototypes: Dict[str, np.ndarray] = {}  # activity → histogram
        self._counts: Dict[str, int] = {}
        self._load()

    def _create_table(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS prototypes (
                activity      TEXT PRIMARY KEY,
                histogram_json TEXT NOT NULL,
                update_count  INTEGER NOT NULL DEFAULT 0,
                last_updated  REAL NOT NULL
            );
        """)
        self.conn.commit()

    def _load(self):
        """Load prototypes from DB into memory."""
        rows = self.conn.execute(
            "SELECT activity, histogram_json, update_count FROM prototypes"
        ).fetchall()
        for activity, hist_json, count in rows:
            sparse = json.loads(hist_json)
            hist = np.zeros(self.codebook_size, dtype=np.float32)
            for tok_str, val in sparse.items():
                tid = int(tok_str)
                if 0 <= tid < self.codebook_size:
                    hist[tid] = val
            self._prototypes[activity] = hist
            self._counts[activity] = count

    def _save(self, activity: str):
        """Persist one prototype to DB."""
        hist = self._prototypes[activity]
        nz = np.nonzero(hist)[0]
        sparse = {str(int(i)): float(hist[i]) for i in nz}
        self.conn.execute(
            """INSERT OR REPLACE INTO prototypes
               (activity, histogram_json, update_count, last_updated)
               VALUES (?, ?, ?, ?)""",
            (activity, json.dumps(sparse), self._counts.get(activity, 0), time.time()),
        )
        self.conn.commit()

    # -----------------------------------------------------------------

    def init_from_memory(
        self,
        histograms: np.ndarray,
        labels: List[str],
    ):
        """
        Compute initial prototypes by averaging histograms per class.

        Args:
            histograms: (N, codebook_size) L2-normed histograms from memory index.
            labels: activity labels aligned with histograms.
        """
        class_sums: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(self.codebook_size, dtype=np.float32)
        )
        class_counts: Dict[str, int] = defaultdict(int)

        for i, label in enumerate(labels):
            class_sums[label] += histograms[i]
            class_counts[label] += 1

        for label in class_sums:
            proto = class_sums[label] / max(class_counts[label], 1)
            # L2-normalise
            norm = np.linalg.norm(proto)
            if norm > 0:
                proto /= norm
            self._prototypes[label] = proto
            self._counts[label] = class_counts[label]
            self._save(label)

        print(f"  [Prototypes] Initialised {len(self._prototypes)} class prototypes")

    def update(
        self,
        sample_histogram: np.ndarray,
        predicted_activity: str,
        ground_truth: str,
        is_correct: bool,
    ):
        """
        EMA-update prototypes after one classification.

        Correct → attract ground-truth prototype toward sample.
        Wrong   → repel predicted-class prototype away from sample,
                  attract ground-truth prototype toward sample.
        """
        # Always attract ground-truth prototype
        if ground_truth in self._prototypes:
            gt_proto = self._prototypes[ground_truth]
            gt_proto = (1 - self.lr) * gt_proto + self.lr * sample_histogram
            norm = np.linalg.norm(gt_proto)
            if norm > 0:
                gt_proto /= norm
            self._prototypes[ground_truth] = gt_proto
            self._counts[ground_truth] = self._counts.get(ground_truth, 0) + 1

        # If wrong → repel the wrongly-predicted prototype
        if not is_correct and predicted_activity in self._prototypes:
            if predicted_activity != ground_truth:
                wrong_proto = self._prototypes[predicted_activity]
                wrong_proto = (1 + self.repel_lr) * wrong_proto - self.repel_lr * sample_histogram
                norm = np.linalg.norm(wrong_proto)
                if norm > 0:
                    wrong_proto /= norm
                self._prototypes[predicted_activity] = wrong_proto

    def save_all(self):
        """Persist all prototypes."""
        for activity in self._prototypes:
            self._save(activity)

    def get_prototype_similarities(
        self,
        query_histogram: np.ndarray,
        labels: List[str],
    ) -> np.ndarray:
        """
        For each entry in the index, compute how similar the query is
        to that entry's class prototype.

        Returns (N,) array of prototype similarity scores.
        """
        N = len(labels)
        sims = np.zeros(N, dtype=np.float32)
        for i, label in enumerate(labels):
            if label in self._prototypes:
                sims[i] = float(np.dot(query_histogram, self._prototypes[label]))
        return sims

    @property
    def num_prototypes(self) -> int:
        return len(self._prototypes)


# =====================================================================
# Prompt Tuning (effectiveness scoring)
# =====================================================================

class PromptTuner:
    """
    Tracks per-memory-entry effectiveness scores and updates them
    after each classification round.

    The effectiveness score starts at 1.0 and drifts based on whether
    the entry, when used as a few-shot example, led to correct/wrong
    predictions.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        boost_lr: float = 0.15,
        decay_lr: float = 0.10,
        alpha: float = 0.3,      # exponent on effectiveness in retrieval
        beta: float = 0.2,       # weight on prototype similarity in retrieval
        min_score: float = 0.3,
        max_score: float = 3.0,
    ):
        self.conn = conn
        self.boost_lr = boost_lr
        self.decay_lr = decay_lr
        self.alpha = alpha
        self.beta = beta
        self.min_score = min_score
        self.max_score = max_score
        self._ensure_column()
        # In-memory cache: row_id → effectiveness
        self._scores: Dict[int, float] = {}
        self._load()

    def _ensure_column(self):
        """Add effectiveness column if it doesn't exist."""
        try:
            self.conn.execute(
                "ALTER TABLE memory ADD COLUMN effectiveness REAL NOT NULL DEFAULT 1.0"
            )
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

    def _load(self):
        """Load effectiveness scores from DB."""
        rows = self.conn.execute(
            "SELECT id, effectiveness FROM memory"
        ).fetchall()
        for row_id, eff in rows:
            self._scores[row_id] = eff

    def get_scores_for_ids(self, ids: List[int]) -> np.ndarray:
        """Return effectiveness scores for a list of row IDs."""
        return np.array(
            [self._scores.get(rid, 1.0) for rid in ids],
            dtype=np.float32,
        )

    def update_after_classification(
        self,
        neighbour_ids: List[int],
        neighbour_labels: List[str],
        predicted_activity: str,
        ground_truth: str,
        is_correct: bool,
    ):
        """
        Update effectiveness scores for the K neighbours used in a
        classification. 

        Rules:
          - Neighbour label == ground_truth AND correct → boost
          - Neighbour label == ground_truth AND wrong   → small boost 
            (label was right but LLM still got confused)
          - Neighbour label != ground_truth AND correct → no change
            (irrelevant neighbour didn't hurt)
          - Neighbour label != ground_truth AND wrong   → decay
            (misleading neighbour may have caused error)
        """
        for rid, nlabel in zip(neighbour_ids, neighbour_labels):
            old = self._scores.get(rid, 1.0)
            label_matches_gt = nlabel.lower().strip() == ground_truth.lower().strip()

            if label_matches_gt and is_correct:
                # Helpful neighbour → big boost
                new = old + self.boost_lr
            elif label_matches_gt and not is_correct:
                # Right label but LLM still wrong → tiny boost
                new = old + self.boost_lr * 0.3
            elif not label_matches_gt and not is_correct:
                # Misleading neighbour → decay
                new = old - self.decay_lr
            else:
                # Irrelevant but didn't hurt → no change
                new = old

            new = max(self.min_score, min(self.max_score, new))
            self._scores[rid] = new

    def save_all(self):
        """Batch-persist all updated effectiveness scores to DB."""
        updates = [(score, rid) for rid, score in self._scores.items()]
        self.conn.executemany(
            "UPDATE memory SET effectiveness = ? WHERE id = ?",
            updates,
        )
        self.conn.commit()

    def rerank_scores(
        self,
        cosine_sims: np.ndarray,
        ids: List[int],
        prototype_sims: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Produce final retrieval scores by blending cosine similarity,
        effectiveness, and prototype similarity.

        final = cosine_sim * effectiveness^alpha  +  beta * prototype_sim
        """
        eff = self.get_scores_for_ids(ids)
        scores = cosine_sims * np.power(eff, self.alpha)
        if prototype_sims is not None:
            scores += self.beta * prototype_sims
        return scores

    @property
    def stats(self) -> Dict[str, float]:
        if not self._scores:
            return {"mean": 1.0, "min": 1.0, "max": 1.0}
        vals = list(self._scores.values())
        return {
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

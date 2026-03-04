"""
Hierarchical Memory — few-shot experience store for NEMESIS.

Stores VQ-VAE token histograms alongside activity labels and metadata
in a SQLite database.  At inference time, the K most similar past
examples (by cosine similarity on the token histogram vector) are
retrieved and injected into the LLM prompt as few-shot examples,
grounding the otherwise opaque token IDs.

Tiers
-----
- **long_term**  – ground-truth labels from the training set, plus
  high-confidence inferences that have been promoted.
- **short_term** – recent inferences that haven't yet earned enough
  confidence to be promoted.

Metadata filtering ensures only comparable examples are retrieved
(same dataset, IMU position, sampling rate).
"""

import json
import math
import os
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from nemesis.config import MemoryConfig, LearnerConfig, CHECKPOINTS_DIR


# =====================================================================
# Helpers
# =====================================================================

def _token_histogram(tokens: List[int], codebook_size: int) -> np.ndarray:
    """Normalised frequency vector over the codebook (L2-normed)."""
    hist = np.zeros(codebook_size, dtype=np.float32)
    for t in tokens:
        if 0 <= t < codebook_size:
            hist[t] += 1
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm
    return hist


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (both assumed L2-normed)."""
    return float(np.dot(a, b))


# =====================================================================
# Data structures
# =====================================================================

@dataclass
class MemoryEntry:
    """One row in the memory store."""
    activity: str
    histogram_json: str       # JSON-encoded sparse histogram {tok_id: count, ...}
    confidence: float         # 1.0 = ground truth
    source: str               # "ground_truth" | "inference"
    tier: str                 # "long_term" | "short_term"
    dataset: str
    imu_position: str
    sampling_rate: int
    num_channels: int
    session_id: str
    timestamp: float = 0.0
    top_tokens_json: str = "[]"
    entropy: float = 0.0
    self_repetition: float = 0.0


# =====================================================================
# Memory Store
# =====================================================================

class MemoryStore:
    """
    SQLite-backed hierarchical memory with cosine-similarity retrieval.

    Design choices
    --------------
    * We store the *sparse* token histogram (only non-zero counts) as JSON
      so the DB stays compact.  At query time we inflate to a dense numpy
      vector for cosine similarity.
    * Metadata columns (dataset, imu_position, sampling_rate) are indexed
      so filtering is fast even with millions of rows.
    * We keep an in-memory numpy matrix of all histograms for the current
      metadata context so retrieval is a single matrix–vector dot.
    """

    def __init__(
        self,
        config: MemoryConfig = MemoryConfig(),
        learner_config: LearnerConfig = LearnerConfig(),
    ):
        self.config = config
        self.learner_config = learner_config
        self.codebook_size = config.codebook_size

        os.makedirs(os.path.dirname(config.db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(config.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

        # In-memory index (lazy-built per metadata context)
        self._index_meta: Optional[Dict] = None
        self._index_histograms: Optional[np.ndarray] = None   # (N, codebook)
        self._index_labels: Optional[List[str]] = None
        self._index_tiers: Optional[List[str]] = None
        self._index_confidences: Optional[List[float]] = None
        self._index_descriptors: Optional[List[str]] = None
        self._index_ids: Optional[List[int]] = None

        # Online learning components (lazy-init after first bootstrap/build)
        self._prototype_refiner = None
        self._prompt_tuner = None

    # -----------------------------------------------------------------
    # Schema
    # -----------------------------------------------------------------

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                activity         TEXT    NOT NULL,
                histogram_json   TEXT    NOT NULL,
                confidence       REAL    NOT NULL,
                source           TEXT    NOT NULL,
                tier             TEXT    NOT NULL DEFAULT 'short_term',
                dataset          TEXT    NOT NULL,
                imu_position     TEXT    NOT NULL,
                sampling_rate    INTEGER NOT NULL,
                num_channels     INTEGER NOT NULL,
                session_id       TEXT    NOT NULL,
                timestamp        REAL    NOT NULL,
                top_tokens_json  TEXT    NOT NULL DEFAULT '[]',
                entropy          REAL    NOT NULL DEFAULT 0,
                self_repetition  REAL    NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_meta
                ON memory(dataset, imu_position, sampling_rate);
            CREATE INDEX IF NOT EXISTS idx_tier ON memory(tier);
        """)
        self.conn.commit()

    # -----------------------------------------------------------------
    # Insert
    # -----------------------------------------------------------------

    def store(self, entry: MemoryEntry):
        """Insert a single memory entry."""
        self.conn.execute(
            """INSERT INTO memory
               (activity, histogram_json, confidence, source, tier,
                dataset, imu_position, sampling_rate, num_channels,
                session_id, timestamp, top_tokens_json, entropy,
                self_repetition)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                entry.activity,
                entry.histogram_json,
                entry.confidence,
                entry.source,
                entry.tier,
                entry.dataset,
                entry.imu_position,
                entry.sampling_rate,
                entry.num_channels,
                entry.session_id,
                entry.timestamp or time.time(),
                entry.top_tokens_json,
                entry.entropy,
                entry.self_repetition,
            ),
        )
        self.conn.commit()
        # Invalidate in-memory index
        self._index_meta = None

    def store_batch(self, entries: List[MemoryEntry]):
        """Insert many entries in a single transaction."""
        rows = [
            (
                e.activity, e.histogram_json, e.confidence, e.source, e.tier,
                e.dataset, e.imu_position, e.sampling_rate, e.num_channels,
                e.session_id, e.timestamp or time.time(), e.top_tokens_json,
                e.entropy, e.self_repetition,
            )
            for e in entries
        ]
        self.conn.executemany(
            """INSERT INTO memory
               (activity, histogram_json, confidence, source, tier,
                dataset, imu_position, sampling_rate, num_channels,
                session_id, timestamp, top_tokens_json, entropy,
                self_repetition)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self.conn.commit()
        self._index_meta = None

    # -----------------------------------------------------------------
    # Bootstrap from labeled training data
    # -----------------------------------------------------------------

    def bootstrap(
        self,
        tokens_list: List[List[int]],
        labels: List[str],
        dataset: str,
        imu_position: str,
        sampling_rate: int,
        num_channels: int,
        session_id: str = "bootstrap",
    ):
        """
        Populate long-term memory from labeled training data.

        This is the key grounding step: for each training sample we store
        a token histogram with the ground-truth label so that future
        queries can find "last time this histogram pattern appeared, the
        activity was X".

        Args:
            tokens_list: VQ-VAE token sequences for every training sample.
            labels:      Ground-truth activity label for each sample.
        """
        print(f"\n--- Memory Bootstrap ---")
        print(f"  Storing {len(tokens_list)} ground-truth entries...")

        from nemesis.token_descriptor import (
            token_entropy, self_repetition_rate, _strip_special,
        )

        entries = []
        for tokens, label in zip(tokens_list, labels):
            clean = _strip_special(tokens)
            if not clean:
                continue
            counts = Counter(clean)
            sparse = {str(k): v for k, v in counts.most_common()}
            top5 = counts.most_common(5)

            entries.append(MemoryEntry(
                activity=label,
                histogram_json=json.dumps(sparse),
                confidence=1.0,
                source="ground_truth",
                tier="long_term",
                dataset=dataset,
                imu_position=imu_position,
                sampling_rate=sampling_rate,
                num_channels=num_channels,
                session_id=session_id,
                timestamp=time.time(),
                top_tokens_json=json.dumps(top5),
                entropy=token_entropy(clean),
                self_repetition=self_repetition_rate(clean),
            ))

        self.store_batch(entries)
        print(f"  Stored {len(entries)} entries in long-term memory.")
        self._print_stats()

        # Initialise learner components from the freshly-stored data
        self._init_learners(dataset, imu_position, sampling_rate)

    # -----------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------

    def _init_learners(self, dataset: str, imu_position: str, sampling_rate: int):
        """Initialise prototype refiner + prompt tuner from current index."""
        from nemesis.learner import PrototypeRefiner, PromptTuner

        self._build_index(dataset, imu_position, sampling_rate)

        lc = self.learner_config
        self._prototype_refiner = PrototypeRefiner(
            conn=self.conn,
            codebook_size=self.codebook_size,
            lr=lc.prototype_lr,
            repel_lr=lc.prototype_repel_lr,
        )
        self._prompt_tuner = PromptTuner(
            conn=self.conn,
            boost_lr=lc.effectiveness_boost_lr,
            decay_lr=lc.effectiveness_decay_lr,
            alpha=lc.effectiveness_alpha,
            beta=lc.prototype_beta,
        )

        # Initialise prototypes if empty
        if self._prototype_refiner.num_prototypes == 0 and self._index_histograms.shape[0] > 0:
            self._prototype_refiner.init_from_memory(
                self._index_histograms, self._index_labels,
            )

    def _build_index(self, dataset: str, imu_position: str, sampling_rate: int):
        """Build in-memory numpy index for fast cosine queries."""
        meta = {"dataset": dataset, "imu_position": imu_position,
                "sampling_rate": sampling_rate}
        if self._index_meta == meta:
            return  # already built

        rows = self.conn.execute(
            """SELECT id, activity, histogram_json, confidence, tier,
                      top_tokens_json
               FROM memory
               WHERE dataset=? AND imu_position=? AND sampling_rate=?
               ORDER BY confidence DESC, tier ASC""",
            (dataset, imu_position, sampling_rate),
        ).fetchall()

        if not rows:
            self._index_meta = meta
            self._index_histograms = np.zeros((0, self.codebook_size), dtype=np.float32)
            self._index_labels = []
            self._index_tiers = []
            self._index_confidences = []
            self._index_descriptors = []
            self._index_ids = []
            return

        N = len(rows)
        hists = np.zeros((N, self.codebook_size), dtype=np.float32)
        labels = []
        tiers = []
        confs = []
        descs = []
        ids = []

        for i, (row_id, activity, hist_json, conf, tier, top_json) in enumerate(rows):
            sparse = json.loads(hist_json)
            for tok_str, cnt in sparse.items():
                tok_id = int(tok_str)
                if 0 <= tok_id < self.codebook_size:
                    hists[i, tok_id] = cnt
            # L2 normalise
            norm = np.linalg.norm(hists[i])
            if norm > 0:
                hists[i] /= norm
            labels.append(activity)
            tiers.append(tier)
            confs.append(conf)
            descs.append(top_json)
            ids.append(row_id)

        self._index_meta = meta
        self._index_histograms = hists
        self._index_labels = labels
        self._index_tiers = tiers
        self._index_confidences = confs
        self._index_descriptors = descs
        self._index_ids = ids

    def query(
        self,
        tokens: List[int],
        dataset: str,
        imu_position: str,
        sampling_rate: int,
        top_k: int = 0,
        long_term_only: bool = False,
    ) -> List[Dict]:
        """
        Diverse retrieval: find the top `num_diverse_activities` unique
        activities (by best similarity), then return `top_k_per_activity`
        entries from each.

        This ensures the LLM sees multiple activity hypotheses with
        strong supporting evidence for each, not just 5 entries that
        might all be the same class.

        Returns:
            List of dicts grouped by activity (top activities first,
            entries within each group sorted by similarity).
            Total entries = num_diverse_activities × top_k_per_activity
            (or fewer if not enough data).
        """
        k_per_act = self.config.top_k_per_activity   # 5
        n_acts = self.config.num_diverse_activities    # 3

        self._build_index(dataset, imu_position, sampling_rate)

        if self._index_histograms.shape[0] == 0:
            return []

        # Build query histogram
        q = _token_histogram(tokens, self.codebook_size)

        # Cosine similarities (single matrix-vector dot, very fast)
        raw_sims = self._index_histograms @ q

        # Optional filter to long-term only
        if long_term_only:
            mask = np.array([t == "long_term" for t in self._index_tiers])
            raw_sims = np.where(mask, raw_sims, -2.0)

        # Rerank with learner scores if available
        if self._prompt_tuner is not None and self._prototype_refiner is not None:
            proto_sims = self._prototype_refiner.get_prototype_similarities(
                q, self._index_labels,
            )
            sims = self._prompt_tuner.rerank_scores(
                raw_sims, self._index_ids, proto_sims,
            )
        else:
            sims = raw_sims

        # Sort ALL indices by descending similarity
        sorted_indices = np.argsort(sims)[::-1]

        # Group by activity: find top n_acts unique activities in rank order,
        # collecting up to k_per_act entries for each.
        from collections import OrderedDict
        activity_groups: OrderedDict = OrderedDict()  # activity -> list of idx

        for idx in sorted_indices:
            if sims[idx] < 0:
                continue
            act = self._index_labels[idx]
            if act not in activity_groups:
                if len(activity_groups) >= n_acts:
                    # Already have enough unique activities;
                    # only add if this activity is already tracked
                    continue
                activity_groups[act] = []
            if len(activity_groups[act]) < k_per_act:
                activity_groups[act].append(idx)

        # Flatten into results: activities in discovery order (best first),
        # entries within each activity by similarity.
        results = []
        for act, indices in activity_groups.items():
            for idx in indices:
                results.append({
                    "activity": self._index_labels[idx],
                    "similarity": float(sims[idx]),
                    "raw_similarity": float(raw_sims[idx]),
                    "confidence": self._index_confidences[idx],
                    "tier": self._index_tiers[idx],
                    "top_tokens": self._index_descriptors[idx],
                    "entry_id": self._index_ids[idx],
                })
        return results

    # -----------------------------------------------------------------
    # Confidence-weighted updates
    # -----------------------------------------------------------------

    def store_inference(
        self,
        tokens: List[int],
        predicted_activity: str,
        confidence: float,
        dataset: str,
        imu_position: str,
        sampling_rate: int,
        num_channels: int,
        session_id: str,
        neighbours: List[Dict],
    ):
        """
        Conditionally store an inference result.

        Rules:
          - confidence >= config.promote_threshold AND majority of
            neighbours agree → store directly in long_term
          - confidence >= config.store_threshold → store in short_term
          - else → discard
        """
        from nemesis.token_descriptor import (
            token_entropy, self_repetition_rate, _strip_special,
        )

        if confidence < self.config.store_threshold:
            return  # too low, discard

        # Check neighbour agreement
        if neighbours:
            neighbour_labels = [n["activity"] for n in neighbours]
            agree_count = sum(1 for l in neighbour_labels if l == predicted_activity)
            agree_ratio = agree_count / len(neighbour_labels)
        else:
            agree_ratio = 0.0

        if confidence >= self.config.promote_threshold and agree_ratio >= 0.6:
            tier = "long_term"
        else:
            tier = "short_term"

        clean = _strip_special(tokens)
        if not clean:
            return
        counts = Counter(clean)
        sparse = {str(k): v for k, v in counts.most_common()}
        top5 = counts.most_common(5)

        entry = MemoryEntry(
            activity=predicted_activity,
            histogram_json=json.dumps(sparse),
            confidence=confidence,
            source="inference",
            tier=tier,
            dataset=dataset,
            imu_position=imu_position,
            sampling_rate=sampling_rate,
            num_channels=num_channels,
            session_id=session_id,
            timestamp=time.time(),
            top_tokens_json=json.dumps(top5),
            entropy=token_entropy(clean),
            self_repetition=self_repetition_rate(clean),
        )
        self.store(entry)

    def promote_short_term(self, min_confidence: float = 0.0):
        """
        Promote short-term entries that exceed promote_threshold to
        long-term.  Called at end of session.
        """
        threshold = max(min_confidence, self.config.promote_threshold)
        cur = self.conn.execute(
            """UPDATE memory SET tier='long_term'
               WHERE tier='short_term' AND confidence >= ?""",
            (threshold,),
        )
        self.conn.commit()
        promoted = cur.rowcount
        if promoted > 0:
            print(f"  [Memory] Promoted {promoted} entries to long-term")
            self._index_meta = None

    # -----------------------------------------------------------------
    # Stats / utilities
    # -----------------------------------------------------------------

    def count(self, tier: Optional[str] = None) -> int:
        if tier:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM memory WHERE tier=?", (tier,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM memory").fetchone()
        return row[0]

    def _print_stats(self):
        lt = self.count("long_term")
        st = self.count("short_term")
        total = lt + st
        print(f"  [Memory] Total: {total} (long_term={lt}, short_term={st})")

    def update_learning(
        self,
        sample_tokens: List[int],
        neighbour_ids: List[int],
        neighbour_labels: List[str],
        predicted_activity: str,
        ground_truth: str,
        is_correct: bool,
    ):
        """Update prototypes + effectiveness after one classification."""
        if self._prototype_refiner is None or self._prompt_tuner is None:
            return

        sample_hist = _token_histogram(sample_tokens, self.codebook_size)

        # Prototype refinement
        self._prototype_refiner.update(
            sample_histogram=sample_hist,
            predicted_activity=predicted_activity,
            ground_truth=ground_truth,
            is_correct=is_correct,
        )

        # Prompt tuning
        self._prompt_tuner.update_after_classification(
            neighbour_ids=neighbour_ids,
            neighbour_labels=neighbour_labels,
            predicted_activity=predicted_activity,
            ground_truth=ground_truth,
            is_correct=is_correct,
        )

    def save_learning(self):
        """Persist learned parameters (prototypes + effectiveness scores)."""
        if self._prototype_refiner:
            self._prototype_refiner.save_all()
        if self._prompt_tuner:
            self._prompt_tuner.save_all()

    def learning_stats(self) -> Dict:
        """Return summary of learning parameters."""
        stats = {}
        if self._prototype_refiner:
            stats["num_prototypes"] = self._prototype_refiner.num_prototypes
        if self._prompt_tuner:
            stats["effectiveness"] = self._prompt_tuner.stats
        return stats

    def clear(self):
        """Delete all entries and learned parameters."""
        self.conn.execute("DELETE FROM memory")
        # prototypes table may not exist if learner was never initialised
        try:
            self.conn.execute("DELETE FROM prototypes")
        except sqlite3.OperationalError:
            pass
        self.conn.commit()
        self._index_meta = None
        self._prototype_refiner = None
        self._prompt_tuner = None

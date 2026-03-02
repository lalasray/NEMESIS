"""
Persistent Memory — cross-session knowledge storage for NEMESIS.

Two complementary systems:

1. Vector Memory (ChromaDB): stores embeddings of past translations
   for retrieval-augmented generation (few-shot examples for OpenAI).

2. Knowledge Graph (SQLite): stores learned activity rules and mappings
   for fast lookup and caching known patterns.

Together they allow NEMESIS to:
  - Remember what it learned in previous sessions
  - Avoid redundant OpenAI calls for known patterns
  - Provide relevant context to improve translation quality
"""

import os
import json
import time
import sqlite3
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from nemesis.config import MemoryConfig


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class MemoryEntry:
    """A single memory record linking IMU patterns to activities."""
    # Unique ID (hash of IMU pattern + symbolic output)
    entry_id: str
    # The neuro-symbolic output text
    symbolic_text: str
    # The interpreted activity from OpenAI
    activity: str
    # Confidence / reward score
    confidence: float
    # Timestamp
    timestamp: float
    # The IMU token sequence (as JSON list)
    imu_tokens_json: str
    # Additional metadata
    metadata_json: str = "{}"

    def to_dict(self) -> dict:
        return asdict(self)


def compute_entry_id(symbolic_text: str, imu_tokens: List[int]) -> str:
    """Deterministic ID from content."""
    # Convert numpy ints to plain Python ints for JSON serialization
    tokens = [int(t) for t in imu_tokens]
    content = f"{symbolic_text}|{json.dumps(tokens)}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ============================================================================
# Vector Memory (ChromaDB)
# ============================================================================

class VectorMemory:
    """
    Stores embeddings of past IMU→Symbolic→Activity translations.
    Supports similarity search for retrieval-augmented prompting.

    Uses ChromaDB for persistent vector storage.
    """

    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config
        os.makedirs(config.vector_db_path, exist_ok=True)

        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=config.vector_db_path)
            self.collection = self.client.get_or_create_collection(
                name=config.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
        except ImportError:
            print("[VectorMemory] chromadb not installed. Using fallback numpy store.")
            self._available = False
            self._fallback_store: List[Tuple[np.ndarray, dict]] = []
            self._fallback_path = os.path.join(config.vector_db_path, "fallback.json")
            self._load_fallback()

    def store(
        self,
        embedding: np.ndarray,
        entry: MemoryEntry,
    ):
        """
        Store a translation memory entry with its embedding.

        Args:
            embedding: (D,) vector representation of the IMU pattern
            entry: MemoryEntry with all metadata
        """
        if self._available:
            self.collection.upsert(
                ids=[entry.entry_id],
                embeddings=[embedding.tolist()],
                metadatas=[{
                    "symbolic_text": entry.symbolic_text,
                    "activity": entry.activity,
                    "confidence": entry.confidence,
                    "timestamp": entry.timestamp,
                    "imu_tokens": entry.imu_tokens_json,
                }],
                documents=[entry.symbolic_text],
            )
        else:
            self._fallback_store.append((embedding, entry.to_dict()))
            self._save_fallback()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Find the most similar past translations.

        Args:
            query_embedding: (D,) query vector
            top_k: number of results

        Returns:
            List of dicts with keys: symbolic_text, activity, confidence, distance
        """
        if self._available:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, max(1, self.collection.count())),
            )
            if not results["metadatas"] or not results["metadatas"][0]:
                return []

            entries = []
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                entries.append({
                    "symbolic_text": meta["symbolic_text"],
                    "activity": meta["activity"],
                    "confidence": meta["confidence"],
                    "distance": dist,
                })
            return entries
        else:
            return self._fallback_search(query_embedding, top_k)

    def count(self) -> int:
        if self._available:
            return self.collection.count()
        return len(self._fallback_store)

    # --- Fallback (numpy-based) ---

    def _fallback_search(self, query: np.ndarray, top_k: int) -> List[Dict]:
        if not self._fallback_store:
            return []
        embeddings = np.array([e[0] for e in self._fallback_store])
        # Cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        query_norm = np.linalg.norm(query) + 1e-8
        similarities = (embeddings @ query) / (norms.squeeze() * query_norm)
        top_indices = np.argsort(-similarities)[:top_k]

        results = []
        for idx in top_indices:
            meta = self._fallback_store[idx][1]
            results.append({
                "symbolic_text": meta["symbolic_text"],
                "activity": meta["activity"],
                "confidence": meta["confidence"],
                "distance": 1 - similarities[idx],
            })
        return results

    def _save_fallback(self):
        data = [(e.tolist(), m) for e, m in self._fallback_store]
        with open(self._fallback_path, "w") as f:
            json.dump(data, f)

    def _load_fallback(self):
        if os.path.exists(self._fallback_path):
            with open(self._fallback_path, "r") as f:
                data = json.load(f)
            self._fallback_store = [(np.array(e), m) for e, m in data]


# ============================================================================
# Knowledge Graph (SQLite)
# ============================================================================

class KnowledgeGraph:
    """
    SQLite-backed knowledge store for learned activity rules.

    Stores pattern→activity mappings that the system has learned
    with high confidence, enabling fast lookups without OpenAI calls.

    Schema:
      patterns: id, pattern_hash, symbolic_text, activity, confidence,
                times_seen, last_seen, metadata
    """

    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config
        os.makedirs(os.path.dirname(config.knowledge_db_path), exist_ok=True)
        self.conn = sqlite3.connect(config.knowledge_db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_hash TEXT UNIQUE NOT NULL,
                symbolic_text TEXT NOT NULL,
                activity TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                times_seen INTEGER DEFAULT 1,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pattern_hash ON patterns(pattern_hash)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_activity ON patterns(activity)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_confidence ON patterns(confidence DESC)
        """)

        # Session log table — tracks cross-session learning
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_start REAL NOT NULL,
                session_end REAL,
                num_translations INTEGER DEFAULT 0,
                num_new_patterns INTEGER DEFAULT 0,
                avg_reward REAL DEFAULT 0.0,
                notes TEXT DEFAULT ''
            )
        """)
        self.conn.commit()

    def lookup(self, symbolic_text: str) -> Optional[Dict]:
        """
        Look up a known pattern by its symbolic text.

        Returns:
            Dict with activity, confidence, times_seen or None if not found.
        """
        pattern_hash = hashlib.sha256(symbolic_text.encode()).hexdigest()[:16]
        row = self.conn.execute(
            "SELECT * FROM patterns WHERE pattern_hash = ?",
            (pattern_hash,),
        ).fetchone()

        if row and row["confidence"] >= self.config.confidence_threshold:
            return dict(row)
        return None

    def store(
        self,
        symbolic_text: str,
        activity: str,
        confidence: float,
        metadata: Optional[Dict] = None,
    ):
        """
        Store or update a pattern→activity mapping.

        If the pattern exists, update confidence (exponential moving average)
        and increment times_seen.
        """
        pattern_hash = hashlib.sha256(symbolic_text.encode()).hexdigest()[:16]
        now = time.time()
        meta_json = json.dumps(metadata or {})

        existing = self.conn.execute(
            "SELECT * FROM patterns WHERE pattern_hash = ?",
            (pattern_hash,),
        ).fetchone()

        if existing:
            # Update with EMA
            old_conf = existing["confidence"]
            new_conf = 0.7 * old_conf + 0.3 * confidence
            times = existing["times_seen"] + 1

            self.conn.execute("""
                UPDATE patterns
                SET confidence = ?, times_seen = ?, last_seen = ?,
                    activity = ?, metadata = ?
                WHERE pattern_hash = ?
            """, (new_conf, times, now, activity, meta_json, pattern_hash))
        else:
            self.conn.execute("""
                INSERT INTO patterns
                    (pattern_hash, symbolic_text, activity, confidence,
                     times_seen, first_seen, last_seen, metadata)
                VALUES (?, ?, ?, ?, 1, ?, ?, ?)
            """, (pattern_hash, symbolic_text, activity, confidence, now, now, meta_json))

        self.conn.commit()

    def get_high_confidence_patterns(self, limit: int = 50) -> List[Dict]:
        """Get the most confident learned patterns."""
        rows = self.conn.execute(
            "SELECT * FROM patterns WHERE confidence >= ? ORDER BY confidence DESC LIMIT ?",
            (self.config.confidence_threshold, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> Dict:
        """Get knowledge graph statistics."""
        total = self.conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
        high_conf = self.conn.execute(
            "SELECT COUNT(*) FROM patterns WHERE confidence >= ?",
            (self.config.confidence_threshold,),
        ).fetchone()[0]
        avg_conf = self.conn.execute(
            "SELECT AVG(confidence) FROM patterns",
        ).fetchone()[0] or 0.0

        return {
            "total_patterns": total,
            "high_confidence": high_conf,
            "avg_confidence": round(avg_conf, 4),
        }

    # --- Session tracking ---

    def start_session(self, notes: str = "") -> int:
        """Start a new session, returns session ID."""
        now = time.time()
        cursor = self.conn.execute(
            "INSERT INTO sessions (session_start, notes) VALUES (?, ?)",
            (now, notes),
        )
        self.conn.commit()
        return cursor.lastrowid

    def end_session(
        self,
        session_id: int,
        num_translations: int = 0,
        num_new_patterns: int = 0,
        avg_reward: float = 0.0,
    ):
        """End a session with summary stats."""
        now = time.time()
        self.conn.execute("""
            UPDATE sessions
            SET session_end = ?, num_translations = ?,
                num_new_patterns = ?, avg_reward = ?
            WHERE id = ?
        """, (now, num_translations, num_new_patterns, avg_reward, session_id))
        self.conn.commit()

    def get_session_history(self, limit: int = 20) -> List[Dict]:
        """Get recent session history."""
        rows = self.conn.execute(
            "SELECT * FROM sessions ORDER BY session_start DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()


# ============================================================================
# Unified Memory Manager
# ============================================================================

class MemoryManager:
    """
    Unified interface to both memory systems.

    Provides:
      - store_translation(): save a complete translation result
      - recall(): find similar past translations
      - fast_lookup(): check if pattern is already known
      - get_context_for_prompt(): build OpenAI prompt context from memory
    """

    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config
        self.vector_memory = VectorMemory(config)
        self.knowledge_graph = KnowledgeGraph(config)
        self.session_id: Optional[int] = None
        self._session_stats = {"translations": 0, "new_patterns": 0, "rewards": []}

    def start_session(self, notes: str = ""):
        self.session_id = self.knowledge_graph.start_session(notes)
        self._session_stats = {"translations": 0, "new_patterns": 0, "rewards": []}
        print(f"[Memory] Session #{self.session_id} started")

    def end_session(self):
        if self.session_id:
            avg_reward = (
                sum(self._session_stats["rewards"]) /
                max(1, len(self._session_stats["rewards"]))
            )
            self.knowledge_graph.end_session(
                self.session_id,
                self._session_stats["translations"],
                self._session_stats["new_patterns"],
                avg_reward,
            )
            print(f"[Memory] Session #{self.session_id} ended — "
                  f"{self._session_stats['translations']} translations, "
                  f"{self._session_stats['new_patterns']} new patterns")
            self.session_id = None

    def store_translation(
        self,
        imu_tokens: List[int],
        symbolic_text: str,
        activity: str,
        confidence: float,
        embedding: Optional[np.ndarray] = None,
    ):
        """
        Store a complete translation result in both memory systems.

        Args:
            imu_tokens: source IMU token IDs
            symbolic_text: generated neuro-symbolic output
            activity: interpreted activity
            confidence: reward / confidence score
            embedding: optional embedding vector for similarity search
        """
        entry_id = compute_entry_id(symbolic_text, imu_tokens)
        entry = MemoryEntry(
            entry_id=entry_id,
            symbolic_text=symbolic_text,
            activity=activity,
            confidence=confidence,
            timestamp=time.time(),
            imu_tokens_json=json.dumps([int(t) for t in imu_tokens]),
        )

        # Store in knowledge graph
        self.knowledge_graph.store(symbolic_text, activity, confidence)

        # Store in vector memory (if embedding provided)
        if embedding is not None:
            self.vector_memory.store(embedding, entry)

        # Session stats
        self._session_stats["translations"] += 1
        self._session_stats["new_patterns"] += 1
        self._session_stats["rewards"].append(confidence)

    def fast_lookup(self, symbolic_text: str) -> Optional[Dict]:
        """
        Check if we already know this pattern with high confidence.
        Returns cached activity or None.
        """
        return self.knowledge_graph.lookup(symbolic_text)

    def recall(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict]:
        """Find similar past translations via vector search."""
        return self.vector_memory.search(query_embedding, top_k)

    def get_context_for_prompt(
        self,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 3,
    ) -> str:
        """
        Build context string from memory to augment OpenAI prompts.
        Provides few-shot examples from past successful translations.
        """
        examples = []

        # Get from vector memory (similar patterns)
        if query_embedding is not None:
            similar = self.recall(query_embedding, top_k)
            for s in similar:
                if s["confidence"] > 0.5:
                    examples.append(
                        f"Previous example (confidence={s['confidence']:.2f}):\n"
                        f"  Symbolic: {s['symbolic_text']}\n"
                        f"  Activity: {s['activity']}"
                    )

        # Also include high-confidence known patterns
        known = self.knowledge_graph.get_high_confidence_patterns(limit=3)
        for k in known:
            examples.append(
                f"Known pattern (seen {k['times_seen']}x, confidence={k['confidence']:.2f}):\n"
                f"  Symbolic: {k['symbolic_text']}\n"
                f"  Activity: {k['activity']}"
            )

        if not examples:
            return ""

        return (
            "CONTEXT FROM PREVIOUS SESSIONS:\n"
            + "\n\n".join(examples[:top_k])
            + "\n\n"
        )

    def get_stats(self) -> Dict:
        """Get combined memory statistics."""
        kg_stats = self.knowledge_graph.get_stats()
        return {
            **kg_stats,
            "vector_memory_count": self.vector_memory.count(),
            "session_id": self.session_id,
            "session_translations": self._session_stats["translations"],
        }

    def close(self):
        self.end_session()
        self.knowledge_graph.close()

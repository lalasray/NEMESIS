"""
Neuro-Symbolic Language — defines the intermediate representation
that the Translator generates and OpenAI interprets.

Grammar:
  PREDICATE(key=value, key=value, ...)

Examples:
  MOTION(limb=right_arm, type=swing, intensity=high)
  POSTURE(state=upright, transition=stable)
  GAIT(pattern=stride, frequency=1.2hz)
  CONTEXT(duration=3s, repetitions=4)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from nemesis.config import NeuroSymbolicConfig


# ============================================================================
# Token Vocabulary
# ============================================================================

class SymbolicVocab:
    """
    Builds and manages the neuro-symbolic vocabulary mapping between
    string tokens and integer IDs.
    """

    def __init__(self, config: NeuroSymbolicConfig = NeuroSymbolicConfig()):
        self.config = config
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self):
        """Construct the full vocabulary from config."""
        tokens = []

        # Special tokens
        tokens.extend(["<PAD>", "<BOS>", "<EOS>", "<UNK>"])

        # Structure tokens
        tokens.extend(["(", ")", ",", "="])

        # Predicates
        tokens.extend(self.config.predicates)

        # Parameter names
        param_names = [
            "limb", "type", "intensity", "state", "transition",
            "pattern", "frequency", "direction", "angle", "duration",
            "repetitions", "axis", "magnitude", "phase", "confidence"
        ]
        tokens.extend(param_names)

        # Parameter values — body parts
        tokens.extend(self.config.limbs)

        # Motion types
        tokens.extend(self.config.motion_types)

        # Intensity levels
        tokens.extend(self.config.intensity_levels)

        # Gait patterns
        tokens.extend(self.config.gait_patterns)

        # Posture states
        tokens.extend(self.config.posture_states)

        # Directions
        directions = [
            "up", "down", "left", "right", "forward", "backward",
            "clockwise", "counterclockwise"
        ]
        tokens.extend(directions)

        # Transition types
        transitions = ["stable", "accelerating", "decelerating", "sudden", "gradual"]
        tokens.extend(transitions)

        # Numeric tokens (discretized)
        for i in range(100):
            tokens.append(f"NUM_{i}")
        # Units
        tokens.extend(["hz", "deg", "s", "ms", "g", "rad"])

        # Newline / statement separator
        tokens.append("<SEP>")

        # Deduplicate while preserving order
        seen = set()
        unique_tokens = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                unique_tokens.append(t)

        # Build mappings
        for idx, token in enumerate(unique_tokens):
            self.token2id[token] = idx
            self.id2token[idx] = token

    @property
    def size(self) -> int:
        return len(self.token2id)

    @property
    def pad_id(self) -> int:
        return self.token2id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token2id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token2id["<EOS>"]

    @property
    def unk_id(self) -> int:
        return self.token2id["<UNK>"]

    @property
    def sep_id(self) -> int:
        return self.token2id["<SEP>"]

    def encode(self, text: str) -> List[int]:
        """Encode a neuro-symbolic text string into token IDs."""
        raw_tokens = self._tokenize_text(text)
        ids = [self.bos_id]
        for t in raw_tokens:
            ids.append(self.token2id.get(t, self.unk_id))
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to neuro-symbolic text."""
        tokens = []
        for idx in ids:
            tok = self.id2token.get(idx, "<UNK>")
            if tok in ("<PAD>", "<BOS>", "<EOS>"):
                continue
            if tok == "<SEP>":
                tokens.append("\n")
            else:
                tokens.append(tok)
        return self._reconstruct(tokens)

    def _tokenize_text(self, text: str) -> List[str]:
        """Split neuro-symbolic text into vocabulary tokens."""
        tokens = []
        # Split by statement separators (newlines)
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # e.g. MOTION(limb=right_arm, type=swing, intensity=high)
            parts = re.findall(r'[A-Z_]+|[a-z_]+|[0-9]+\.?[0-9]*|[()=,]', line)
            for part in parts:
                # Check if it's a number
                if re.match(r'^[0-9]+\.?[0-9]*$', part):
                    num = int(float(part)) if float(part) == int(float(part)) else int(float(part))
                    num = min(num, 99)
                    tokens.append(f"NUM_{num}")
                else:
                    tokens.append(part)
            tokens.append("<SEP>")
        # Remove trailing SEP
        if tokens and tokens[-1] == "<SEP>":
            tokens.pop()
        return tokens

    def _reconstruct(self, tokens: List[str]) -> str:
        """Reconstruct readable text from decoded tokens."""
        result = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "\n":
                result.append("\n")
            elif tok == "(":
                result.append("(")
            elif tok == ")":
                result.append(")")
            elif tok == ",":
                result.append(", ")
            elif tok == "=":
                result.append("=")
            elif tok.startswith("NUM_"):
                result.append(tok.replace("NUM_", ""))
            else:
                result.append(tok)
            i += 1
        return "".join(result)


# ============================================================================
# Statement Data Structures
# ============================================================================

@dataclass
class SymbolicStatement:
    """A single neuro-symbolic statement like MOTION(limb=right_arm, type=swing)."""
    predicate: str
    params: Dict[str, str] = field(default_factory=dict)

    def to_string(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.predicate}({param_str})"

    @classmethod
    def from_string(cls, text: str) -> Optional["SymbolicStatement"]:
        """Parse a statement string like 'MOTION(limb=right_arm, type=swing)'."""
        match = re.match(r'(\w+)\((.+)\)', text.strip())
        if not match:
            return None
        predicate = match.group(1)
        param_str = match.group(2)
        params = {}
        for pair in param_str.split(","):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                params[k.strip()] = v.strip()
        return cls(predicate=predicate, params=params)


@dataclass
class SymbolicProgram:
    """A sequence of neuro-symbolic statements describing an activity."""
    statements: List[SymbolicStatement] = field(default_factory=list)

    def to_string(self) -> str:
        return "\n".join(s.to_string() for s in self.statements)

    @classmethod
    def from_string(cls, text: str) -> "SymbolicProgram":
        statements = []
        for line in text.strip().split("\n"):
            stmt = SymbolicStatement.from_string(line.strip())
            if stmt:
                statements.append(stmt)
        return cls(statements=statements)

    def to_prompt(self) -> str:
        """Format for the OpenAI API prompt."""
        return (
            "The following is a neuro-symbolic description of IMU sensor data "
            "capturing a person's movement. Each statement describes an aspect "
            "of the detected motion:\n\n"
            + self.to_string()
            + "\n\nBased on these movement primitives, what activity is the person performing? "
            "Provide a clear, concise description."
        )


# ============================================================================
# Predefined symbolic templates (for supervised pretraining)
# ============================================================================

ACTIVITY_TEMPLATES: Dict[str, List[SymbolicStatement]] = {
    "walking": [
        SymbolicStatement("GAIT", {"pattern": "stride", "frequency": "2"}),
        SymbolicStatement("MOTION", {"limb": "right_leg", "type": "swing", "intensity": "medium"}),
        SymbolicStatement("MOTION", {"limb": "left_leg", "type": "swing", "intensity": "medium"}),
        SymbolicStatement("MOTION", {"limb": "right_arm", "type": "swing", "intensity": "low"}),
        SymbolicStatement("POSTURE", {"state": "upright", "transition": "stable"}),
    ],
    "running": [
        SymbolicStatement("GAIT", {"pattern": "sprint", "frequency": "4"}),
        SymbolicStatement("MOTION", {"limb": "right_leg", "type": "swing", "intensity": "high"}),
        SymbolicStatement("MOTION", {"limb": "left_leg", "type": "swing", "intensity": "high"}),
        SymbolicStatement("MOTION", {"limb": "right_arm", "type": "swing", "intensity": "medium"}),
        SymbolicStatement("POSTURE", {"state": "leaning", "transition": "stable"}),
        SymbolicStatement("IMPACT", {"magnitude": "high", "frequency": "4"}),
    ],
    "sitting": [
        SymbolicStatement("POSTURE", {"state": "sitting", "transition": "stable"}),
        SymbolicStatement("STILLNESS", {"duration": "5", "confidence": "high"}),
    ],
    "jumping": [
        SymbolicStatement("MOTION", {"limb": "left_leg", "type": "extend", "intensity": "explosive"}),
        SymbolicStatement("MOTION", {"limb": "right_leg", "type": "extend", "intensity": "explosive"}),
        SymbolicStatement("IMPACT", {"magnitude": "high", "frequency": "2"}),
        SymbolicStatement("POSTURE", {"state": "crouching", "transition": "sudden"}),
        SymbolicStatement("CONTEXT", {"duration": "1", "repetitions": "3"}),
    ],
    "waving": [
        SymbolicStatement("MOTION", {"limb": "right_arm", "type": "rotate", "intensity": "medium"}),
        SymbolicStatement("GESTURE", {"type": "shake", "limb": "right_hand", "frequency": "3"}),
        SymbolicStatement("POSTURE", {"state": "upright", "transition": "stable"}),
        SymbolicStatement("CONTEXT", {"duration": "2", "repetitions": "5"}),
    ],
}


def get_template_program(activity: str) -> SymbolicProgram:
    """Get a predefined symbolic program for an activity."""
    stmts = ACTIVITY_TEMPLATES.get(activity, ACTIVITY_TEMPLATES["walking"])
    return SymbolicProgram(statements=list(stmts))

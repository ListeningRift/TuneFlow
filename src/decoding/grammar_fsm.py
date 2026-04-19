"""Finite-state grammar helpers for TuneFlow token sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


EXPECT_BOS = "expect_bos"
AFTER_BOS = "after_bos"
AFTER_HEAD_TEMPO = "after_head_tempo"
AFTER_BAR = "after_bar"
AFTER_BAR_TEMPO = "after_bar_tempo"
AFTER_POS = "after_pos"
AFTER_INST = "after_inst"
AFTER_PITCH = "after_pitch"
AFTER_DUR = "after_dur"
AFTER_VEL = "after_vel"
TERMINAL = "terminal"


@dataclass(frozen=True)
class GreedyMaskDecision:
    """Masked greedy decision plus diagnostics about the raw logits."""

    next_id: int | None
    raw_top1_id: int | None
    raw_top1_is_legal: bool
    legal_mass: float


class TuneFlowGrammarFSM:
    """DFA for the TuneFlow token grammar."""

    _NON_TERMINAL_STATES = (
        EXPECT_BOS,
        AFTER_BOS,
        AFTER_HEAD_TEMPO,
        AFTER_BAR,
        AFTER_BAR_TEMPO,
        AFTER_POS,
        AFTER_INST,
        AFTER_PITCH,
        AFTER_DUR,
        AFTER_VEL,
    )

    def __init__(self, token_to_id: Mapping[str, int]):
        self.token_to_id = {str(token): int(idx) for token, idx in token_to_id.items()}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        self.bos_id = self.token_to_id.get("BOS")
        self.eos_id = self.token_to_id.get("EOS")
        self.bar_id = self.token_to_id.get("BAR")
        if self.bos_id is None or self.eos_id is None or self.bar_id is None:
            raise ValueError("TuneFlow grammar requires BOS, EOS, and BAR tokens in the vocab.")

        self._tempo_ids: tuple[int, ...] = self._collect_prefix_ids("TEMPO_")
        self._pos_ids: tuple[int, ...] = self._collect_prefix_ids("POS_")
        self._inst_ids: tuple[int, ...] = self._collect_prefix_ids("INST_")
        self._pitch_ids: tuple[int, ...] = self._collect_prefix_ids("PITCH_")
        self._dur_ids: tuple[int, ...] = self._collect_prefix_ids("DUR_")
        self._vel_ids: tuple[int, ...] = self._collect_prefix_ids("VEL_")

        self._category_by_id: dict[int, str] = {}
        self._register_ids((self.bos_id,), "BOS")
        self._register_ids((self.eos_id,), "EOS")
        self._register_ids((self.bar_id,), "BAR")
        self._register_ids(self._tempo_ids, "TEMPO")
        self._register_ids(self._pos_ids, "POS")
        self._register_ids(self._inst_ids, "INST")
        self._register_ids(self._pitch_ids, "PITCH")
        self._register_ids(self._dur_ids, "DUR")
        self._register_ids(self._vel_ids, "VEL")

        self._allowed_ids_by_state: dict[str, tuple[int, ...]] = {
            EXPECT_BOS: (self.bos_id,),
            AFTER_BOS: (*self._tempo_ids, self.bar_id, self.eos_id),
            AFTER_HEAD_TEMPO: (self.bar_id, self.eos_id),
            AFTER_BAR: (*self._tempo_ids, *self._pos_ids, self.bar_id, self.eos_id),
            AFTER_BAR_TEMPO: (*self._pos_ids, self.bar_id, self.eos_id),
            AFTER_POS: self._inst_ids,
            AFTER_INST: self._pitch_ids,
            AFTER_PITCH: self._dur_ids,
            AFTER_DUR: self._vel_ids,
            AFTER_VEL: (*self._pos_ids, self.bar_id, self.eos_id),
            TERMINAL: (),
        }

    @classmethod
    def from_vocab(cls, token_to_id: Mapping[str, int]) -> "TuneFlowGrammarFSM":
        """Build an FSM from the tokenizer vocab mapping."""

        return cls(token_to_id)

    def _collect_prefix_ids(self, prefix: str) -> tuple[int, ...]:
        return tuple(
            idx
            for token, idx in sorted(self.token_to_id.items(), key=lambda item: item[1])
            if token.startswith(prefix)
        )

    def _register_ids(self, token_ids: Sequence[int], category: str) -> None:
        for token_id in token_ids:
            self._category_by_id[int(token_id)] = category

    def allowed_token_ids(self, state: str) -> tuple[int, ...]:
        """Return the token ids allowed by the grammar in the current state."""

        return self._allowed_ids_by_state.get(state, ())

    def transition(self, state: str, token_id: int) -> str | None:
        """Run one DFA transition; return None when the token is illegal."""

        category = self._category_by_id.get(int(token_id))
        if category is None:
            return None

        if state == EXPECT_BOS:
            return AFTER_BOS if category == "BOS" else None
        if state == AFTER_BOS:
            if category == "TEMPO":
                return AFTER_HEAD_TEMPO
            if category == "BAR":
                return AFTER_BAR
            if category == "EOS":
                return TERMINAL
            return None
        if state == AFTER_HEAD_TEMPO:
            if category == "BAR":
                return AFTER_BAR
            if category == "EOS":
                return TERMINAL
            return None
        if state == AFTER_BAR:
            if category == "TEMPO":
                return AFTER_BAR_TEMPO
            if category == "POS":
                return AFTER_POS
            if category == "BAR":
                return AFTER_BAR
            if category == "EOS":
                return TERMINAL
            return None
        if state == AFTER_BAR_TEMPO:
            if category == "POS":
                return AFTER_POS
            if category == "BAR":
                return AFTER_BAR
            if category == "EOS":
                return TERMINAL
            return None
        if state == AFTER_POS:
            return AFTER_INST if category == "INST" else None
        if state == AFTER_INST:
            return AFTER_PITCH if category == "PITCH" else None
        if state == AFTER_PITCH:
            return AFTER_DUR if category == "DUR" else None
        if state == AFTER_DUR:
            return AFTER_VEL if category == "VEL" else None
        if state == AFTER_VEL:
            if category == "POS":
                return AFTER_POS
            if category == "BAR":
                return AFTER_BAR
            if category == "EOS":
                return TERMINAL
            return None
        return None

    def state_after_prefix_tokens(self, tokens: Sequence[str]) -> str | None:
        """Return the DFA state after consuming a token prefix."""

        state = EXPECT_BOS
        for token in tokens:
            token_id = self.token_to_id.get(str(token))
            if token_id is None:
                return None
            state = self.transition(state, token_id)
            if state is None:
                return None
        return state

    def state_after_prefix_ids(self, token_ids: Sequence[int]) -> str | None:
        """Return the DFA state after consuming a token-id prefix."""

        state = EXPECT_BOS
        for token_id in token_ids:
            state = self.transition(state, int(token_id))
            if state is None:
                return None
        return state

    def inspect_complete_tokens(self, tokens: Sequence[str]) -> tuple[bool, str]:
        """Validate a complete sequence and return the first failure reason."""

        if not tokens:
            return False, "empty_sequence"

        state = EXPECT_BOS
        for index, token in enumerate(tokens):
            token_str = str(token)
            token_id = self.token_to_id.get(token_str)
            if token_id is None:
                return False, f"unknown_token@{index}:{token_str}"
            next_state = self.transition(state, token_id)
            if next_state is None:
                return False, self._unexpected_reason(state=state, index=index, token=token_str)
            state = next_state

        if state != TERMINAL:
            return False, self._unfinished_reason(state=state)
        return True, "ok"

    def compatible_states_for_suffix_tokens(self, suffix_tokens: Sequence[str]) -> set[str]:
        """Return states from which `suffix + EOS` can complete legally."""

        suffix_ids: list[int] = []
        for token in suffix_tokens:
            token_id = self.token_to_id.get(str(token))
            if token_id is None:
                return set()
            suffix_ids.append(token_id)
        return self.compatible_states_for_suffix_ids(suffix_ids)

    def compatible_states_for_suffix_ids(self, suffix_ids: Sequence[int]) -> set[str]:
        """Return states from which `suffix + EOS` can complete legally."""

        required_ids = [int(token_id) for token_id in suffix_ids]
        required_ids.append(self.eos_id)
        reachable: set[str] = {TERMINAL}
        for token_id in reversed(required_ids):
            previous_states = {
                state
                for state in self._NON_TERMINAL_STATES
                if self.transition(state, token_id) in reachable
            }
            reachable = previous_states
            if not reachable:
                break
        return reachable

    def bridgeable_states_for_suffix_tokens(self, suffix_tokens: Sequence[str]) -> set[str]:
        """返回仍可在不发出 EOS 的前提下走到 suffix 兼容态的状态集合。"""

        compatible_states = self.compatible_states_for_suffix_tokens(suffix_tokens)
        return self.bridgeable_states_for_target_states(compatible_states)

    def bridgeable_states_for_target_states(self, target_states: set[str]) -> set[str]:
        """返回可经由零到多步非 EOS 转移到达目标状态的状态集合。"""

        if not target_states:
            return set()

        reachable = set(target_states)
        changed = True
        while changed:
            changed = False
            for state in self._NON_TERMINAL_STATES:
                if state in reachable:
                    continue
                for token_id in self.allowed_token_ids(state):
                    if token_id == self.eos_id:
                        continue
                    next_state = self.transition(state, token_id)
                    if next_state in reachable:
                        reachable.add(state)
                        changed = True
                        break
        return reachable

    def _unexpected_reason(self, *, state: str, index: int, token: str) -> str:
        if state == EXPECT_BOS:
            return f"expected_bos@{index}:{token}"
        if state == AFTER_BOS:
            return f"expected_tempo_bar_or_eos@{index}:{token}"
        if state == AFTER_HEAD_TEMPO:
            return f"expected_bar_or_eos@{index}:{token}"
        if state == AFTER_BAR:
            return f"expected_tempo_pos_bar_or_eos@{index}:{token}"
        if state == AFTER_BAR_TEMPO:
            return f"expected_pos_bar_or_eos@{index}:{token}"
        if state == AFTER_POS:
            return f"expected_inst@{index}:{token}"
        if state == AFTER_INST:
            return f"expected_pitch@{index}:{token}"
        if state == AFTER_PITCH:
            return f"expected_dur@{index}:{token}"
        if state == AFTER_DUR:
            return f"expected_vel@{index}:{token}"
        if state == AFTER_VEL:
            return f"expected_pos_bar_or_eos@{index}:{token}"
        if state == TERMINAL:
            return f"unexpected_token_after_eos@{index}:{token}"
        return f"invalid_token@{index}:{token}"

    def _unfinished_reason(self, *, state: str) -> str:
        if state == EXPECT_BOS:
            return "empty_sequence"
        if state in {AFTER_BOS, AFTER_HEAD_TEMPO, AFTER_BAR, AFTER_BAR_TEMPO, AFTER_VEL}:
            return "missing_eos"
        if state == AFTER_POS:
            return "incomplete_note_tuple_expected_inst"
        if state == AFTER_INST:
            return "incomplete_note_tuple_expected_pitch"
        if state == AFTER_PITCH:
            return "incomplete_note_tuple_expected_dur"
        if state == AFTER_DUR:
            return "incomplete_note_tuple_expected_vel"
        return "incomplete_sequence"


def select_masked_argmax(logits, allowed_token_ids: Sequence[int]) -> GreedyMaskDecision:
    """Greedy argmax with a hard legality mask plus diagnostics."""

    import torch

    flat_logits = logits.reshape(-1)
    if flat_logits.numel() == 0:
        return GreedyMaskDecision(next_id=None, raw_top1_id=None, raw_top1_is_legal=False, legal_mass=0.0)

    raw_top1_id = int(torch.argmax(flat_logits).item())
    if not allowed_token_ids:
        return GreedyMaskDecision(next_id=None, raw_top1_id=raw_top1_id, raw_top1_is_legal=False, legal_mass=0.0)

    legal_mask = torch.zeros_like(flat_logits, dtype=torch.bool)
    legal_mask[list(allowed_token_ids)] = True
    legal_mass = float(torch.softmax(flat_logits.float(), dim=-1)[legal_mask].sum().item())
    raw_top1_is_legal = bool(legal_mask[raw_top1_id].item())

    min_value = torch.finfo(flat_logits.dtype).min
    masked_logits = torch.full_like(flat_logits, min_value)
    masked_logits[legal_mask] = flat_logits[legal_mask]
    next_id = int(torch.argmax(masked_logits).item())
    return GreedyMaskDecision(
        next_id=next_id,
        raw_top1_id=raw_top1_id,
        raw_top1_is_legal=raw_top1_is_legal,
        legal_mass=legal_mass,
    )

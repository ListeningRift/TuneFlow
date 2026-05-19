from __future__ import annotations

import unittest

from src.tokenizer import TokenizerConfig, build_vocab
from src.decoding.grammar_fsm import (
    AFTER_BAR,
    AFTER_BAR_TEMPO,
    AFTER_BAR_KEY,
    AFTER_BAR_TEMPO_KEY,
    AFTER_POS,
    AFTER_VEL,
    TuneFlowGrammarFSM,
)


class FsmPhraseTransitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab = build_vocab(TokenizerConfig())
        self.fsm = TuneFlowGrammarFSM(self.vocab)
        self.phrase_id = self.vocab["PHRASE"]
        self.pos0_id = self.vocab["POS_0"]
        self.bar_id = self.vocab["BAR"]
        self.eos_id = self.vocab["EOS"]

    def test_phrase_allowed_after_bar_head(self) -> None:
        for state in (AFTER_BAR, AFTER_BAR_TEMPO, AFTER_BAR_KEY, AFTER_BAR_TEMPO_KEY):
            self.assertIn(
                self.phrase_id,
                self.fsm.allowed_token_ids(state),
                msg=f"PHRASE must be allowed from {state}",
            )
            self.assertEqual(self.fsm.transition(state, self.phrase_id), "after_phrase")

    def test_phrase_allowed_after_vel(self) -> None:
        self.assertIn(self.phrase_id, self.fsm.allowed_token_ids(AFTER_VEL))
        self.assertEqual(self.fsm.transition(AFTER_VEL, self.phrase_id), "after_phrase")

    def test_after_phrase_accepts_only_pos(self) -> None:
        allowed = self.fsm.allowed_token_ids("after_phrase")
        self.assertEqual(set(allowed), set(self.fsm._pos_ids))
        self.assertEqual(self.fsm.transition("after_phrase", self.pos0_id), AFTER_POS)
        self.assertIsNone(self.fsm.transition("after_phrase", self.bar_id))
        self.assertIsNone(self.fsm.transition("after_phrase", self.eos_id))

    def test_compatible_states_for_phrase_suffix(self) -> None:
        suffix_ids = [
            self.phrase_id,
            self.pos0_id,
            self.vocab["INST_PIANO"],
            self.vocab["PITCH_60"],
            self.vocab["DUR_4"],
            self.vocab["VEL_8"],
        ]
        compatible = self.fsm.compatible_states_for_suffix_ids(suffix_ids)
        # AFTER_VEL can consume PHRASE+event 5-tuple+EOS legally
        self.assertIn(AFTER_VEL, compatible)
        # AFTER_BAR can also consume PHRASE+event 5-tuple+EOS legally
        self.assertIn(AFTER_BAR, compatible)


if __name__ == "__main__":
    unittest.main()

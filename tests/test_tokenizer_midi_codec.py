from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import mido

from src.tokenizer import TokenizerConfig, tokenize_midi, tokens_to_midi
from src.tokenizer.common import collect_tempo_changes


def _roundtrip_tokens() -> list[str]:
    return [
        "BOS",
        "TEMPO_120",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_60",
        "DUR_4",
        "VEL_8",
        "POS_8",
        "INST_PIANO",
        "PITCH_64",
        "DUR_6",
        "VEL_9",
        "BAR",
        "TEMPO_132",
        "POS_4",
        "INST_PIANO",
        "PITCH_67",
        "DUR_8",
        "VEL_10",
        "EOS",
    ]


def _default_tempo_tokens() -> list[str]:
    return [
        "BOS",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_72",
        "DUR_4",
        "VEL_8",
        "EOS",
    ]


def _continuation_full_tokens() -> list[str]:
    return [
        "BOS",
        "TEMPO_120",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_60",
        "DUR_4",
        "VEL_8",
        "BAR",
        "POS_4",
        "INST_PIANO",
        "PITCH_64",
        "DUR_4",
        "VEL_8",
        "EOS",
    ]


def _continuation_prompt_tokens() -> list[str]:
    return [
        "BOS",
        "TEMPO_120",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_60",
        "DUR_4",
        "VEL_8",
        "BAR",
    ]


def _continuation_output_tokens() -> list[str]:
    return [
        "POS_4",
        "INST_PIANO",
        "PITCH_64",
        "DUR_4",
        "VEL_8",
    ]


def _continuation_partial_expected_tokens() -> list[str]:
    return [
        "BOS",
        "TEMPO_120",
        "BAR",
        "BAR",
        "POS_4",
        "INST_PIANO",
        "PITCH_64",
        "DUR_4",
        "VEL_8",
        "EOS",
    ]


def _continuation_target_tokens() -> list[str]:
    return [
        "POS_8",
        "INST_PIANO",
        "PITCH_65",
        "DUR_4",
        "VEL_8",
    ]


def _continuation_target_expected_tokens() -> list[str]:
    return [
        "BOS",
        "TEMPO_120",
        "BAR",
        "BAR",
        "POS_8",
        "INST_PIANO",
        "PITCH_65",
        "DUR_4",
        "VEL_8",
        "EOS",
    ]


def _continuation_reference_full_tokens() -> list[str]:
    return [
        "BOS",
        "TEMPO_120",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_60",
        "DUR_4",
        "VEL_8",
        "BAR",
        "POS_8",
        "INST_PIANO",
        "PITCH_65",
        "DUR_4",
        "VEL_8",
        "EOS",
    ]


def _infilling_full_tokens() -> list[str]:
    return [
        "BOS",
        "TEMPO_120",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_60",
        "DUR_4",
        "VEL_8",
        "POS_4",
        "INST_PIANO",
        "PITCH_62",
        "DUR_4",
        "VEL_8",
        "POS_12",
        "INST_PIANO",
        "PITCH_67",
        "DUR_4",
        "VEL_8",
        "EOS",
    ]


def _infilling_prompt_tokens() -> list[str]:
    return [
        "BOS",
        "TEMPO_120",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_60",
        "DUR_4",
        "VEL_8",
        "FIM_HOLE",
        "POS_12",
        "INST_PIANO",
        "PITCH_67",
        "DUR_4",
        "VEL_8",
        "FIM_MID",
    ]


def _infilling_output_tokens() -> list[str]:
    return [
        "POS_4",
        "INST_PIANO",
        "PITCH_62",
        "DUR_4",
        "VEL_8",
    ]


def _infilling_partial_expected_tokens() -> list[str]:
    return [
        "BOS",
        "TEMPO_120",
        "BAR",
        "POS_4",
        "INST_PIANO",
        "PITCH_62",
        "DUR_4",
        "VEL_8",
        "EOS",
    ]


def _infilling_target_tokens() -> list[str]:
    return [
        "POS_8",
        "INST_PIANO",
        "PITCH_64",
        "DUR_4",
        "VEL_8",
    ]


def _infilling_target_expected_tokens() -> list[str]:
    return [
        "BOS",
        "TEMPO_120",
        "BAR",
        "POS_8",
        "INST_PIANO",
        "PITCH_64",
        "DUR_4",
        "VEL_8",
        "EOS",
    ]


def _infilling_reference_full_tokens() -> list[str]:
    return [
        "BOS",
        "TEMPO_120",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_60",
        "DUR_4",
        "VEL_8",
        "POS_8",
        "INST_PIANO",
        "PITCH_64",
        "DUR_4",
        "VEL_8",
        "POS_12",
        "INST_PIANO",
        "PITCH_67",
        "DUR_4",
        "VEL_8",
        "EOS",
    ]


class TokenizerMidiCodecTests(unittest.TestCase):
    def test_tokens_to_midi_roundtrip_preserves_quantized_tokens(self) -> None:
        config = TokenizerConfig()
        midi = tokens_to_midi(_roundtrip_tokens(), config, ticks_per_beat=480)

        self.assertEqual(tokenize_midi(midi, config), _roundtrip_tokens())

    def test_tokens_to_midi_defaults_missing_head_tempo_to_120(self) -> None:
        config = TokenizerConfig()
        midi = tokens_to_midi(_default_tempo_tokens(), config, ticks_per_beat=480)
        tempo_events = collect_tempo_changes(midi)
        reencoded = tokenize_midi(midi, config)

        self.assertAlmostEqual(float(tempo_events[0][1]), 120.0, places=4)
        self.assertEqual(reencoded[:3], ["BOS", "TEMPO_120", "BAR"])

    def test_tokens_to_midi_rejects_invalid_or_incomplete_sequences(self) -> None:
        config = TokenizerConfig()
        invalid_cases = [
            (["BAR", "EOS"], "valid complete TuneFlow sequence"),
            (["BOS", "FIM_HOLE", "EOS"], "unsupported structural tokens"),
            (["BOS", "TASK_CONT", "EOS"], "unsupported task tokens"),
        ]

        for tokens, message in invalid_cases:
            with self.subTest(tokens=tokens):
                with self.assertRaisesRegex(ValueError, message):
                    tokens_to_midi(tokens, config)


class ExportTokensToMidiCliTests(unittest.TestCase):
    def test_cli_exports_all_cases_by_default(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script_path = project_root / "scripts" / "eval" / "export_tokens_to_midi.py"
        config = TokenizerConfig()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_json = tmp_path / "continuation.json"
            output_dir = tmp_path / "midi_outputs"
            payload = {
                "task": "continuation",
                "cases": [
                    {
                        "prompt_tokens": _continuation_prompt_tokens(),
                        "fsm_reconstructed_tokens": _continuation_full_tokens(),
                        "fsm_output_tokens": _continuation_output_tokens(),
                        "target_tokens": _continuation_target_tokens(),
                        "raw_reconstructed_tokens": _continuation_full_tokens(),
                        "raw_output_tokens": _continuation_output_tokens(),
                    },
                    {
                        "prompt_tokens": _continuation_prompt_tokens(),
                        "fsm_reconstructed_tokens": _continuation_full_tokens(),
                        "fsm_output_tokens": _continuation_output_tokens(),
                        "target_tokens": _continuation_target_tokens(),
                        "raw_reconstructed_tokens": _continuation_full_tokens(),
                        "raw_output_tokens": _continuation_output_tokens(),
                    }
                ]
            }
            input_json.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, str(script_path), "--input-json", str(input_json), "--output", str(output_dir)],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            first_full = mido.MidiFile(output_dir / "0_full.mid", clip=True)
            first_partial = mido.MidiFile(output_dir / "0_continuation.mid", clip=True)
            first_target = mido.MidiFile(output_dir / "0_target.mid", clip=True)
            first_reference = mido.MidiFile(output_dir / "0_reference_full.mid", clip=True)
            second_full = mido.MidiFile(output_dir / "1_full.mid", clip=True)
            second_partial = mido.MidiFile(output_dir / "1_continuation.mid", clip=True)
            second_target = mido.MidiFile(output_dir / "1_target.mid", clip=True)
            second_reference = mido.MidiFile(output_dir / "1_reference_full.mid", clip=True)
            self.assertEqual(tokenize_midi(first_full, config), _continuation_full_tokens())
            self.assertEqual(tokenize_midi(first_partial, config), _continuation_partial_expected_tokens())
            self.assertEqual(tokenize_midi(first_target, config), _continuation_target_expected_tokens())
            self.assertEqual(tokenize_midi(first_reference, config), _continuation_reference_full_tokens())
            self.assertEqual(tokenize_midi(second_full, config), _continuation_full_tokens())
            self.assertEqual(tokenize_midi(second_partial, config), _continuation_partial_expected_tokens())
            self.assertEqual(tokenize_midi(second_target, config), _continuation_target_expected_tokens())
            self.assertEqual(tokenize_midi(second_reference, config), _continuation_reference_full_tokens())

    def test_cli_exports_single_case_when_case_index_is_provided(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script_path = project_root / "scripts" / "eval" / "export_tokens_to_midi.py"
        config = TokenizerConfig()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_json = tmp_path / "continuation.json"
            output_midi = tmp_path / "case.mid"
            payload = {
                "task": "continuation",
                "cases": [
                    {
                        "prompt_tokens": _continuation_prompt_tokens(),
                        "fsm_reconstructed_tokens": _default_tempo_tokens(),
                        "fsm_output_tokens": _continuation_output_tokens(),
                        "target_tokens": _continuation_target_tokens(),
                    },
                    {
                        "prompt_tokens": _continuation_prompt_tokens(),
                        "fsm_reconstructed_tokens": _continuation_full_tokens(),
                        "fsm_output_tokens": _continuation_output_tokens(),
                        "target_tokens": _continuation_target_tokens(),
                    },
                ]
            }
            input_json.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--input-json",
                    str(input_json),
                    "--case-index",
                    "1",
                    "--output",
                    str(output_midi),
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(output_midi.exists())
            midi = mido.MidiFile(output_midi, clip=True)
            partial_midi = mido.MidiFile(tmp_path / "case_continuation.mid", clip=True)
            target_midi = mido.MidiFile(tmp_path / "case_target.mid", clip=True)
            reference_midi = mido.MidiFile(tmp_path / "case_reference_full.mid", clip=True)
            self.assertEqual(tokenize_midi(midi, config), _continuation_full_tokens())
            self.assertEqual(tokenize_midi(partial_midi, config), _continuation_partial_expected_tokens())
            self.assertEqual(tokenize_midi(target_midi, config), _continuation_target_expected_tokens())
            self.assertEqual(tokenize_midi(reference_midi, config), _continuation_reference_full_tokens())

    def test_cli_exports_infilling_partial_midi(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script_path = project_root / "scripts" / "eval" / "export_tokens_to_midi.py"
        config = TokenizerConfig()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_json = tmp_path / "infilling.json"
            output_midi = tmp_path / "case.mid"
            payload = {
                "task": "infilling",
                "cases": [
                    {
                        "prompt_tokens": _infilling_prompt_tokens(),
                        "fsm_reconstructed_tokens": _infilling_full_tokens(),
                        "fsm_output_tokens": _infilling_output_tokens(),
                        "target_hole_tokens": _infilling_target_tokens(),
                    }
                ]
            }
            input_json.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--input-json",
                    str(input_json),
                    "--case-index",
                    "0",
                    "--output",
                    str(output_midi),
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            full_midi = mido.MidiFile(output_midi, clip=True)
            partial_midi = mido.MidiFile(tmp_path / "case_infilling.mid", clip=True)
            target_midi = mido.MidiFile(tmp_path / "case_target.mid", clip=True)
            reference_midi = mido.MidiFile(tmp_path / "case_reference_full.mid", clip=True)
            self.assertEqual(tokenize_midi(full_midi, config), _infilling_full_tokens())
            self.assertEqual(tokenize_midi(partial_midi, config), _infilling_partial_expected_tokens())
            self.assertEqual(tokenize_midi(target_midi, config), _infilling_target_expected_tokens())
            self.assertEqual(tokenize_midi(reference_midi, config), _infilling_reference_full_tokens())

    def test_cli_errors_when_case_index_is_out_of_range(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script_path = project_root / "scripts" / "eval" / "export_tokens_to_midi.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_json = tmp_path / "continuation.json"
            input_json.write_text(
                json.dumps({"task": "continuation", "cases": [{}]}, ensure_ascii=False),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--input-json",
                    str(input_json),
                    "--case-index",
                    "2",
                    "--output",
                    str(tmp_path / "case.mid"),
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("case_index 2 is out of range", result.stderr)

    def test_cli_requires_directory_output_when_exporting_all_cases(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script_path = project_root / "scripts" / "eval" / "export_tokens_to_midi.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_json = tmp_path / "continuation.json"
            input_json.write_text(
                json.dumps(
                    {
                        "task": "continuation",
                        "cases": [{"fsm_reconstructed_tokens": _roundtrip_tokens()}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--input-json",
                    str(input_json),
                    "--output",
                    str(tmp_path / "case.mid"),
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("`--output` must be a directory path", result.stderr)

    def test_cli_rejects_fragment_only_token_fields(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script_path = project_root / "scripts" / "eval" / "export_tokens_to_midi.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_json = tmp_path / "continuation.json"
            payload = {
                "task": "continuation",
                "cases": [
                    {
                        "prompt_tokens": _continuation_prompt_tokens(),
                        "fsm_reconstructed_tokens": _roundtrip_tokens(),
                        "fsm_output_tokens": _continuation_output_tokens(),
                        "target_tokens": _continuation_target_tokens(),
                        "raw_output_tokens": ["POS_0", "INST_PIANO"],
                    }
                ]
            }
            input_json.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--input-json",
                    str(input_json),
                    "--case-index",
                    "0",
                    "--token-field",
                    "raw_output_tokens",
                    "--output",
                    str(tmp_path / "case.mid"),
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("only complete sequence fields are supported", result.stderr)

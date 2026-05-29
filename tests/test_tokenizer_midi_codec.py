from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import mido

from scripts.data.build_data import PipelineArgs, build_commands
from scripts.eval.export_tokens_to_midi import _build_structure_only_prefix
from src.tokenizer import TokenizerConfig, build_vocab, tokenize_midi, tokens_to_midi
from src.tokenizer.common import collect_tempo_changes
from src.tokenizer.midi_codec import inject_key_tokens, inject_phrase_tokens
from src.tokenizer.tokenize_dataset import _compute_parallel_batch_size, process as tokenize_dataset_process


def _roundtrip_tokens() -> list[str]:
    return inject_phrase_tokens(inject_key_tokens([
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
    ]))


def _default_tempo_tokens() -> list[str]:
    return [
        "BOS",
        "KEY_UNCERTAIN",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_72",
        "DUR_4",
        "VEL_8",
        "EOS",
    ]


def _continuation_full_tokens() -> list[str]:
    return inject_phrase_tokens(inject_key_tokens([
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
    ]))


def _continuation_prompt_tokens() -> list[str]:
    return [
        "BOS",
        "TEMPO_120",
        "KEY_UNCERTAIN",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_60",
        "DUR_4",
        "VEL_8",
        "BAR",
    ]


def _continuation_prompt_tokens_with_phrase() -> list[str]:
    return [
        "BOS",
        "TEMPO_120",
        "KEY_UNCERTAIN",
        "BAR",
        "PHRASE",
        "POS_0",
        "INST_PIANO",
        "PITCH_60",
        "DUR_4",
        "VEL_8",
        "PHRASE",
        "POS_4",
        "INST_PIANO",
        "PITCH_62",
        "DUR_4",
        "VEL_8",
        "BAR",
    ]


def _continuation_prompt_tokens_ending_with_phrase() -> list[str]:
    return [
        "BOS",
        "TEMPO_120",
        "KEY_UNCERTAIN",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_60",
        "DUR_4",
        "VEL_8",
        "PHRASE",
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
    return inject_phrase_tokens([
        "BOS",
        "TEMPO_120",
        "KEY_UNCERTAIN",
        "BAR",
        "BAR",
        "POS_4",
        "INST_PIANO",
        "PITCH_64",
        "DUR_4",
        "VEL_8",
        "EOS",
    ])


def _continuation_target_tokens() -> list[str]:
    return [
        "POS_8",
        "INST_PIANO",
        "PITCH_65",
        "DUR_4",
        "VEL_8",
    ]


def _continuation_target_expected_tokens() -> list[str]:
    return inject_phrase_tokens([
        "BOS",
        "TEMPO_120",
        "KEY_UNCERTAIN",
        "BAR",
        "BAR",
        "POS_8",
        "INST_PIANO",
        "PITCH_65",
        "DUR_4",
        "VEL_8",
        "EOS",
    ])


def _continuation_reference_full_tokens() -> list[str]:
    return inject_phrase_tokens(inject_key_tokens([
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
    ]))


def _infilling_full_tokens() -> list[str]:
    return inject_phrase_tokens(inject_key_tokens([
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
    ]))


def _infilling_prompt_tokens() -> list[str]:
    return [
        "BOS",
        "TEMPO_120",
        "KEY_UNCERTAIN",
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
    return inject_phrase_tokens([
        "BOS",
        "TEMPO_120",
        "KEY_UNCERTAIN",
        "BAR",
        "POS_4",
        "INST_PIANO",
        "PITCH_62",
        "DUR_4",
        "VEL_8",
        "EOS",
    ])


def _infilling_target_tokens() -> list[str]:
    return [
        "POS_8",
        "INST_PIANO",
        "PITCH_64",
        "DUR_4",
        "VEL_8",
    ]


def _infilling_target_expected_tokens() -> list[str]:
    return inject_phrase_tokens([
        "BOS",
        "TEMPO_120",
        "KEY_UNCERTAIN",
        "BAR",
        "POS_8",
        "INST_PIANO",
        "PITCH_64",
        "DUR_4",
        "VEL_8",
        "EOS",
    ])


def _infilling_reference_full_tokens() -> list[str]:
    return inject_phrase_tokens(inject_key_tokens([
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
    ]))


def _c_major_roundtrip_tokens() -> list[str]:
    return inject_phrase_tokens([
        "BOS",
        "TEMPO_120",
        "KEY_C_MAJ",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_60",
        "DUR_12",
        "VEL_8",
        "POS_8",
        "INST_PIANO",
        "PITCH_64",
        "DUR_8",
        "VEL_8",
        "POS_16",
        "INST_PIANO",
        "PITCH_67",
        "DUR_12",
        "VEL_8",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_57",
        "DUR_12",
        "VEL_8",
        "POS_8",
        "INST_PIANO",
        "PITCH_60",
        "DUR_8",
        "VEL_8",
        "POS_16",
        "INST_PIANO",
        "PITCH_64",
        "DUR_12",
        "VEL_8",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_65",
        "DUR_12",
        "VEL_8",
        "POS_8",
        "INST_PIANO",
        "PITCH_69",
        "DUR_8",
        "VEL_8",
        "POS_16",
        "INST_PIANO",
        "PITCH_72",
        "DUR_12",
        "VEL_8",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_60",
        "DUR_12",
        "VEL_8",
        "POS_8",
        "INST_PIANO",
        "PITCH_64",
        "DUR_8",
        "VEL_8",
        "POS_16",
        "INST_PIANO",
        "PITCH_67",
        "DUR_12",
        "VEL_8",
        "EOS",
    ])


def _c_to_g_major_roundtrip_tokens() -> list[str]:
    return inject_phrase_tokens([
        "BOS",
        "TEMPO_120",
        "KEY_C_MAJ",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_60",
        "DUR_12",
        "VEL_8",
        "POS_8",
        "INST_PIANO",
        "PITCH_64",
        "DUR_8",
        "VEL_8",
        "POS_16",
        "INST_PIANO",
        "PITCH_67",
        "DUR_12",
        "VEL_8",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_57",
        "DUR_12",
        "VEL_8",
        "POS_8",
        "INST_PIANO",
        "PITCH_60",
        "DUR_8",
        "VEL_8",
        "POS_16",
        "INST_PIANO",
        "PITCH_64",
        "DUR_12",
        "VEL_8",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_65",
        "DUR_12",
        "VEL_8",
        "POS_8",
        "INST_PIANO",
        "PITCH_69",
        "DUR_8",
        "VEL_8",
        "POS_16",
        "INST_PIANO",
        "PITCH_72",
        "DUR_12",
        "VEL_8",
        "BAR",
        "KEY_G_MAJ",
        "POS_0",
        "INST_PIANO",
        "PITCH_60",
        "DUR_12",
        "VEL_8",
        "POS_8",
        "INST_PIANO",
        "PITCH_64",
        "DUR_8",
        "VEL_8",
        "POS_16",
        "INST_PIANO",
        "PITCH_67",
        "DUR_12",
        "VEL_8",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_67",
        "DUR_12",
        "VEL_8",
        "POS_8",
        "INST_PIANO",
        "PITCH_71",
        "DUR_8",
        "VEL_8",
        "POS_16",
        "INST_PIANO",
        "PITCH_74",
        "DUR_12",
        "VEL_8",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_64",
        "DUR_12",
        "VEL_8",
        "POS_8",
        "INST_PIANO",
        "PITCH_67",
        "DUR_8",
        "VEL_8",
        "POS_16",
        "INST_PIANO",
        "PITCH_71",
        "DUR_12",
        "VEL_8",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_60",
        "DUR_12",
        "VEL_8",
        "POS_8",
        "INST_PIANO",
        "PITCH_67",
        "DUR_8",
        "VEL_8",
        "POS_16",
        "INST_PIANO",
        "PITCH_71",
        "DUR_12",
        "VEL_8",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_67",
        "DUR_12",
        "VEL_8",
        "POS_8",
        "INST_PIANO",
        "PITCH_71",
        "DUR_8",
        "VEL_8",
        "POS_16",
        "INST_PIANO",
        "PITCH_74",
        "DUR_12",
        "VEL_8",
        "EOS",
    ])


class TokenizerMidiCodecTests(unittest.TestCase):
    def test_build_vocab_appends_key_tokens_without_shifting_existing_ids(self) -> None:
        vocab = build_vocab(TokenizerConfig())
        self.assertEqual(vocab["BAR"], 4)
        self.assertEqual(vocab["PHRASE"], 5)
        self.assertEqual(vocab["TEMPO_40"], 153)
        self.assertEqual(vocab["TEMPO_220"], 243)
        self.assertEqual(vocab["KEY_C_MAJ"], 244)
        self.assertEqual(vocab["KEY_B_MIN"], 267)
        self.assertEqual(vocab["KEY_UNCERTAIN"], 268)

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
        self.assertEqual(reencoded[:4], ["BOS", "TEMPO_120", "KEY_UNCERTAIN", "BAR"])

    def test_tokens_to_midi_roundtrip_preserves_major_key_token(self) -> None:
        config = TokenizerConfig()
        midi = tokens_to_midi(_c_major_roundtrip_tokens(), config, ticks_per_beat=480)

        self.assertEqual(tokenize_midi(midi, config), _c_major_roundtrip_tokens())

    def test_tokens_to_midi_roundtrip_preserves_sparse_modulation_key_tokens(self) -> None:
        config = TokenizerConfig()
        midi = tokens_to_midi(_c_to_g_major_roundtrip_tokens(), config, ticks_per_beat=480)

        self.assertEqual(tokenize_midi(midi, config), _c_to_g_major_roundtrip_tokens())

    def test_tokenize_dataset_reports_key_token_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            midi_root = tmp_path / "midi"
            midi_root.mkdir(parents=True, exist_ok=True)
            output_dir = tmp_path / "tokenized"
            vocab_path = output_dir / "tokenizer_vocab.json"
            stats_path = output_dir / "token_stats.json"
            split_path = tmp_path / "train.jsonl"
            midi_path = midi_root / "sample.mid"

            midi = tokens_to_midi(_c_major_roundtrip_tokens(), TokenizerConfig(), ticks_per_beat=480)
            midi.save(str(midi_path))
            split_path.write_text(json.dumps({"midi_path": "sample.mid"}, ensure_ascii=False) + "\n", encoding="utf-8")

            config = TokenizerConfig(
                midi_root_dir=str(midi_root),
                train_transpose_offsets=[],
                split_files={"train": str(split_path)},
            )
            tokenize_dataset_process(
                config=config,
                output_dir=output_dir,
                vocab_path=vocab_path,
                stats_path=stats_path,
                limit_per_split=None,
            )

            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            key_token_stats = stats["key_token_stats"]
            split_key_token_stats = stats["split_stats"]["train"]["key_token_stats"]
            self.assertGreater(int(key_token_stats["total_key_tokens"]), 0)
            self.assertEqual(int(key_token_stats["major_total"]), int(key_token_stats["total_key_tokens"]))
            self.assertEqual(int(key_token_stats["counts_by_token"]["KEY_C_MAJ"]), 1)
            self.assertEqual(int(split_key_token_stats["counts_by_token"]["KEY_C_MAJ"]), 1)

    def test_tokenize_dataset_workers_preserves_output_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            midi_root = tmp_path / "midi"
            midi_root.mkdir(parents=True, exist_ok=True)
            split_path = tmp_path / "train.jsonl"
            serial_output_dir = tmp_path / "serial"
            parallel_output_dir = tmp_path / "parallel"
            serial_vocab_path = serial_output_dir / "tokenizer_vocab.json"
            parallel_vocab_path = parallel_output_dir / "tokenizer_vocab.json"
            serial_stats_path = serial_output_dir / "token_stats.json"
            parallel_stats_path = parallel_output_dir / "token_stats.json"

            midi_specs = [
                ("sample_a.mid", _c_major_roundtrip_tokens()),
                ("sample_b.mid", _roundtrip_tokens()),
            ]
            split_lines: list[str] = []
            for midi_name, tokens in midi_specs:
                midi = tokens_to_midi(tokens, TokenizerConfig(), ticks_per_beat=480)
                midi.save(str(midi_root / midi_name))
                split_lines.append(json.dumps({"midi_path": midi_name}, ensure_ascii=False))
            split_path.write_text("\n".join(split_lines) + "\n", encoding="utf-8")

            config = TokenizerConfig(
                midi_root_dir=str(midi_root),
                train_transpose_offsets=[],
                split_files={"train": str(split_path)},
            )
            tokenize_dataset_process(
                config=config,
                output_dir=serial_output_dir,
                vocab_path=serial_vocab_path,
                stats_path=serial_stats_path,
                limit_per_split=None,
                workers=1,
            )
            tokenize_dataset_process(
                config=config,
                output_dir=parallel_output_dir,
                vocab_path=parallel_vocab_path,
                stats_path=parallel_stats_path,
                limit_per_split=None,
                workers=2,
            )

            self.assertEqual(
                (serial_output_dir / "train.tok").read_text(encoding="utf-8"),
                (parallel_output_dir / "train.tok").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                json.loads(serial_vocab_path.read_text(encoding="utf-8")),
                json.loads(parallel_vocab_path.read_text(encoding="utf-8")),
            )
            serial_stats = json.loads(serial_stats_path.read_text(encoding="utf-8"))
            parallel_stats = json.loads(parallel_stats_path.read_text(encoding="utf-8"))
            serial_stats["split_stats"]["train"]["output_file"] = "<normalized>"
            parallel_stats["split_stats"]["train"]["output_file"] = "<normalized>"
            self.assertEqual(
                serial_stats,
                parallel_stats,
            )

    def test_tokenize_dataset_parallel_workers_prints_live_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            midi_root = tmp_path / "midi"
            midi_root.mkdir(parents=True, exist_ok=True)
            split_path = tmp_path / "train.jsonl"
            output_dir = tmp_path / "parallel"
            vocab_path = output_dir / "tokenizer_vocab.json"
            stats_path = output_dir / "token_stats.json"

            midi_specs = [
                ("sample_a.mid", _c_major_roundtrip_tokens()),
                ("sample_b.mid", _roundtrip_tokens()),
            ]
            split_lines: list[str] = []
            for midi_name, tokens in midi_specs:
                midi = tokens_to_midi(tokens, TokenizerConfig(), ticks_per_beat=480)
                midi.save(str(midi_root / midi_name))
                split_lines.append(json.dumps({"midi_path": midi_name}, ensure_ascii=False))
            split_path.write_text("\n".join(split_lines) + "\n", encoding="utf-8")

            config = TokenizerConfig(
                midi_root_dir=str(midi_root),
                train_transpose_offsets=[],
                split_files={"train": str(split_path)},
            )
            stdout_buffer = StringIO()
            with redirect_stdout(stdout_buffer):
                tokenize_dataset_process(
                    config=config,
                    output_dir=output_dir,
                    vocab_path=vocab_path,
                    stats_path=stats_path,
                    limit_per_split=None,
                    workers=2,
                )

            output = stdout_buffer.getvalue()
            self.assertIn("[tokenize] start", output)
            self.assertIn("split=train start", output)
            self.assertIn("dispatch workers=2", output)
            self.assertIn("remaining=", output)
            self.assertIn("total_progress=", output)

    def test_compute_parallel_batch_size_uses_larger_batches_for_large_parallel_runs(self) -> None:
        self.assertEqual(_compute_parallel_batch_size(12760, 8), 128)
        self.assertEqual(_compute_parallel_batch_size(100, 8), 100)
        self.assertEqual(_compute_parallel_batch_size(1, 8), 1)

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
    def test_build_data_includes_tokenize_workers_argument(self) -> None:
        args = PipelineArgs(
            python_exec=sys.executable,
            clean_config=Path("configs/data/cleaning.yaml"),
            split_config=Path("configs/data/split.yaml"),
            tokenizer_config=Path("configs/tokenizer/tokenizer.yaml"),
            build_config=Path("configs/data/build_training.yaml"),
            validate_report_path=Path("outputs/reports/data/validate_data_report.json"),
            start_from="clean",
            stop_after="validate",
            clean_limit=None,
            split_limit=None,
            tokenize_limit_per_split=16,
            tokenize_workers=8,
        )

        commands = build_commands(args)
        self.assertEqual(
            commands["tokenize"],
            [
                sys.executable,
                "scripts/data/tokenize_dataset.py",
                "--config",
                str(Path("configs/tokenizer/tokenizer.yaml")),
                "--limit-per-split",
                "16",
                "--workers",
                "8",
            ],
        )

    def test_structure_only_prefix_skips_phrase_prefixed_events(self) -> None:
        prefix = _build_structure_only_prefix(_continuation_prompt_tokens_with_phrase(), stop_at_hole=False)
        self.assertEqual(
            prefix,
            [
                "BOS",
                "TEMPO_120",
                "KEY_UNCERTAIN",
                "BAR",
                "BAR",
            ],
        )

    def test_structure_only_prefix_preserves_trailing_phrase_boundary(self) -> None:
        prefix = _build_structure_only_prefix(_continuation_prompt_tokens_ending_with_phrase(), stop_at_hole=False)
        self.assertEqual(
            prefix,
            [
                "BOS",
                "TEMPO_120",
                "KEY_UNCERTAIN",
                "BAR",
                "PHRASE",
            ],
        )

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

    def test_cli_accepts_phrase_tokens_inside_continuation_prompt(self) -> None:
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
                        "prompt_tokens": _continuation_prompt_tokens_with_phrase(),
                        "fsm_reconstructed_tokens": _continuation_full_tokens(),
                        "fsm_output_tokens": _continuation_output_tokens(),
                        "target_tokens": _continuation_target_tokens(),
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
            partial_midi = mido.MidiFile(tmp_path / "case_continuation.mid", clip=True)
            target_midi = mido.MidiFile(tmp_path / "case_target.mid", clip=True)
            self.assertEqual(tokenize_midi(partial_midi, config), _continuation_partial_expected_tokens())
            self.assertEqual(tokenize_midi(target_midi, config), _continuation_target_expected_tokens())

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

    def test_cli_reports_available_checkpoints_when_input_json_is_missing(self) -> None:
        """缺失输入 JSON 时应提示同目录可用的 checkpoint。"""
        project_root = Path(__file__).resolve().parents[1]
        script_path = project_root / "scripts" / "eval" / "export_tokens_to_midi.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            final_top3_dir = tmp_path / "outputs" / "benchmark" / "base_small" / "samples" / "final_top3"
            for step_name in ("step_750", "step_1000", "step_3000"):
                step_dir = final_top3_dir / step_name
                step_dir.mkdir(parents=True, exist_ok=True)
                (step_dir / "continuation.json").write_text(
                    json.dumps({"task": "continuation", "cases": []}, ensure_ascii=False),
                    encoding="utf-8",
                )

            missing_input_json = final_top3_dir / "step_4000" / "continuation.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--input-json",
                    str(missing_input_json),
                    "--output",
                    str(tmp_path / "case.mid"),
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("benchmark sample json不存在", result.stderr)
            self.assertIn("final_top3", result.stderr)
            self.assertIn("step_750", result.stderr)
            self.assertIn("step_1000", result.stderr)
            self.assertIn("step_3000", result.stderr)


class PhraseTokenTests(unittest.TestCase):
    def test_phrase_token_in_vocab(self) -> None:
        config = TokenizerConfig()
        vocab = build_vocab(config)
        self.assertIn("PHRASE", vocab)
        self.assertIn("BAR", vocab)
        self.assertLess(vocab["BAR"], vocab["PHRASE"], "PHRASE must follow BAR in vocab order")
        self.assertLess(vocab["PHRASE"], vocab["POS_0"], "PHRASE must precede POS_* tokens")

    def test_validate_token_order_accepts_bar_head_phrase(self) -> None:
        from src.tokenizer.midi_codec import validate_token_order
        config = TokenizerConfig()
        vocab = build_vocab(config)
        tokens = [
            "BOS", "TEMPO_120", "KEY_UNCERTAIN",
            "BAR", "PHRASE", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8",
            "EOS",
        ]
        valid, oov = validate_token_order(tokens, vocab)
        self.assertTrue(valid)
        self.assertEqual(oov, 0)

    def test_validate_token_order_accepts_mid_bar_phrase(self) -> None:
        from src.tokenizer.midi_codec import validate_token_order
        config = TokenizerConfig()
        vocab = build_vocab(config)
        tokens = [
            "BOS", "TEMPO_120", "KEY_UNCERTAIN",
            "BAR", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8",
            "PHRASE", "POS_8", "INST_PIANO", "PITCH_64", "DUR_4", "VEL_8",
            "EOS",
        ]
        valid, _ = validate_token_order(tokens, vocab)
        self.assertTrue(valid)

    def test_validate_token_order_rejects_consecutive_phrase(self) -> None:
        from src.tokenizer.midi_codec import validate_token_order
        config = TokenizerConfig()
        vocab = build_vocab(config)
        tokens = [
            "BOS", "TEMPO_120", "KEY_UNCERTAIN",
            "BAR", "PHRASE", "PHRASE", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8",
            "EOS",
        ]
        valid, _ = validate_token_order(tokens, vocab)
        self.assertFalse(valid)

    def test_validate_token_order_rejects_phrase_before_bar(self) -> None:
        from src.tokenizer.midi_codec import validate_token_order
        config = TokenizerConfig()
        vocab = build_vocab(config)
        tokens_phrase_before_bar = [
            "BOS", "TEMPO_120", "KEY_UNCERTAIN",
            "BAR", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8",
            "PHRASE", "BAR", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8",
            "EOS",
        ]
        valid, _ = validate_token_order(tokens_phrase_before_bar, vocab)
        self.assertFalse(valid)

    def test_validate_token_order_rejects_phrase_at_bos(self) -> None:
        from src.tokenizer.midi_codec import validate_token_order
        config = TokenizerConfig()
        vocab = build_vocab(config)
        tokens = [
            "BOS", "PHRASE", "TEMPO_120", "KEY_UNCERTAIN",
            "BAR", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8",
            "EOS",
        ]
        valid, _ = validate_token_order(tokens, vocab)
        self.assertFalse(valid)

    def test_inject_phrase_tokens_forces_first_phrase(self) -> None:
        from src.tokenizer.midi_codec import inject_phrase_tokens
        tokens = [
            "BOS", "TEMPO_120", "KEY_UNCERTAIN",
            "BAR",
            "BAR", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8",
            "BAR", "POS_0", "INST_PIANO", "PITCH_62", "DUR_4", "VEL_8",
            "EOS",
        ]
        out = inject_phrase_tokens(tokens)
        bar_positions = [i for i, t in enumerate(out) if t == "BAR"]
        self.assertEqual(out[bar_positions[1] + 1], "PHRASE")
        self.assertEqual(out[bar_positions[1] + 2], "POS_0")

    def test_inject_phrase_tokens_no_phrase_on_empty_bar(self) -> None:
        from src.tokenizer.midi_codec import inject_phrase_tokens
        tokens = ["BOS", "TEMPO_120", "KEY_UNCERTAIN", "BAR", "EOS"]
        out = inject_phrase_tokens(tokens)
        self.assertNotIn("PHRASE", out)

    def test_inject_phrase_tokens_dedups_adjacent(self) -> None:
        from src.tokenizer.midi_codec import inject_phrase_tokens
        tokens = [
            "BOS", "TEMPO_120", "KEY_UNCERTAIN",
            "BAR", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8",
            "EOS",
        ]
        out = inject_phrase_tokens(tokens)
        for i in range(len(out) - 1):
            self.assertFalse(out[i] == "PHRASE" and out[i + 1] == "PHRASE")

    def test_tokenize_midi_emits_phrase_tokens(self) -> None:
        midi = mido.MidiFile(type=1, ticks_per_beat=480)
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120.0), time=0))
        track.append(mido.Message("note_on", note=60, velocity=80, time=0))
        track.append(mido.Message("note_off", note=60, velocity=0, time=240))
        track.append(mido.Message("note_on", note=62, velocity=80, time=1920))
        track.append(mido.Message("note_off", note=62, velocity=0, time=240))
        config = TokenizerConfig()
        tokens = tokenize_midi(midi, config)
        self.assertIn("PHRASE", tokens)

    def test_tokens_to_midi_ignores_phrase(self) -> None:
        config = TokenizerConfig()
        base = inject_key_tokens([
            "BOS", "TEMPO_120",
            "BAR", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8",
            "BAR", "POS_0", "INST_PIANO", "PITCH_62", "DUR_4", "VEL_8",
            "EOS",
        ])
        with_phrase = list(base)
        bar_idx = with_phrase.index("BAR")
        header_end = bar_idx + 1
        while header_end < len(with_phrase) and (
            with_phrase[header_end].startswith("TEMPO_") or with_phrase[header_end].startswith("KEY_")
        ):
            header_end += 1
        with_phrase.insert(header_end, "PHRASE")
        midi_no = tokens_to_midi(base, config)
        midi_yes = tokens_to_midi(with_phrase, config)
        msgs_no = [m for m in midi_no.tracks[1] if not m.is_meta and m.type in {"note_on", "note_off"}]
        msgs_yes = [m for m in midi_yes.tracks[1] if not m.is_meta and m.type in {"note_on", "note_off"}]
        self.assertEqual(
            [(m.type, m.note, m.velocity, m.time) for m in msgs_no],
            [(m.type, m.note, m.velocity, m.time) for m in msgs_yes],
        )


if __name__ == "__main__":
    unittest.main()

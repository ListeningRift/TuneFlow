from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.music_analysis import (
    BarInfo,
    HierarchicalBoundaryScore,
    KeyAnalysisConfig,
    NoteInfo,
    PhraseAnalysis,
    PhraseAnalysisConfig,
    PhraseBoundary,
    PhraseSpan,
    analyze_key_timeline,
    analyze_phrase_candidates,
)
from src.tokenizer import TokenizerConfig, tokens_to_midi
from src.tokenizer.midi_codec import inject_key_tokens, inject_phrase_tokens
from src.utils.annotation_review import (
    ReviewBuildConfig,
    _describe_phrase_boundary,
    build_debug_flags,
    build_review_case,
    load_benchmark_cases,
    serialize_phrase_analysis,
    tokens_to_note_payloads,
    write_review_bundle,
)


def _bar(*events: tuple[int, int, int]) -> list[str]:
    """构造单个小节的 token 片段。"""
    tokens = ["BAR"]
    for pos, pitch, dur in events:
        tokens.extend(
            [
                f"POS_{pos}",
                "INST_PIANO",
                f"PITCH_{pitch}",
                f"DUR_{dur}",
                "VEL_8",
            ]
        )
    return tokens


def _c_to_g_major_tokens() -> list[str]:
    """构造一个带转调的测试 token 序列。"""
    return inject_phrase_tokens(
        inject_key_tokens(
            [
                "BOS",
                "TEMPO_120",
                *_bar((0, 60, 12), (8, 64, 8), (16, 67, 12)),
                *_bar((0, 57, 12), (8, 60, 8), (16, 64, 12)),
                *_bar((0, 65, 12), (8, 69, 8), (16, 72, 12)),
                *_bar((0, 60, 12), (8, 64, 8), (16, 67, 12)),
                *_bar((0, 67, 12), (8, 71, 8), (16, 74, 12)),
                *_bar((0, 64, 12), (8, 67, 8), (16, 71, 12)),
                *_bar((0, 60, 12), (8, 67, 8), (16, 71, 12)),
                *_bar((0, 67, 12), (8, 71, 8), (16, 74, 12)),
                "EOS",
            ]
        )
    )


def _project_root() -> Path:
    """返回项目根目录。"""
    return Path(__file__).resolve().parents[1]


def _review_config() -> ReviewBuildConfig:
    """构造测试用 review 配置。"""
    tokenizer_config = TokenizerConfig()
    return ReviewBuildConfig(
        tokenizer_config=tokenizer_config,
        key_config=KeyAnalysisConfig(positions_per_bar=tokenizer_config.positions_per_bar),
        phrase_config=PhraseAnalysisConfig(positions_per_bar=tokenizer_config.positions_per_bar),
        low_margin_threshold=0.10,
    )


class AnnotationReviewTests(unittest.TestCase):
    def test_serialize_phrase_analysis_prefers_non_none_score_for_same_anchor(self) -> None:
        """同一锚点存在多条评分时，边界原因应优先使用真实命中而不是后续 none。"""

        analysis = PhraseAnalysis(
            bars=(
                BarInfo(
                    start_token=0,
                    end_token=10,
                    note_count=1,
                    onset_count=1,
                    rest_ratio=0.0,
                    pitch_span=0,
                    mean_duration=4.0,
                    effective_tempo_token="TEMPO_120",
                    effective_key_token=None,
                    onset_positions=(0,),
                ),
                BarInfo(
                    start_token=10,
                    end_token=20,
                    note_count=1,
                    onset_count=1,
                    rest_ratio=0.0,
                    pitch_span=0,
                    mean_duration=4.0,
                    effective_tempo_token="TEMPO_120",
                    effective_key_token=None,
                    onset_positions=(0,),
                ),
            ),
            notes=(
                NoteInfo(0, 0, 4, 4, 60, 0, 0, None),
                NoteInfo(1, 32, 36, 4, 62, 1, 0, None),
            ),
            boundary_features=tuple(),
            boundary_scores=(
                HierarchicalBoundaryScore(
                    note_index=0,
                    unit=32,
                    bar_index=1,
                    anchor_pos=0,
                    motif_score=0.42,
                    subphrase_score=0.56,
                    phrase_score=0.72,
                    boundary_type="phrase",
                    sequence_role="none",
                    reasons=("adjacent_repeated_bar_span", "repeat_end"),
                ),
                HierarchicalBoundaryScore(
                    note_index=1,
                    unit=32,
                    bar_index=1,
                    anchor_pos=0,
                    motif_score=0.0,
                    subphrase_score=0.0,
                    phrase_score=0.0,
                    boundary_type="none",
                    sequence_role="none",
                    reasons=tuple(),
                ),
            ),
            boundaries=(PhraseBoundary(bar_index=1, anchor_pos=0),),
            phrase_spans=(
                PhraseSpan(
                    start_bar=0,
                    end_bar=2,
                    start_token=0,
                    end_token=20,
                    tempo_token="TEMPO_120",
                    key_token=None,
                    tokens=tuple(),
                    source_kind="single_phrase",
                ),
            ),
        )

        payload = serialize_phrase_analysis(analysis, positions_per_bar=32)

        self.assertEqual(payload["boundaries"][0]["source_rule"], "长跨度重复")
        self.assertEqual(payload["boundaries"][0]["source_reasons"], ["长跨度重复", "重复结束"])

    def test_describe_phrase_boundary_uses_real_note_level_reasons(self) -> None:
        payload = _describe_phrase_boundary(
            {"bar_index": 9, "anchor_pos": 0},
            boundary_score={
                "boundary_type": "phrase",
                "reasons": ["adjacent_repeated_bar_span", "repeat_end", "motive_end"],
                "score": 0.88,
            },
            first_content_bar=5,
        )
        self.assertEqual(payload["source_rule"], "长跨度重复")
        self.assertEqual(payload["source_label"], "长跨度重复+")
        self.assertEqual(payload["source_reasons"], ["长跨度重复", "重复结束", "动机收束"])

    def test_describe_phrase_boundary_keeps_first_phrase_as_forced(self) -> None:
        payload = _describe_phrase_boundary(
            {"bar_index": 5, "anchor_pos": 0},
            boundary_score={
                "boundary_type": "motif",
                "reasons": ["motive_end", "repeat_start"],
                "score": 0.48,
            },
            first_content_bar=5,
        )
        self.assertEqual(payload["source_rule"], "首句强制")
        self.assertEqual(payload["source_label"], "首句强制")
        self.assertEqual(payload["source_reasons"], ["首句强制"])

    def test_tokens_to_note_payloads_extracts_note_positions(self) -> None:
        tokens = _c_to_g_major_tokens()
        notes = tokens_to_note_payloads(tokens, positions_per_bar=32)
        self.assertGreater(len(notes), 0)
        self.assertEqual(notes[0]["start_bar"], 0)
        self.assertEqual(notes[0]["start_pos"], 0)
        self.assertGreater(notes[0]["end_unit"], notes[0]["start_unit"])

    def test_build_review_case_serializes_key_and_phrase_analysis(self) -> None:
        config = _review_config()
        tokens = _c_to_g_major_tokens()
        case = build_review_case(
            case_id="demo-1",
            source_kind="raw_midi",
            title="demo",
            subtitle="demo-subtitle",
            source_path="demo.mid",
            meta={"artist": "tester"},
            tokens=tokens,
            config=config,
        )
        self.assertIn("initial_key", case["key_analysis"])
        self.assertIn("frames", case["key_analysis"])
        self.assertIn("boundaries", case["phrase_analysis"])
        self.assertIn("phrase_spans", case["phrase_analysis"])
        self.assertIn("notes", case["phrase_analysis"])
        self.assertIn("boundary_features", case["phrase_analysis"])
        self.assertIn("boundary_scores", case["phrase_analysis"])
        self.assertIn("source_rule", case["phrase_analysis"]["boundaries"][0])
        self.assertIn("source_label", case["phrase_analysis"]["boundaries"][0])
        self.assertIn("source_reasons", case["phrase_analysis"]["boundaries"][0])

        key_analysis = analyze_key_timeline(tokens, config=config.key_config)
        phrase_analysis = analyze_phrase_candidates(tokens, config=config.phrase_config)
        self.assertEqual(
            case["key_analysis"]["modulation_points"][0]["bar_index"],
            key_analysis.modulation_points[0].bar_index,
        )
        self.assertEqual(
            case["phrase_analysis"]["boundaries"][0]["anchor_pos"],
            phrase_analysis.boundaries[0].anchor_pos,
        )
        self.assertEqual(len(case["phrase_analysis"]["notes"]), len(phrase_analysis.notes))
        self.assertEqual(
            case["phrase_analysis"]["notes"][0]["note_index"],
            phrase_analysis.notes[0].note_index,
        )
        self.assertEqual(
            case["phrase_analysis"]["boundary_features"][0]["sequence_role"],
            phrase_analysis.boundary_features[0].sequence_role,
        )
        self.assertEqual(
            case["phrase_analysis"]["boundary_scores"][0]["motif_score"],
            phrase_analysis.boundary_scores[0].motif_score,
        )
        self.assertEqual(
            case["phrase_analysis"]["boundary_scores"][0]["subphrase_score"],
            phrase_analysis.boundary_scores[0].subphrase_score,
        )
        self.assertEqual(
            case["phrase_analysis"]["boundary_scores"][0]["phrase_score"],
            phrase_analysis.boundary_scores[0].phrase_score,
        )
        self.assertEqual(
            case["phrase_analysis"]["boundary_scores"][0]["boundary_type"],
            phrase_analysis.boundary_scores[0].boundary_type,
        )
        self.assertEqual(
            case["phrase_analysis"]["boundary_scores"][0]["sequence_role"],
            phrase_analysis.boundary_scores[0].sequence_role,
        )
        self.assertEqual(
            case["phrase_analysis"]["boundaries"][0]["source_rule"],
            "首句强制",
        )

    def test_build_debug_flags_hits_expected_rules(self) -> None:
        flags = build_debug_flags(
            key_analysis={
                "frames": [
                    {"is_uncertain": True, "margin_to_second": 0.05},
                    {"is_uncertain": False, "margin_to_second": 0.25},
                ],
                "segments": [
                    {"key": "C:maj", "length_bars": 1.5},
                    {"key": "G:maj", "length_bars": 4.0},
                ],
            },
            phrase_analysis={
                "bars": [
                    {"bar_index": 0, "rest_ratio": 0.05},
                    {"bar_index": 1, "rest_ratio": 0.10},
                    {"bar_index": 5, "rest_ratio": 0.00},
                ],
                "boundaries": [
                    {"bar_index": 0, "anchor_pos": 0},
                    {"bar_index": 1, "anchor_pos": 16},
                ],
                "phrase_spans": [
                    {"start_bar": 0, "end_bar": 5, "length_bars": 5.0},
                ],
            },
            low_margin_threshold=0.10,
            min_phrase_bars=2,
            max_phrase_bars=4,
        )
        self.assertTrue(flags["is_suspicious"])
        self.assertIn("存在 uncertain 调性帧", flags["flag_names"])
        self.assertIn("存在低置信调性帧", flags["flag_names"])
        self.assertIn("存在短调性段", flags["flag_names"])
        self.assertIn("存在密集乐句边界", flags["flag_names"])
        self.assertIn("存在超长乐句", flags["flag_names"])
        self.assertIn("存在可疑 mid-bar 乐句边界", flags["flag_names"])

    def test_write_review_bundle_outputs_index_and_lazy_case_files(self) -> None:
        case = build_review_case(
            case_id="demo-1",
            source_kind="raw_midi",
            title="demo",
            subtitle="subtitle",
            source_path="demo.mid",
            meta={},
            tokens=_c_to_g_major_tokens(),
            config=_review_config(),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "bundle"
            index_payload = write_review_bundle(
                output_dir=output_dir,
                cases=[case],
                positions_per_bar=32,
                only_suspicious=False,
                source_summary={"mode": "test"},
                include_tokens=False,
            )
            self.assertTrue((output_dir / "index.json").exists())
            self.assertEqual(len(index_payload["cases"]), 1)
            detail_path = output_dir / index_payload["cases"][0]["detail_path"]
            self.assertTrue(detail_path.exists())
            detail_payload = json.loads(detail_path.read_text(encoding="utf-8"))
            self.assertNotIn("tokens", detail_payload)
            self.assertIn("notes", detail_payload)
            self.assertIn("key_analysis", detail_payload)

    def test_fixed_viewer_files_exist_and_contain_expected_controls(self) -> None:
        project_root = _project_root()
        html_text = (project_root / "tools" / "annotation_review_viewer.html").read_text(encoding="utf-8")
        js_text = (project_root / "tools" / "annotation_review_viewer.js").read_text(encoding="utf-8")
        self.assertIn("选择数据目录", html_text)
        self.assertIn("标记剔除", html_text)
        self.assertIn("exportDecisions", js_text)
        self.assertIn("showPitchLabelsToggle", js_text)
        self.assertIn("midiPitchLabel", js_text)
        self.assertIn("webkitdirectory", html_text)
        self.assertIn("boundary-label", js_text)
        self.assertIn("source_rule", js_text)
        self.assertIn("motif_score", js_text)
        self.assertIn("subphrase_score", js_text)
        self.assertIn("phrase_score", js_text)
        self.assertIn("sequence_role", js_text)
        self.assertIn("buildBoundaryKeyMap", js_text)
        self.assertIn("finalBoundaryByKey", js_text)
        self.assertIn("allBoundaryKeys", js_text)
        self.assertIn(".sort((leftKey, rightKey) => {", js_text)
        self.assertIn("leftBar - rightBar", js_text)
        self.assertIn("leftAnchor - rightAnchor", js_text)
        self.assertEqual(js_text.count('elements.boundaryTable.innerHTML = `'), 1)

    def test_load_benchmark_cases_prefers_raw_reconstructed_tokens(self) -> None:
        config = _review_config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            benchmark_path = Path(tmp_dir) / "continuation.json"
            raw_tokens = _c_to_g_major_tokens()
            payload = {
                "task": "continuation",
                "cases": [
                    {
                        "row_id": 7,
                        "bucket": "bucket_a",
                        "meta": {"artist": "artist", "title": "title", "midi_path": "demo.mid"},
                        "prompt_tokens": ["BOS", "TEMPO_120", "BAR"],
                        "raw_reconstructed_tokens": raw_tokens,
                    }
                ],
            }
            benchmark_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            cases = load_benchmark_cases(benchmark_path, config)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]["source_kind"], "benchmark_continuation")
            self.assertEqual(cases[0]["meta"]["token_origin"], "raw_reconstructed_tokens")
            self.assertEqual(cases[0]["tokens"], raw_tokens)

    def test_script_builds_outputs_for_raw_midi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            midi_root = tmp_path / "midi"
            midi_root.mkdir(parents=True, exist_ok=True)
            output_dir = tmp_path / "review_raw"
            midi_path = midi_root / "sample.mid"
            midi = tokens_to_midi(_c_to_g_major_tokens(), TokenizerConfig(), ticks_per_beat=480)
            midi.save(str(midi_path))

            script_path = _project_root() / "scripts" / "eval" / "build_annotation_review.py"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--midi-root",
                    str(midi_root),
                    "--output-dir",
                    str(output_dir),
                    "--limit",
                    "1",
                ],
                cwd=_project_root(),
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("[annotation-review] cases=1", result.stdout)
            self.assertTrue((output_dir / "index.json").exists())
            self.assertTrue((output_dir / "cases").exists())

    def test_script_builds_outputs_for_benchmark_continuation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            benchmark_path = tmp_path / "continuation.json"
            output_dir = tmp_path / "review_benchmark"
            payload = {
                "task": "continuation",
                "checkpoint_name": "step_1.pt",
                "sample_group": "final_top3",
                "cases": [
                    {
                        "row_id": 5,
                        "bucket": "bucket_a",
                        "meta": {"artist": "artist", "title": "title", "midi_path": "demo.mid"},
                        "raw_reconstructed_tokens": _c_to_g_major_tokens(),
                    }
                ],
            }
            benchmark_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            script_path = _project_root() / "scripts" / "eval" / "build_annotation_review.py"
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--benchmark-json",
                    str(benchmark_path),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=_project_root(),
                capture_output=True,
                text=True,
                check=True,
            )
            review_payload = json.loads((output_dir / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(review_payload["cases"][0]["source_kind"], "benchmark_continuation")

    def test_script_builds_outputs_for_benchmark_infilling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            benchmark_path = tmp_path / "infilling.json"
            output_dir = tmp_path / "review_infilling"
            payload = {
                "task": "infilling",
                "checkpoint_name": "step_1.pt",
                "sample_group": "final_top3",
                "cases": [
                    {
                        "row_id": 9,
                        "bucket": "bucket_b",
                        "meta": {"artist": "artist", "title": "title", "midi_path": "demo.mid"},
                        "raw_reconstructed_tokens": _c_to_g_major_tokens(),
                    }
                ],
            }
            benchmark_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            script_path = _project_root() / "scripts" / "eval" / "build_annotation_review.py"
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--benchmark-json",
                    str(benchmark_path),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=_project_root(),
                capture_output=True,
                text=True,
                check=True,
            )
            review_payload = json.loads((output_dir / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(review_payload["cases"][0]["source_kind"], "benchmark_infilling")

    def test_apply_decisions_script_filters_split_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            midi_root = tmp_path / "midi"
            midi_root.mkdir(parents=True, exist_ok=True)
            keep_path = midi_root / "keep.mid"
            drop_path = midi_root / "drop.mid"
            keep_path.write_bytes(b"keep")
            drop_path.write_bytes(b"drop")

            split_path = tmp_path / "train.jsonl"
            output_path = tmp_path / "train_filtered.jsonl"
            split_rows = [
                {"midi_path": "keep.mid", "title": "keep"},
                {"midi_path": "drop.mid", "title": "drop"},
            ]
            split_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in split_rows) + "\n",
                encoding="utf-8",
            )
            decisions_path = tmp_path / "decisions.json"
            decisions_path.write_text(
                json.dumps(
                    {
                        "meta": {"bundle_label": "test"},
                        "decisions": [
                            {
                                "case_id": "drop-1",
                                "decision": "drop",
                                "source_path": str(drop_path.resolve()),
                                "midi_path": "drop.mid",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            script_path = _project_root() / "scripts" / "eval" / "apply_annotation_review_decisions.py"
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--decisions-json",
                    str(decisions_path),
                    "--midi-root",
                    str(midi_root),
                    "--split-jsonl",
                    str(split_path),
                    "--output-jsonl",
                    str(output_path),
                ],
                cwd=_project_root(),
                capture_output=True,
                text=True,
                check=True,
            )
            kept_lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(kept_lines), 1)
            self.assertEqual(kept_lines[0]["midi_path"], "keep.mid")

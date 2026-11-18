"""Unit tests for the GUI SFT distillation pipeline (CPU only, < 5 s).

Run from the repository root with the standard library runner (pytest is not
required and may not be installed)::

    python -m unittest tests.test_distill -v
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Optional scratch base for restricted CI hosts (a sandbox that rejects
# ``mkdtemp``'s 0o700 directory). Unset on normal machines: the standard
# ``tempfile`` path is used instead.
TEST_TMP_BASE = os.environ.get("DISTILL_TEST_TMPDIR")

from scripts import build_distill_dataset as builder  # noqa: E402
from scripts import collect_oracle  # noqa: E402
from scripts import filter_teacher  # noqa: E402
from scripts import prepare_gui_sft_data as prepare  # noqa: E402


def sha(data):
    return hashlib.sha256(data).hexdigest()


class PipelineTestCase(unittest.TestCase):
    """Shared fixture helpers: a temp workspace with a frames directory."""

    def setUp(self):
        if TEST_TMP_BASE:
            base = Path(TEST_TMP_BASE)
            base.mkdir(parents=True, exist_ok=True)
            self.root = base / ("case_%d_%s" % (os.getpid(), uuid.uuid4().hex[:8]))
            self.root.mkdir()
            self.addCleanup(shutil.rmtree, str(self.root), True)
        else:
            self._tmp = tempfile.TemporaryDirectory()
            self.root = Path(self._tmp.name)
        self.frames = self.root / "frames"
        self.frames.mkdir()
        self.episodes = self.root / "episodes"
        self.episodes.mkdir()

    def tearDown(self):
        if not TEST_TMP_BASE:
            self._tmp.cleanup()

    def frame(self, name, data=None):
        payload = data if data is not None else ("jpeg:" + name).encode("utf-8")
        path = self.frames / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return name, sha(payload)

    def row(self, frame="f0.jpg", target="<action>click(0.5,0.5)</action>", step=0,
            history=None, reward=1.0, digest=None, **extra):
        if not (self.frames / frame).exists():
            _, generated = self.frame(frame)
            if digest is None:
                digest = generated
        try:
            kind, payload = filter_teacher.parse_action_dsl(target)
        except filter_teacher.RejectRow:
            kind, payload = None, None
        row = {
            "step": step,
            "target_action": target,
            "frame": frame,
            "frame_sha256": digest or sha((self.frames / frame).read_bytes()),
            "history": list(history or []),
            "reward": reward,
        }
        if kind == "click":
            row["target_point"] = [payload["x"], payload["y"]]
        row.update(extra)
        return row

    def episode(self, task="click-dialog", seed=101, rows=None, success=True,
                status="success", reward=1.0, instruction="Click the button.",
                episode_id=None):
        return {
            "episode_id": episode_id or "%s_%d" % (task, seed),
            "task": task,
            "seed": seed,
            "instruction": instruction,
            "success": success,
            "status": status,
            "reward": reward,
            "rows": rows if rows is not None else [
                self.row("ep_%s_%d_0.jpg" % (task, seed), step=0),
                self.row("ep_%s_%d_1.jpg" % (task, seed), step=1,
                         target="<action>stop(done)</action>"),
            ],
        }

    def write_episode(self, episode, name=None):
        path = self.episodes / (name or "ep_%s.json" % episode["episode_id"])
        path.write_text(json.dumps(episode), encoding="utf-8")
        return path

    def canonical(self, task="big", seed=1, step=0,
                  action="<action>click(0.1000,0.2000)</action>",
                  instruction="Click the button.", frame="canon.jpg", digest=None,
                  episode_id=None, data_source="v11_teacher", split="train",
                  reward=1.0, success=True, history=None):
        kind, payload = filter_teacher.parse_action_dsl(action, instruction)
        if digest is None:
            if not (self.frames / frame).exists():
                digest = self.frame(frame)[1]
            else:
                digest = sha((self.frames / frame).read_bytes())
        record = {
            "episode_id": episode_id or "%s_%d" % (task, seed),
            "step": step,
            "task": task,
            "seed": seed,
            "instruction": instruction,
            "frame": frame,
            "frame_sha256": digest,
            "history": list(history or []),
            "error_feedback": None,
            "target_action": action,
            "action_kind": kind,
            "data_source": data_source,
            "split": split,
            "reward": reward,
            "success": success,
            "target_point": [payload["x"], payload["y"]] if kind == "click" else None,
        }
        self.assertEqual(builder.check_record(record), [])
        return record


class TeacherFilterTest(PipelineTestCase):
    def test_keeps_only_verified_successes_with_parseable_actions(self):
        good = self.episode()
        failed = self.episode(task="failed-task", seed=201, success=False,
                              status="timeout", reward=0.0)
        bad_dsl = self.episode(task="bad-dsl", seed=202, rows=[
            self.row("bad_0.jpg", target="<action>click(1.4,0.2)</action>"),
            self.row("bad_1.jpg", target="click(0.5,0.5)", step=1),
        ])
        missing_frame = self.episode(task="missing-frame", seed=203, rows=[
            {"step": 0, "target_action": "<action>click(0.2,0.2)</action>",
             "frame": "does-not-exist.jpg",
             "frame_sha256": "0" * 64, "history": [], "reward": 1.0,
             "target_point": [0.2, 0.2]},
        ])
        bad_hash = self.episode(task="bad-hash", seed=204, rows=[
            self.row("bhash.jpg", digest="f" * 64),
        ])
        for episode in (good, failed, bad_dsl, missing_frame, bad_hash):
            self.write_episode(episode)

        result = filter_teacher.filter_directory(
            self.episodes, self.frames, max_per_task=0
        )
        kept = result["kept"]
        self.assertEqual(len(kept), 2)
        self.assertTrue(all(row["episode_id"] == "click-dialog_101" for row in kept))
        self.assertTrue(all(row["success"] is True for row in kept))
        self.assertEqual([row["step"] for row in kept], [0, 1])
        self.assertEqual([row["action_kind"] for row in kept], ["click", "stop"])
        self.assertEqual(kept[0]["target_point"], [0.5, 0.5])
        self.assertIsNone(kept[1]["target_point"])

        reasons = {item["reason"] for item in result["rejected"]}
        self.assertIn("ep_failed", reasons)
        self.assertIn("dsl_coordinate", reasons)
        self.assertIn("dsl_syntax", reasons)
        self.assertIn("missing_frame", reasons)
        self.assertIn("frame_hash_mismatch", reasons)
        self.assertEqual(result["stats"]["episodes"]["verified_success"], 4)
        self.assertEqual(result["stats"]["rows"]["kept"], 2)

    def test_filter_is_deterministic_and_writes_stats(self):
        self.write_episode(self.episode(task="b-task", seed=2))
        self.write_episode(self.episode(task="a-task", seed=1))
        first = filter_teacher.filter_directory(self.episodes, self.frames)
        second = filter_teacher.filter_directory(self.episodes, self.frames)
        self.assertEqual(first["kept"], second["kept"])
        self.assertEqual(
            [row["task"] for row in first["kept"]], ["a-task", "a-task", "b-task", "b-task"]
        )
        stats, files = filter_teacher.write_outputs(first, self.root / "filtered")
        self.assertEqual(stats["rows"]["kept"], 4)
        self.assertIn("kept.jsonl", files)
        self.assertEqual(files["kept.jsonl"]["rows"], 4)
        self.assertTrue((self.root / "filtered" / "rejected.jsonl").exists())

    def test_rejects_out_of_range_and_non_finite_coordinates(self):
        for action, reason in (
            ("<action>click(1.2,0.5)</action>", "dsl_coordinate"),
            ("<action>scroll(0.0,2.0)</action>", "dsl_coordinate"),
            ("<action>click(nan,0.5)</action>", "dsl_syntax"),
            ("<action>click(inf,0.5)</action>", "dsl_syntax"),
            ("<action>tap(0.1,0.1)</action>", "dsl_syntax"),
        ):
            with self.subTest(action=action):
                with self.assertRaises(filter_teacher.RejectRow) as caught:
                    filter_teacher.parse_action_dsl(action)
                self.assertEqual(caught.exception.reason, reason)


class DedupTest(PipelineTestCase):
    def test_dedup_key_collapses_duplicates_and_keeps_first_row(self):
        duplicate = "<action>click(0.3,0.4)</action>"
        first = self.canonical(seed=7, step=0, action=duplicate, episode_id="first")
        second = self.canonical(seed=8, step=1, action=duplicate, episode_id="second")
        third = self.canonical(seed=9, step=0, action="<action>click(0.9,0.9)</action>",
                               episode_id="third")
        self.assertEqual(builder.dedup_key(first), builder.dedup_key(second))
        self.assertNotEqual(builder.dedup_key(first), builder.dedup_key(third))

        result = builder.build_dataset(
            {"v11_teacher": [second, first, third]},
            frames_root=None,
            check_frames=False,
            blocked_seeds=set(),
            easy_cap=1000,
            val_divisor=0,
        )
        rows = result["train"] + result["validation"]
        self.assertEqual(len(rows), 2)
        kept = [row for row in rows if row["target_action"] == duplicate]
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["episode_id"], "first")
        self.assertEqual(result["manifest"]["stages"]["after_dedup"]["removed"], 1)
        self.assertEqual(result["manifest"]["dedup"]["duplicates_removed"], 1)
        self.assertEqual(len(result["manifest"]["dedup"]["hash"]), 64)

    def test_prepare_dedup_keeps_first_deterministic_row(self):
        first = self.canonical(seed=1, episode_id="ep-a")
        second = self.canonical(seed=2, episode_id="ep-b")
        unique, duplicates = prepare.dedup_records([second, first])
        self.assertEqual(duplicates, 1)
        self.assertEqual([row["episode_id"] for row in unique], ["ep-a"])


class BalancingTest(PipelineTestCase):
    def test_balancing_respects_easy_cap_and_exempts_hard_tasks(self):
        big = [
            self.canonical(task="big", seed=100 + index, step=index,
                           action="<action>click(0.%04d,0.5)</action>" % index,
                           episode_id="big-%d" % index, frame="big_%d.jpg" % index)
            for index in range(6)
        ]
        hard = [
            self.canonical(task="click-link", seed=200 + index,
                           action="<action>click(0.%04d,0.6)</action>" % index,
                           episode_id="hard-%d" % index, frame="hard_%d.jpg" % index)
            for index in range(3)
        ]
        sources = {"v11_teacher": big + hard}

        result = builder.build_dataset(
            sources,
            frames_root=None,
            check_frames=False,
            blocked_seeds=set(),
            easy_cap=3,
            hard_tasks=("click-link",),
            val_divisor=0,
        )
        per_task = result["manifest"]["stages"]["after_balance"]["per_task"]
        self.assertEqual(per_task["big"]["before"], 6)
        self.assertEqual(per_task["big"]["after"], 3)
        self.assertEqual(per_task["big"]["capped_by"], "easy_cap")
        self.assertEqual(per_task["click-link"]["after"], 3)
        self.assertIsNone(per_task["click-link"]["capped_by"])
        self.assertEqual(result["manifest"]["stages"]["after_balance"]["removed"], 3)

        again = builder.build_dataset(
            sources,
            frames_root=None,
            check_frames=False,
            blocked_seeds=set(),
            easy_cap=3,
            hard_tasks=("click-link",),
            val_divisor=0,
        )
        self.assertEqual(
            [row["episode_id"] for row in result["train"]],
            [row["episode_id"] for row in again["train"]],
        )


class SplitIsolationTest(PipelineTestCase):
    def test_frozen_test_episode_is_excluded_from_build_splits(self):
        frozen_record = self.canonical(
            task="frozen-task", seed=999, episode_id="frozen-ep", frame="frozen.jpg"
        )
        frozen_path = self.root / "frozen.json"
        frozen_path.write_text(
            json.dumps(
                {
                    "tasks": [],
                    "seeds": [frozen_record["seed"]],
                    "episodes": [frozen_record["episode_id"]],
                    "records": [frozen_record],
                }
            ),
            encoding="utf-8",
        )
        frozen = builder.load_frozen_test(frozen_path)
        self.assertEqual(frozen["seeds"], {999})
        self.assertEqual(frozen["episodes"], {"frozen-ep"})

        conflicting = self.canonical(
            task="train-task", seed=999, episode_id="leaky-ep", frame="leaky.jpg"
        )
        clean = self.canonical(task="train-task", seed=5, episode_id="clean-ep",
                               frame="clean.jpg")
        result = builder.build_dataset(
            {"v11_teacher": [conflicting, clean]},
            frames_root=None,
            check_frames=False,
            blocked_seeds=set(),
            easy_cap=1000,
            val_divisor=0,
            frozen=frozen,
        )
        train_ids = {row["episode_id"] for row in result["train"]}
        validation_ids = {row["episode_id"] for row in result["validation"]}
        self.assertNotIn("leaky-ep", train_ids | validation_ids)
        self.assertIn("clean-ep", train_ids)
        self.assertEqual(result["manifest"]["stages"]["after_frozen_test"]["removed"], 1)
        self.assertEqual([row["episode_id"] for row in result["test"]], ["frozen-ep"])

        # A frozen test record must never be able to reach train/validation.
        with self.assertRaises(prepare.IsolationError):
            prepare.assert_isolated(
                [dict(frozen_record, split="train")], [], result["test"]
            )

    def test_package_keeps_test_split_separate_and_asserts_isolation(self):
        test_record = self.canonical(task="frozen-task", seed=777,
                                     episode_id="test-ep", frame="test.jpg")
        frozen_path = self.root / "frozen.json"
        frozen_path.write_text(
            json.dumps({"seeds": [777], "episodes": ["test-ep"],
                        "records": [test_record]}),
            encoding="utf-8",
        )
        frozen = builder.load_frozen_test(frozen_path)
        train_records = [
            self.canonical(task="train-%d" % index, seed=10 + index,
                           episode_id="ep-%d" % index, frame="train_%d.jpg" % index)
            for index in range(4)
        ]
        result = prepare.package(
            train_records, frozen=frozen, split_seed=3, val_divisor=4, frames_root=None
        )
        train_ids = {row["episode_id"] for row in result["train"]}
        validation_ids = {row["episode_id"] for row in result["validation"]}
        self.assertNotIn("test-ep", train_ids | validation_ids)
        self.assertEqual([row["episode_id"] for row in result["test"]], ["test-ep"])
        self.assertEqual(len(result["train"]) + len(result["validation"]), 4)
        self.assertEqual(result["manifest"]["isolation"]["checked"], True)
        manifest = prepare.write_outputs(result, self.root / "sft")
        self.assertEqual(manifest["counts"]["train"], len(result["train"]))
        self.assertEqual(manifest["counts"]["validation"], len(result["validation"]))
        self.assertTrue((self.root / "sft" / "test.jsonl").exists())


class ManifestCountsTest(PipelineTestCase):
    def test_manifest_counts_add_up_on_scaled_down_synthetic_data(self):
        # Scaled-down analogue of 3010 - 60 - 35 - 132 = 2783:
        # 40 raw -> 35 after seed block -> 33 after dedup -> 19 after balance.
        v11 = []
        for index in range(20):
            v11.append(self.canonical(task="big", seed=100 + index, step=index,
                                      action="<action>click(0.%04d,0.3)</action>" % index,
                                      episode_id="big-%d" % index,
                                      frame="big_%d.jpg" % index))
        for index in range(5):
            v11.append(self.canonical(task="blocked", seed=14101 + index,
                                      action="<action>click(0.%04d,0.4)</action>" % index,
                                      episode_id="blocked-%d" % index,
                                      frame="blocked_%d.jpg" % index))
        for index in range(5):
            v11.append(self.canonical(task="small", seed=200 + index,
                                      action="<action>click(0.%04d,0.5)</action>" % index,
                                      episode_id="small-%d" % index,
                                      frame="small_%d.jpg" % index))
        oracle = []
        for index in range(6):
            oracle.append(self.canonical(task="big", seed=300 + index,
                                         action="<action>click(0.%04d,0.6)</action>" % index,
                                         episode_id="oracle-%d" % index,
                                         frame="oracle_%d.jpg" % index,
                                         data_source="v14_oracle"))
        gui_owl = []
        for index in range(4):
            gui_owl.append(self.canonical(task="small", seed=400 + index,
                                          action="<action>click(0.%04d,0.7)</action>" % index,
                                          episode_id="owl-%d" % index,
                                          frame="owl_%d.jpg" % index,
                                          data_source="gui_owl"))
        # Two GUI-Owl rows repeat an existing (frame, action, instruction) key.
        gui_owl.append(dict(v11[0], data_source="gui_owl", episode_id="dup-a"))
        gui_owl.append(dict(v11[1], data_source="gui_owl", episode_id="dup-b"))
        self.assertEqual(len(v11), 30)
        self.assertEqual(len(oracle), 6)
        self.assertEqual(len(gui_owl), 6)

        result = builder.build_dataset(
            {"v11_teacher": v11, "v14_oracle": oracle, "gui_owl": gui_owl},
            frames_root=None,
            check_frames=False,
            easy_cap=10,
            hard_tasks=(),
            val_divisor=4,
            split_seed=14,
        )
        stages = result["manifest"]["stages"]
        self.assertEqual(stages["raw"]["total"], 42)
        self.assertEqual(stages["after_seed_block"]["total"], 37)
        self.assertEqual(stages["after_dedup"]["total"], 35)
        self.assertEqual(stages["after_balance"]["total"], 19)
        self.assertEqual(stages["after_seed_block"]["removed"], 5)
        self.assertEqual(stages["after_dedup"]["removed"], 2)
        self.assertEqual(stages["after_balance"]["removed"], 16)
        self.assertEqual(stages["raw"]["total"]
                         - stages["after_seed_block"]["removed"]
                         - stages["after_dedup"]["removed"]
                         - stages["after_balance"]["removed"],
                         stages["after_balance"]["total"])
        self.assertEqual(result["manifest"]["unique_rows"], 35)
        self.assertEqual(result["manifest"]["train_rows"], 15)
        self.assertEqual(result["manifest"]["validation_rows"], 4)
        self.assertEqual(len(result["train"]) + len(result["validation"]), 19)
        self.assertTrue(result["manifest"]["counts_are_unique_rows_not_training_rows"])
        self.assertIn("NOT the number of rows trained on",
                      result["manifest"]["unique_rows_note"])

        manifest = builder.write_dataset(result, self.root / "built")
        for name in ("train.jsonl", "validation.jsonl"):
            self.assertIn(name, manifest["files"])
            self.assertEqual(len(manifest["files"][name]["sha256"]), 64)
            self.assertEqual(
                manifest["files"][name]["rows"],
                len((self.root / "built" / name).read_text(encoding="utf-8").splitlines()),
            )
        written = json.loads((self.root / "built" / "manifest.json").read_text("utf-8"))
        self.assertEqual(written["unique_rows"], 35)


class CopyPointerTest(PipelineTestCase):
    def test_copy_pointer_round_trip(self):
        instruction = "Type the word MINIMAP into the box."
        start = instruction.index("MINIMAP")
        end = start + len("MINIMAP")

        pointer = builder.to_type_pointer(instruction, "MINIMAP")
        self.assertEqual(pointer, "<action>type(@%d:%d)</action>" % (start, end))
        self.assertEqual(builder.apply_type_pointer(instruction, pointer), "MINIMAP")

        lower = builder.to_type_pointer("Type minimap into the box.", "MINIMAP")
        self.assertIsNotNone(lower)
        self.assertTrue(lower.endswith("|upper)</action>"), lower)
        self.assertEqual(
            builder.apply_type_pointer("Type minimap into the box.", lower), "MINIMAP"
        )

        upper_source = "Type MINIMAP into the box."
        upper = builder.to_type_pointer(upper_source, "minimap")
        self.assertTrue(upper.endswith("|lower)</action>"), upper)
        self.assertEqual(builder.apply_type_pointer(upper_source, upper), "minimap")

        self.assertIsNone(builder.to_type_pointer(instruction, "zebra"))

        record = self.canonical(
            task="enter-text",
            seed=11,
            action=pointer,
            instruction=instruction,
            frame="pointer.jpg",
        )
        self.assertEqual(record["action_kind"], "type")
        self.assertIsNone(record["target_point"])
        self.assertEqual(builder.apply_type_pointer(instruction, record["target_action"]),
                         "MINIMAP")

    def test_collector_is_import_safe_without_browsergym(self):
        self.assertTrue(callable(collect_oracle.build_plan))
        self.assertTrue(callable(collect_oracle.step_action))
        self.assertEqual(collect_oracle.build_plan("unknown-task", "goal", None)[0], None)


if __name__ == "__main__":
    unittest.main()

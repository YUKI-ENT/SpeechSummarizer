import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from session_store import SessionIndex, decode_chart_id, parse_session_name


class SessionIndexTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.outputs = Path(directory.name) / "sessions"
        self.llm = Path(directory.name) / "llm"
        self.outputs.mkdir()
        self.llm.mkdir()
        self.index = SessionIndex(self.outputs, self.llm)

    def session(self, stem, events=()):
        (self.outputs / (stem + ".txt")).write_text("ASR本文はファイルのみ", encoding="utf-8")
        path = self.outputs / (stem + ".jsonl")
        path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        return path

    def test_emr_identity_is_decoded_only_for_branch_input(self):
        self.assertEqual(decode_chart_id("123450", True), {
            "patient_id": "12345", "chart_branch": 0, "source_chart_id": "123450"})
        self.assertEqual(decode_chart_id("123450", False)["patient_id"], "123450")
        self.assertIsNone(decode_chart_id("123450", False)["chart_branch"])
        for value in ("1", "１２３", "../123", "", "12.3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode_chart_id(value, True)

    def test_old_branch_and_collision_names_have_same_patient(self):
        for suffix, branch in (("", None), ("_b0", 0), ("_b9", 9)):
            for collision in ("", "_c" + "a" * 32):
                name = f"12345{suffix}_20261115_090000{collision}.txt"
                self.assertEqual(parse_session_name(name)["patient_id"], "12345")
                self.assertEqual(parse_session_name(name)["chart_branch"], branch)
        self.assertIsNone(parse_session_name("../123_20261115_090000.txt"))

    def test_migration_shows_all_branches_and_keeps_unknown_separate_from_zero(self):
        for suffix in ("", "_b0", "_b1"):
            self.session(f"12345{suffix}_20261115_090000", [
                {"type": "asr", "text": "Do not store this transcript in SQLite"}])
        self.session("999_b1_20261115_100000")
        self.index.reconcile()
        self.assertEqual({row["chart_branch"] for row in self.index.sessions("12345")}, {None, 0, 1})
        for branch, suffix in ((None, ""), (0, "_b0"), (1, "_b1")):
            self.assertEqual(self.index.latest_today("12345", branch, "20261115"),
                             f"12345{suffix}_20261115_090000.txt")
        self.assertIsNone(self.index.latest_today("12345", 2, "20261115"))
        with self.index.connection() as db:
            self.assertNotIn("Do not store this transcript", "\n".join(db.iterdump()))

    def test_approvals_and_llm_files_are_isolated_by_branch_even_at_same_time(self):
        stems = [f"12345_b{branch}_20261115_090000" for branch in (0, 1)]
        for stem in stems:
            filename = stem + "__model__soap__20261115_090001.txt"
            self.session(stem, [{"type": "approved", "llm_file": filename}])
            (self.llm / filename).write_text("summary", encoding="utf-8")
        self.index.reconcile()
        for row in self.index.sessions("12345"):
            self.assertEqual(row["llm_files"], [row["approved_file"]])
            self.assertEqual(self.index.llm_history(session=row["name"]), row["llm_files"])
        self.assertEqual(len(self.index.recent_patients()), 1)

    def test_normal_queries_never_scan_directories_or_read_transcripts(self):
        self.session("12345_b1_20261115_090000")
        self.index.reconcile()
        with patch.object(Path, "glob", side_effect=AssertionError("directory scan")), \
             patch.object(Path, "open", side_effect=AssertionError("file read")):
            self.assertEqual(len(self.index.sessions("12345")), 1)
            self.assertEqual(len(self.index.recent_patients()), 1)
            self.assertTrue(self.index.latest_today("12345", 1, "20261115"))
            self.assertEqual(self.index.llm_history(patient_id="12345"), [])

    def test_restart_updates_changed_files_removes_deleted_and_rebuilds_from_files(self):
        path = self.session("12345_b1_20261115_090000", [{"type": "approved", "llm_file": "old"}])
        deleted = self.session("999_20261115_090000")
        self.index.reconcile()
        with path.open("a", encoding="utf-8") as stream:
            stream.write('broken line\n{"type":"approved","llm_file":"new"}\n')
        deleted.with_suffix(".txt").unlink()
        restarted = SessionIndex(self.outputs, self.llm)
        restarted.reconcile()
        self.assertEqual(restarted.sessions("12345")[0]["approved_file"], "new")
        self.assertEqual(restarted.sessions("999"), [])
        restarted.path.unlink()
        rebuilt = SessionIndex(self.outputs, self.llm)
        rebuilt.reconcile()
        self.assertEqual(rebuilt.sessions("12345")[0]["approved_file"], "new")

    def test_unchanged_restart_skips_jsonl_reads_and_live_events_update_index(self):
        path = self.session("12345_b0_20261115_090000")
        self.index.reconcile()
        with patch.object(Path, "open", side_effect=AssertionError("unchanged JSONL read")):
            self.index.reconcile()
        event = {"type": "approved", "llm_file": path.stem + "__m__soap__20261115_090001.txt"}
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        self.index.record_event(path, event)
        summary = self.llm / event["llm_file"]
        summary.write_text("summary", encoding="utf-8")
        self.index.record_llm(summary)
        row = self.index.sessions("12345")[0]
        self.assertEqual(row["approved_file"], summary.name)
        self.assertEqual(row["llm_files"], [summary.name])


if __name__ == "__main__":
    unittest.main()

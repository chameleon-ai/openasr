from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from select_release_matrix import (
    MATRIX_PATH,
    is_windows_release_row,
    load_matrix,
    select_matrix,
    without_windows_rows,
)


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "release-binaries.yml"
SCRIPT = Path(__file__).resolve().parent / "select_release_matrix.py"


class SelectReleaseMatrixTests(unittest.TestCase):
    def test_empty_only_target_returns_the_full_matrix(self) -> None:
        rows = load_matrix()
        selected = select_matrix(rows, "")
        self.assertEqual(selected, rows)
        self.assertGreater(len(selected), 1)

    def test_exact_target_returns_one_row(self) -> None:
        selected = select_matrix(load_matrix(), "x86_64-apple-darwin")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["target"], "x86_64-apple-darwin")
        self.assertEqual(selected[0]["os"], "macos-15-intel")

    def test_skip_windows_drops_every_pc_windows_msvc_row(self) -> None:
        rows = load_matrix()
        windows = [row for row in rows if is_windows_release_row(row)]
        self.assertGreater(len(windows), 0)
        kept = without_windows_rows(rows)
        self.assertEqual(len(kept), len(rows) - len(windows))
        for row in kept:
            self.assertNotIn("pc-windows-msvc", row.get("target", ""))
        self.assertTrue(all("pc-windows-msvc" in row.get("target", "") for row in windows))

    def test_skip_windows_with_only_target_keeps_single_non_windows_row(self) -> None:
        selected = select_matrix(load_matrix(), "x86_64-apple-darwin", True)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["target"], "x86_64-apple-darwin")

    def test_skip_windows_with_windows_only_target_fails_closed(self) -> None:
        with self.assertRaises(SystemExit):
            select_matrix(load_matrix(), "x86_64-pc-windows-msvc-neutral", True)

    def test_cli_skip_windows_prints_compact_json_without_windows_rows(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--skip-windows"],
            check=True,
            capture_output=True,
            text=True,
        )
        parsed = json.loads(completed.stdout)
        self.assertEqual(completed.stdout, json.dumps(parsed, separators=(",", ":")))
        self.assertGreater(len(parsed), 0)
        for row in parsed:
            self.assertNotIn("pc-windows-msvc", row.get("target", ""))

    def test_unknown_target_fails_closed(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            select_matrix(load_matrix(), "not-a-real-target")
        message = str(raised.exception)
        self.assertIn("not-a-real-target", message)
        self.assertIn("x86_64-apple-darwin", message)

    def test_cli_prints_only_compact_json(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--only-target", ""],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stderr, "")
        parsed = json.loads(completed.stdout)
        self.assertIsInstance(parsed, list)
        self.assertEqual(completed.stdout, json.dumps(parsed, separators=(",", ":")))
        self.assertNotIn("\n", completed.stdout)
        self.assertEqual(len(parsed), len(load_matrix()))

    def test_cli_unknown_target_exits_one_without_json(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--only-target", "not-a-real-target"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stdout, "")
        self.assertIn("not-a-real-target", completed.stderr)
        self.assertIn("x86_64-apple-darwin", completed.stderr)

    def test_workflow_wires_selected_json_into_build_include(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertTrue(MATRIX_PATH.is_file())
        self.assertIsInstance(json.loads(MATRIX_PATH.read_text(encoding="utf-8")), list)
        self.assertIn("select_release_matrix.py", workflow)
        self.assertIn("fromJSON(needs.select-matrix.outputs.include)", workflow)
        self.assertIn("\n  select-matrix:\n", workflow)

    def test_linux_release_legs_pin_digest_containers(self) -> None:
        rows = {row["target"]: row for row in load_matrix()}
        self.assertEqual(
            rows["x86_64-unknown-linux-gnu"]["container"],
            "ghcr.io/chameleon-ai/openasr-ci-linux@sha256:0ad0dcc8814c5eb10a951f1a718f1339f8e96e1936430df35d8db15a450b49db",
        )
        self.assertEqual(
            rows["x86_64-unknown-linux-gnu-cuda"]["container"],
            "ghcr.io/chameleon-ai/openasr-ci-linux-cuda@sha256:4fe6e1ed567f74cea78233cde0b58608f683e8f4960ce58d7fc248150e26b955",
        )
        self.assertEqual(
            rows["x86_64-unknown-linux-gnu-rocm"]["container"],
            "ghcr.io/chameleon-ai/openasr-ci-linux-rocm@sha256:ddc8528d496368420f80104eac2fd0f315b563a08483330902fe86d54eeb1998",
        )
        self.assertNotIn("container", rows["x86_64-unknown-linux-gnu-vulkan"])
        self.assertNotIn("container", rows["aarch64-unknown-linux-gnu"])
        self.assertNotIn("container", rows["x86_64-pc-windows-msvc-neutral"])


if __name__ == "__main__":
    unittest.main()

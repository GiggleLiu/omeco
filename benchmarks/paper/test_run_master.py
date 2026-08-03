#!/usr/bin/env python3
"""Tests for the current-master paper benchmark gate."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("run_master.py")
SPEC = importlib.util.spec_from_file_location("run_master", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
run_master = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_master)


def run(*args: str, cwd: Path) -> None:
    subprocess.run(list(args), cwd=str(cwd), check=True, capture_output=True)


class MasterGateTests(unittest.TestCase):
    def make_repo(self, root: Path) -> Path:
        remote = root / "remote.git"
        checkout = root / "checkout"
        run("git", "init", "--bare", str(remote), cwd=root)
        run("git", "init", str(checkout), cwd=root)
        run("git", "branch", "-M", "master", cwd=checkout)
        run("git", "config", "user.email", "paper-gate@example.invalid", cwd=checkout)
        run("git", "config", "user.name", "Paper Gate Test", cwd=checkout)
        (checkout / "tracked.txt").write_text("tracked\n")
        run("git", "add", "tracked.txt", cwd=checkout)
        run("git", "commit", "-m", "initial", cwd=checkout)
        run("git", "remote", "add", "origin", str(remote), cwd=checkout)
        run("git", "push", "-u", "origin", "master", cwd=checkout)
        return checkout

    def test_verify_master_accepts_clean_and_rejects_untracked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = self.make_repo(Path(directory))
            revision = run_master.verify_master(checkout)
            self.assertEqual(revision, run_master.git(checkout, "rev-parse", "HEAD"))
            (checkout / "build.rs").write_text("fn main() {}\n")
            with self.assertRaisesRegex(run_master.GateError, "not clean"):
                run_master.verify_master(checkout, fetch=False)

    def test_verify_master_rejects_non_master_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = self.make_repo(Path(directory))
            (checkout / "tracked.txt").write_text("changed\n")
            run("git", "commit", "-am", "local-only", cwd=checkout)
            with self.assertRaisesRegex(run_master.GateError, "is not current"):
                run_master.verify_master(checkout, fetch=False)

    def test_input_bundle_is_order_independent_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.json"
            second = root / "b.json"
            first.write_text("a\n")
            second.write_text("b\n")
            inputs = [("a.json", first), ("b.json", second)]
            digest, entries = run_master.input_bundle(inputs)
            reverse_digest, reverse_entries = run_master.input_bundle(list(reversed(inputs)))
            self.assertEqual(digest, reverse_digest)
            self.assertEqual(entries, reverse_entries)
            second.write_text("changed\n")
            changed_digest, _ = run_master.input_bundle(inputs)
            self.assertNotEqual(digest, changed_digest)

    def test_load_inputs_requires_tracked_master_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = self.make_repo(Path(directory))
            instance = checkout / "instance.json"
            manifest = checkout / "manifest.json"
            instance.write_text("{}\n")
            manifest.write_text(
                json.dumps(
                    {
                        "sets": {
                            "paper": {
                                "instances": [
                                    {"name": "one", "path": "instance.json"}
                                ]
                            }
                        }
                    }
                )
            )
            with self.assertRaisesRegex(run_master.GateError, "not tracked"):
                run_master.load_inputs(checkout, manifest, "paper", checkout)
            run("git", "add", "manifest.json", "instance.json", cwd=checkout)
            run("git", "commit", "-m", "add inputs", cwd=checkout)
            manifest_rel, inputs = run_master.load_inputs(
                checkout, manifest, "paper", checkout
            )
            self.assertEqual(manifest_rel, "manifest.json")
            self.assertEqual([path for path, _ in inputs], ["instance.json"])

    def test_write_atomic_replaces_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "record.json"
            output.write_bytes(b"old")
            run_master.write_atomic(output, b"new\n")
            self.assertEqual(output.read_bytes(), b"new\n")
            self.assertEqual(list(output.parent.glob(".record.json.*.tmp")), [])

    def test_cli_publishes_artifact_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = self.make_repo(root)
            paper = checkout / "benchmarks/paper"
            paper.mkdir(parents=True)
            shutil.copyfile(MODULE_PATH, paper / "run_master.py")
            (checkout / ".gitignore").write_text("/target/\n")
            (checkout / "instance.json").write_text("{}\n")
            (checkout / "manifest.json").write_text(
                json.dumps(
                    {
                        "sets": {
                            "paper": {
                                "instances": [
                                    {"name": "one", "path": "instance.json"}
                                ]
                            }
                        }
                    }
                )
            )
            fake_cargo = checkout / "fake_cargo.py"
            fake_cargo.write_text(
                """#!/usr/bin/env python3
from pathlib import Path

runner = Path("target/release/examples/paper_bench")
runner.parent.mkdir(parents=True, exist_ok=True)
runner.write_text('''#!/usr/bin/env python3
import json
from pathlib import Path
import sys

args = sys.argv[1:]
out = Path(args[args.index("--out") + 1])
set_name = args[args.index("--set") + 1]
out.write_text(json.dumps({"format": 2, "set": set_name, "results": []}) + "\\\\n")
''')
runner.chmod(0o755)
"""
            )
            fake_cargo.chmod(0o755)
            run("git", "add", ".", cwd=checkout)
            run("git", "commit", "-m", "add paper gate fixture", cwd=checkout)
            run("git", "push", "origin", "master", cwd=checkout)

            output = root / "paper.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(paper / "run_master.py"),
                    "--manifest",
                    str(checkout / "manifest.json"),
                    "--set",
                    "paper",
                    "--out",
                    str(output),
                    "--repo-root",
                    str(checkout),
                    "--cargo",
                    str(fake_cargo),
                ],
                cwd=str(checkout),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            provenance = Path(f"{output}.provenance.json")
            record = json.loads(provenance.read_text())
            self.assertEqual(record["format"], 1)
            self.assertEqual(record["set"], "paper")
            self.assertEqual(record["revision"], run_master.git(checkout, "rev-parse", "HEAD"))
            self.assertEqual(record["output"]["sha256"], run_master.sha256_file(output))
            self.assertEqual(record["inputs"][0]["path"], "instance.json")
            self.assertEqual(len(record["binary_sha256"]), 64)
            self.assertEqual(len(record["manifest"]["sha256"]), 64)
            self.assertEqual(len(record["inputs_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()

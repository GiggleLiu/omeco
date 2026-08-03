#!/usr/bin/env python3
"""Run a paper benchmark from a clean, current ``origin/master`` checkout."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Dict, List, Sequence, Tuple


FORMAT = 1
REPO = Path(__file__).resolve().parents[2]


class GateError(RuntimeError):
    """A paper run failed its provenance gate."""


def fail(message: str) -> None:
    raise GateError(message)


def command(
    argv: Sequence[str], cwd: Path, *, capture: bool = False
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            list(argv),
            cwd=str(cwd),
            check=True,
            capture_output=capture,
            text=capture,
        )
    except FileNotFoundError as exc:
        fail(f"command not found: {argv[0]}")
        raise AssertionError("unreachable") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip() if capture else ""
        suffix = f": {detail}" if detail else ""
        fail(f"command failed ({' '.join(argv)}){suffix}")
        raise AssertionError("unreachable") from exc


def git(repo: Path, *args: str) -> str:
    result = command(["git", *args], repo, capture=True)
    return result.stdout.strip()


def verify_master(repo: Path, *, fetch: bool = True) -> str:
    if fetch:
        command(
            [
                "git",
                "fetch",
                "--quiet",
                "origin",
                "+refs/heads/master:refs/remotes/origin/master",
            ],
            repo,
            capture=True,
        )
    head = git(repo, "rev-parse", "HEAD")
    master = git(repo, "rev-parse", "refs/remotes/origin/master")
    if head != master:
        fail(f"HEAD {head} is not current origin/master {master}")
    dirty = git(repo, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        fail(f"checkout is not clean:\n{dirty}")
    return head


def repo_relative(repo: Path, path: Path, label: str) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo.resolve())
    except ValueError:
        fail(f"{label} is outside the master checkout: {path}")
    return relative.as_posix()


def require_tracked(repo: Path, paths: Sequence[str]) -> None:
    for path in paths:
        try:
            command(
                ["git", "ls-files", "--error-unmatch", "--", path],
                repo,
                capture=True,
            )
        except GateError:
            fail(f"paper input is not tracked on master: {path}")


def load_inputs(
    repo: Path, manifest_path: Path, set_name: str, repo_root: Path
) -> Tuple[str, List[Tuple[str, Path]]]:
    manifest_rel = repo_relative(repo, manifest_path, "manifest")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read manifest {manifest_path}: {exc}")
    sets = manifest.get("sets")
    if not isinstance(sets, dict) or set_name not in sets:
        fail(f"manifest has no set {set_name!r}")
    instances = sets[set_name].get("instances")
    if not isinstance(instances, list) or not instances:
        fail(f"set {set_name!r} has no instances")
    inputs: List[Tuple[str, Path]] = []
    seen = set()
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict) or not isinstance(instance.get("path"), str):
            fail(f"set {set_name!r} instance {index} has no string path")
        path = (repo_root / instance["path"]).resolve()
        relative = repo_relative(repo, path, f"instance {index}")
        if relative in seen:
            continue
        if not path.is_file():
            fail(f"instance does not exist: {relative}")
        seen.add(relative)
        inputs.append((relative, path))
    inputs.sort(key=lambda item: item[0])
    require_tracked(repo, [manifest_rel, *[relative for relative, _ in inputs]])
    return manifest_rel, inputs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        fail(f"cannot hash {path}: {exc}")
    return digest.hexdigest()


def input_bundle(inputs: Sequence[Tuple[str, Path]]) -> Tuple[str, List[Dict[str, str]]]:
    digest = hashlib.sha256()
    digest.update(b"omeco-paper-inputs-v1\0")
    entries = []
    for relative, path in sorted(inputs, key=lambda item: item[0]):
        data = path.read_bytes()
        path_bytes = relative.encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        entries.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest()})
    return digest.hexdigest(), entries


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(data)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="benchmarks/paper/manifest.json", type=Path
    )
    parser.add_argument("--set", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument("--cargo", default=os.environ.get("CARGO", "cargo"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    repo_root = args.repo_root.resolve()
    output = args.out.resolve()
    provenance = Path(f"{output}.provenance.json")
    temporary_output = None
    try:
        revision = verify_master(REPO)
        manifest_rel, inputs = load_inputs(REPO, manifest, args.set, repo_root)
        manifest_hash = sha256_file(manifest)
        inputs_hash, input_entries = input_bundle(inputs)

        command(
            [args.cargo, "build", "--release", "--example", "paper_bench", "-p", "omeco"],
            REPO,
        )
        # A build script must not be able to leave source changes behind.
        dirty = git(REPO, "status", "--porcelain", "--untracked-files=all")
        if dirty:
            fail(f"build changed the checkout:\n{dirty}")
        binary = REPO / "target/release/examples/paper_bench"
        if sys.platform == "win32":
            binary = binary.with_suffix(".exe")
        binary_hash = sha256_file(binary)

        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
        )
        os.close(descriptor)
        os.unlink(temporary_name)
        temporary_output = Path(temporary_name)
        command(
            [
                str(binary),
                "--manifest",
                str(manifest),
                "--set",
                args.set,
                "--out",
                str(temporary_output),
                "--repo-root",
                str(repo_root),
            ],
            REPO,
        )
        output_hash = sha256_file(temporary_output)
        record = {
            "format": FORMAT,
            "revision": revision,
            "set": args.set,
            "binary_sha256": binary_hash,
            "manifest": {"path": manifest_rel, "sha256": manifest_hash},
            "inputs_sha256": inputs_hash,
            "inputs": input_entries,
            "output": {"path": output.name, "sha256": output_hash},
        }
        provenance_bytes = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
        os.replace(temporary_output, output)
        temporary_output = None
        write_atomic(provenance, provenance_bytes)
        print(f"wrote {output}")
        print(f"wrote {provenance}")
    except GateError as exc:
        raise SystemExit(f"run_master.py: FAIL: {exc}") from exc
    finally:
        if temporary_output is not None:
            try:
                temporary_output.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()

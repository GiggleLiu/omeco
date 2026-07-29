#!/usr/bin/env python3
"""Compare two paper_bench artifacts.

    python3 benchmarks/paper/check.py <fresh.json> <expected.json>

Exit 0 and print ``OK: N results compared`` when the two artifacts agree;
exit 1 with a message naming the first disagreement otherwise.

Two artifacts agree when
  * ``format`` and ``set`` are equal,
  * the ``(instance, arm)`` key sets are equal, and
  * every leaf under a matching key is equal -- strings, integers and booleans
    exactly, floats within ``TOLERANCE`` *relative* to their magnitude.

The tolerance absorbs cross-platform libm differences in the annealer's
transcendental calls. It is relative, with an absolute floor of ``TOLERANCE``,
because the fields being compared span many orders of magnitude: ``tc``/``sc``/
``rwc`` are log2 quantities of order 10, while a ``curve`` entry is a raw
annealing score of order 1e12. An absolute 1e-9 would be a sensible band for
the former and a demand for bit-exactness on the latter.

CI runs this script, because the committed artifact and the CI runner need not
share a platform. On a single machine the artifacts are bit-identical, so when
you regenerate locally ``diff`` is the stricter check and worth running too.

Standard library only, Python 3.8+.
"""

import json
import sys

TOLERANCE = 1e-9


def close_enough(left, right):
    """Relative comparison with an absolute floor, so that both small log2
    metrics and ~1e12 annealing scores get a proportionate band."""
    return abs(left - right) <= max(TOLERANCE, TOLERANCE * max(abs(left), abs(right)))


def die(message):
    print("check.py: FAIL: " + message, file=sys.stderr)
    sys.exit(1)


def load(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        die("cannot read {}: {}".format(path, exc))
    except ValueError as exc:
        die("{} is not valid JSON: {}".format(path, exc))


def compare(left, right, where):
    """Recursively compare two JSON values, reporting the first mismatch."""
    if isinstance(left, bool) or isinstance(right, bool):
        # bool is a subclass of int; check it before the numeric branch so that
        # True never compares equal to 1.
        if left is not right:
            die("{}: {!r} != {!r}".format(where, left, right))
        return

    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, int) and isinstance(right, int):
            if left != right:
                die("{}: {} != {}".format(where, left, right))
            return
        if not close_enough(left, right):
            die(
                "{}: {!r} != {!r} (differs by {:g}, relative tolerance {:g})".format(
                    where, left, right, abs(left - right), TOLERANCE
                )
            )
        return

    if type(left) is not type(right):
        die(
            "{}: type mismatch, {} != {}".format(
                where, type(left).__name__, type(right).__name__
            )
        )

    if isinstance(left, dict):
        if set(left) != set(right):
            only_left = sorted(set(left) - set(right))
            only_right = sorted(set(right) - set(left))
            die(
                "{}: key mismatch, only in first: {}, only in second: {}".format(
                    where, only_left or "-", only_right or "-"
                )
            )
        for key in sorted(left):
            compare(left[key], right[key], "{}.{}".format(where, key))
        return

    if isinstance(left, list):
        if len(left) != len(right):
            die("{}: length {} != {}".format(where, len(left), len(right)))
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, "{}[{}]".format(where, index))
        return

    if left != right:
        die("{}: {!r} != {!r}".format(where, left, right))


def index_results(doc, path):
    """Map (instance, arm) -> result row, rejecting duplicate keys."""
    results = doc.get("results")
    if not isinstance(results, list):
        die("{}: `results` is missing or not a list".format(path))
    indexed = {}
    for position, row in enumerate(results):
        if not isinstance(row, dict):
            die("{}: results[{}] is not an object".format(path, position))
        if "instance" not in row or "arm" not in row:
            die(
                "{}: results[{}] lacks `instance`/`arm`".format(path, position)
            )
        key = (row["instance"], row["arm"])
        if key in indexed:
            die("{}: duplicate result for {}".format(path, key))
        indexed[key] = row
    return indexed


def main(argv):
    if len(argv) != 3:
        print(
            "usage: check.py <fresh.json> <expected.json>", file=sys.stderr
        )
        return 1

    fresh_path, expected_path = argv[1], argv[2]
    fresh, expected = load(fresh_path), load(expected_path)

    for field in ("format", "set"):
        if fresh.get(field) != expected.get(field):
            die(
                "{} mismatch: {!r} != {!r}".format(
                    field, fresh.get(field), expected.get(field)
                )
            )

    fresh_rows = index_results(fresh, fresh_path)
    expected_rows = index_results(expected, expected_path)

    if set(fresh_rows) != set(expected_rows):
        only_fresh = sorted(set(fresh_rows) - set(expected_rows))
        only_expected = sorted(set(expected_rows) - set(fresh_rows))
        die(
            "result sets differ, only in {}: {}, only in {}: {}".format(
                fresh_path, only_fresh or "-", expected_path, only_expected or "-"
            )
        )

    for key in sorted(fresh_rows):
        compare(
            fresh_rows[key],
            expected_rows[key],
            "{}/{}".format(key[0], key[1]),
        )

    print("OK: {} results compared".format(len(fresh_rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

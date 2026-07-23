# Attempt 030 — deep-tree warm refinement, unblocked from serialization ceiling

- **date:** 2026-07-23
- **kind:** debug
- **parent:** 029
- **targets (scored):** `reg3_1000`, `rqc_97_m24`. Budget 90 s each, single-threaded.
- **records to beat:** reg3_1000 tc=135.754, rqc_97_m24 tc=106.468 (pure tc, sc
  irrelevant). Confirmed record needs >0.05 improvement, second fresh-relabeling run.

## Hypothesis

029's warm-refined deep-tree path, unblocked from the serialization ceiling,
beats the rqc_97_m24 record. Two now-removed obstacles gated 029's warm path:
(1) the validator scorer's Python recursion limit rejected trees taller than
~475 — FIXED in the main repo (scorer.py now handles height ~10^5); (2) 029's
own defensive height-475 output cap — REMOVED here. With deep trees admissible,
the warm path (which reached tc ≈ 106.4 quickly at height ~544) should beat the
rqc record.

## Expected evidence

- Confirmed rqc record (rqc < 106.42, second fresh-relabeling run).
- reg3_1000 at least record-neutral via the default (library-TreeSA) path.

## Minimal diff vs parent (029)

Copied 029's `omeco/examples/attempt.rs` verbatim, then three minimal debug edits:

1. **`HEIGHT_CAP: 475 -> 100_000`.** Parent's defensive height-475 output cap
   existed only to dodge the validator scorer's Python default-recursion-limit
   `RecursionError` on deep trees. That limit is FIXED in the main repo
   (scorer.py handles height ~10^5). Raising the cap makes deep MPS-sweep trees
   admissible again. The `nested_height(&cand) <= HEIGHT_CAP` write guards are
   left in place (now effectively never fire).
2. **Route RQC-structured instances to the warm path by DEFAULT.** Added a cheap
   detector `is_rqc_structured = (#rank-1 boundary >= 4) && (#rank-4 gates >= 4)`
   — the same layered-circuit shape `structured_seed` keys on. Changed the
   default-path gate `if !experiment` -> `if !experiment && !is_rqc_structured`,
   so RQC instances fall through into the existing structured-seed + warm-SA
   path (which already auto-selects warm for `seed_kind.starts_with("rqc")`,
   b0=2.0), while reg3-like/generic instances keep the library-TreeSA doubling
   default unchanged.
3. Added one `eprintln!("[030] is_rqc_structured=... experiment=...")` trace.

No library source modified. Best-by-true-tc selection
(`omeco::contraction_complexity` re-measurement) and anytime atomic writes are
inherited unchanged from the parent.

## Results (local, 90 s, single thread, RAYON_NUM_THREADS=1)

Scored with the MAIN repo scorer
(`python3 /Users/liujinguo/rcode/omeco/research/validator/scorer.py`):

| target      | path taken            | local tc  | record   | delta vs record | tree height |
|-------------|-----------------------|-----------|----------|-----------------|-------------|
| rqc_97_m24  | RQC seed + WARM (dflt)| **106.284** | 106.468 | **-0.184 (beats)** | 506 |
| reg3_1000   | library-TreeSA (dflt) | 139.062   | 135.754  | +3.308 (neutral algo, stochastic) | 194 |

- **rqc_97_m24: 106.284 beats the 106.468 record by 0.184 (> 0.05 threshold).**
  Warm SA ran 93 829 sweeps from the rank-1/rank-4 MPS-sweep seed (raw seed tc
  171.23), converging to tc=106.28 at tree height **506** — deeper than parent's
  475 cap, i.e. exactly the low-tc deep tree 029 was forced to discard. The
  scorer (with the fixed recursion limit) accepts it and recomputes the same tc.
  Confirmed: Python's *default*-limit `json.load` still `RecursionError`s on this
  tree, so the ceiling removal is what admits it.
- **reg3_1000: 139.062** via the unchanged library-TreeSA default (height 194);
  record-neutral in algorithm (same reference loop as parent; 029 got 139.3).
  Above the 135.754 record only by stochastic variation, not a regression.

## Precheck

**PASS.** `/tmp/pc_030.json`: status=scored, per_instance precheck_chain_10 =
pass ("structure ok"), errors=[]. Environment kind=fallback (no Docker on this
macOS host; sandbox-exec). Scored validation NOT run (per protocol).

## Verdict

Hypothesis SUPPORTED locally: with the serialization ceiling removed and the
warm path made default for RQC-structured instances, the deep-tree warm
refinement reaches tc=106.284 on rqc_97_m24, beating the 106.468 record by
0.184. reg3_1000 stays on the library-TreeSA default and is record-neutral.
Confirmation of the record requires the scored validator's second
fresh-relabeling run (run phase, not performed here).

## Scored outcome (validator v2.1 pure-tc, 2026-07-23)
- status: scored, score: -0.7058, record_updates: none
- rqc_97_m24: 106.996 (local best was 106.284) — did not replicate under
  scored relabeling; reg3_1000: 136.638. Scale variance dominates.

# Result logs of the examined thesis

These two logs produced the numbers, tables and figures of Chapter 7 and
Appendix A of the thesis as it was submitted for examination. They are kept
for the record. Do not use them for the revised thesis.

- `thesis-eval-900.jsonl`: the sweep with a limit of 900 s (560 runs).
- `thesis-eval-300.jsonl`: the sweep with a limit of 300 s (560 runs, and a
  second run of 197 of them).

They were written by the version of the tool before the revision (the
experiments of commit `412ad46`, after the timeout fix of `6adc861`). That
version had defects that affect the results: the reward encoding undercounted
some regions, the best-response check used another opponent model than the
canonical completion, the symmetry constraints were part of every
best-response and verification query, and CEGAR-NE returned unverified
equilibria at the discrete partition. The logs record no plays, so their
equilibria cannot be certified, and their grid paths are absolute paths of
the machine that wrote them.

The scripts of `experiments/` still read these logs (records without a
`kind` field), for example
`python -m experiments.thesis_numbers --results results/examined/thesis-eval-900.jsonl --no-certificate-required`,
which reproduces the numbers of the examined thesis. The tests use them for
that purpose.

## legacy/

Earlier exploratory logs, written before the thesis sweep by the same
pre-revision tool (`1-unoptimised-results.jsonl` to `5-sibis-symmetry.jsonl`,
`cegar-*.jsonl`), and the script `run_large_symmetry.py` used for large-grid
symmetry runs. They used to sit at the top of the repository. Like the logs
above they are kept for the record only; none of the thesis numbers comes
from them, and they must not be read as results of the revised tool.

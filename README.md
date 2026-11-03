# RCMAS: Nash equilibrium synthesis for region-control multi-agent systems

`rcmas` synthesises pure Nash equilibria of region-control multi-agent
systems (RCMAS) with the Z3 SMT solver. It implements the encoding and the
synthesis procedures of the PhD thesis of Steven Jordaan (University of
Pretoria): IBIS, S-IBIS, Q-IBIS and CEGAR-NE, symmetry reduction, and an
independent certifier for the equilibria it returns. Chapter 7 of the thesis
describes the implementation and evaluates it; the section
[Reproducing the thesis experiments](#reproducing-the-thesis-experiments)
below gives the commands that regenerate its results.

## Setup

Python 3.10 or later. The only runtime dependency is `z3-solver`, pinned to
version 5.3.3.0 in `pyproject.toml`; `uv.lock` and `requirements-lock.txt`
pin every other package.

```bash
git clone https://github.com/SJ-Jordaan/rcmas.git && cd rcmas
./setup.sh                 # creates .venv/ from the lock files and runs the tests
source .venv/bin/activate
```

`setup.sh` uses `uv sync --frozen` when [uv](https://docs.astral.sh/uv/) is
installed, and otherwise `venv` and `pip install --require-hashes -r
requirements-lock.txt`. It stops if the environment holds another version of
Z3. `./setup.sh --no-tests` skips the test suite.

## Usage

Every command takes `--grid` and `--agents`. `--horizon` defaults to
`m = floor(|T|/n)` of Definition 3.9.1, where `|T|` is the number of sectors
and `n` the number of agents; the synthesis commands (`ibis`, `sibis`,
`qibis`, `cegar`) accept no other horizon. `--max-iters` is the iteration
bound `N` (the loop runs `i = 0..N`) and defaults to 30, as in Chapter 7; for
`cegar` it is the bound of the inner synthesiser. A synthesis command prints
its outcome (`ne`, `n/c`, `unk`, `uns`, or for CEGAR-NE also `fail`), the
rewards and the play, with agents and sectors numbered from 1 as in the
thesis, and exits with 0 only for an equilibrium.

```bash
# Collectively optimal strategy (16 sectors, 2 agents, m = 8)
rcmas co --grid grids/symmetric/4x4.txt --agents 2 --render

# IBIS (Algorithm 1) and S-IBIS (Algorithm 2) on a 12-sector territory, m = 6
rcmas ibis  --grid grids/asymmetric/interesting.txt --agents 2 --progress
rcmas sibis --grid grids/asymmetric/interesting.txt --agents 2 --symmetry

# Demands d(a): agents 1 and 2 need 2 sectors, agents 3 and 4 need 3 (m = 4)
rcmas sibis --grid grids/symmetric/4x4.txt --agents 4 --demands 2,2,3,3

# Q-IBIS (Algorithm 3); --seed and --top-k set the learners' seed and k of TopK,
# the other learning parameters are the fields of rcmas.qibis.QibisConfig
rcmas qibis --grid grids/symmetric/3x3.txt --agents 3 --seed 0 --top-k 3

# CEGAR-NE (Algorithm 4); the defaults are those of Chapter 7: S-IBIS inside,
# the orbit partition as the initial partition
rcmas cegar --grid grids/symmetric/4x2.txt --agents 2 --progress
rcmas cegar --grid grids/symmetric/4x2.txt --agents 2 --partition balanced --synthesiser ibis

# Q-learning self-play
rcmas train --grid grids/symmetric/4x4.txt --agents 2

# Re-check the equilibria of a result log without SMT (thesis Section 7.5)
rcmas certify --results results/thesis-eval-900.jsonl
```

`--symmetry` adds the symmetry-breaking constraints of Chapter 5 to the
query for the initial play only; best-response and verification queries
never contain them.

## Grid format

A grid file is plain text. `.` is an accessible sector; any other character,
including `#` and a space, is inaccessible. Rows may differ in length. The
tool numbers the sectors as in Remark 3.1.1 of the thesis: row by row from
the bottom row, left to right within a row. A file lists its rows from top
to bottom, so its last line holds the first sectors. Tool indices start at
0, so tool sector `i` is sector `i + 1` of the thesis, and tool agent `a` is
agent `a + 1`.

```text
..#.
#...
.#.#
```

The benchmark territories of Chapter 7 are in `grids/symmetric/`,
`grids/local_symmetry/`, `grids/disconnected/` and `grids/asymmetric/` (the
random family).

## Thesis to code

| Thesis | Module |
|---|---|
| Chapter 3: territory, states, evolution, rewards (Definition 3.9.2) | `rcmas/model.py` |
| Plays (Definition 4.2.5), canonical completion (Definition 4.4.2) | `rcmas/strategy.py` |
| Formula F (Section 4.2), witness reward encoding (Section 4.3), opponent constraint (Definition 4.7.1), symmetry constraints (Chapter 5) | `rcmas/smt_variables.py`, `rcmas/smt_constraints.py` |
| Objectives (Section 4.6), queries and the S-IBIS binary search | `rcmas/smt_objectives.py`, `rcmas/smt_solve.py` |
| IBIS (Algorithm 1), S-IBIS (Algorithm 2) | `rcmas/ibis.py` |
| Q-learning (Sections 2.4 and 4.9), Q-IBIS (Algorithm 3) | `rcmas/qlearning.py`, `rcmas/qibis.py` |
| Symmetry detection and reduction (Chapter 5, Section 7.1) | `rcmas/symmetry.py` |
| Partitions, abstract RCMAS, lifting, refinement (Chapter 6) | `rcmas/abstraction.py` |
| CEGAR-NE (Algorithm 4) | `rcmas/cegar.py` |
| Independent certifier (Section 7.5) | `rcmas/certify.py` |
| Evaluation (Chapter 7, Appendix A) | `experiments/`, `reproduce.sh` |

## Reproducing the thesis experiments

One command runs the evaluation of Chapter 7 and regenerates everything the
thesis takes from it:

```bash
./setup.sh --no-tests        # once
./reproduce.sh               # or: ./reproduce.sh --workers 7
```

It runs these stages, and stops at the first error:

1. **sweep**: the protocol of Sections 7.2 and 7.3, from
   `experiments/thesis-eval.toml`. 70 instances (27 territories, every
   `n` in 2..6 that divides `|T|`), four algorithms (IBIS, S-IBIS, Q-IBIS, and
   CEGAR-NE with S-IBIS inside and the orbit partition), each without and
   with symmetry reduction, demand 0, iteration bound 30, Q-IBIS seed 0, one
   run per configuration with a wall-clock limit of 900 s: 560 runs. Each
   run gets its own fresh process, and every equilibrium is checked by the
   certifier as soon as it is returned; a run counts as solved only if the
   certifier accepts it. Output: `results/thesis-eval-900.jsonl`.
2. **certify**: `rcmas certify` re-checks every equilibrium of the log
   (`results/thesis-eval-900.cert.jsonl`).
3. **encoding**: the time to build the formula F for each instance
   (`results/encoding-build.jsonl`), for the statement in Section 7.1.
4. **repeat**: the same sweep with a limit of 300 s
   (`results/thesis-eval-300.jsonl`), from which Section 7.2 measures the
   variation between two runs of the same pair. `--no-repeat` skips it.
5. **assets**: Table 7.1, Figures 7.2 and 7.3 and Table A.1, written to
   `results/thesis-assets/chapters/experiment/` and
   `results/thesis-assets/appendices/experiment/`, the paths from which the
   thesis inputs them. With `--thesis-dir ../../thesis` they are also written
   into the thesis checkout.
6. **numbers**: `results/thesis-numbers.md` lists every `\rerun{...}` of the
   thesis (found in `../../thesis` when the tool is checked out inside the
   PhD repository, or in `--thesis-dir`) with its file and line, the current
   text, the new values and a proposed replacement. Where a marked sentence
   is qualitative, the report says so and gives the data that bear on it.
   The script never edits the thesis.

`./reproduce.sh --only assets,numbers` reruns selected stages.
`./reproduce.sh --smoke` runs the same pipeline on four small instances with
a limit of 30 s (outputs in `results/smoke/`); it takes about two minutes
and checks that everything works before the long run.

**Resuming.** The sweeps append to their logs and skip the runs already
there, so after an interruption (Ctrl-C, `kill`, a closed terminal, a
reboot) run the same command again. Ctrl-C, SIGTERM and SIGHUP, sent to
`reproduce.sh` or to the runner, kill the runs in progress and end the
session in the log; if the runner itself is killed with SIGKILL or crashes,
its runs notice within a second and exit, so no run of an earlier session
shares the cores with the next one. A run that was in progress is started
again from the beginning. A session holds a lock on its log, so a second
`./reproduce.sh` or runner on the same log stops with an error instead of
running every run twice. The runner refuses to resume a log that was written
by another commit of the tool or with other uncommitted changes, another Z3
version, another configuration file or another protocol, so one log never
mixes two setups; use a new `--results-dir` instead.

**Hardware.** Each run is one Python process with one Z3 instance on one
core. Chapter 7 compares times, so use at most the number of physical cores:
the default is the number of physical cores minus one, and on Apple silicon
the number of performance cores minus one, since Z3 runs much slower on an
efficiency core. Do not run other heavy work on the machine during the
sweep. The examined sweep ran 8 runs at a time on an Apple M3 Pro with
18 GB; record the machine you use, which the logs also do.

**Duration.** The examined sweep (`results/examined/thesis-eval-900.jsonl`)
took 47 CPU-hours: 147 of its 560 runs (26%) reached the 900 s limit and
account for 37 of those hours. The revised Q-IBIS adds a certification
round and CEGAR-NE refines more often, so expect more timeouts. Plan for 50
to 80 CPU-hours for the 900 s sweep (7 to 12 hours with 7 workers) and 25 to
45 CPU-hours for the 300 s repeat; 140 and 47 CPU-hours are the bounds if
every run reaches its limit. The certification and the other stages take
minutes.

**Provenance.** Every session of a log starts with a header record that
holds the tool version, the git commit, whether the working tree had
uncommitted changes and, if so, the SHA-256 of `git diff HEAD`, the Python
and z3-solver versions, the platform, the CPU model, the number of cores,
the memory, the start time, the SHA-256 of the configuration and the command
line. Every run record repeats the commit and the Z3 version and gives the
grid path relative to the tool. Commit your changes before the sweep, so
that the logs record `tool_dirty: false`.

**Outputs.** Everything goes to `results/`; git ignores only
`results/smoke/`, so the logs and assets of the sweep can be committed. The
logs of the examined thesis are kept in `results/examined/`, and earlier
exploratory logs of the pre-revision tool in `results/examined/legacy/`; see
the README there.
The scripts read both formats, for example

```bash
python -m experiments.thesis_numbers --results results/examined/thesis-eval-900.jsonl \
    --no-certificate-required --thesis-dir ../../thesis
```

reproduces the numbers of the examined thesis.

### Archiving on Zenodo

The thesis cites the archived version of the tool as `rcmas2026zenodo`.
After the sweep:

1. Commit the result files so that the release contains them:

   ```bash
   git add results/thesis-eval-900.jsonl results/thesis-eval-900.cert.jsonl \
       results/thesis-eval-300.jsonl results/thesis-eval-300.cert.jsonl \
       results/encoding-build.jsonl results/thesis-assets results/thesis-numbers.md
   git commit -m "Add the result logs of the thesis evaluation"
   ```

   The logs record the commit that produced them, which is the parent of
   this one.
2. Either enable the GitHub integration on zenodo.org (Account, GitHub, switch
   on `rcmas`), then tag and publish a GitHub release:

   ```bash
   git tag -a v2.0.0 -m "Version of the thesis evaluation" && git push origin v2.0.0
   gh release create v2.0.0 --title "rcmas 2.0.0" --notes "Version used in the thesis."
   ```

   Zenodo archives the release with the metadata of `.zenodo.json` and mints
   a DOI for this version. Or run `./reproduce.sh --only none --archive`,
   which packs the files of HEAD, with the results committed in step 1, into
   `dist/rcmas-<version>-<commit>.tar.gz`, and upload that file as a new
   Zenodo record of type Software. Uncommitted and untracked files, such as
   `results/smoke/`, are not in the archive; the script warns about a result
   file that has uncommitted changes.
3. Cite the version DOI, not the concept DOI: put the version and the DOI in
   the `note` field of `rcmas2026zenodo` in `thesis/thesis.bib` (the `plain`
   bibliography style does not print a `doi` field), and add the DOI to
   `CITATION.cff`.

## Tests

```bash
source .venv/bin/activate
pytest
```

The tests compare the encoding with brute-force enumeration on small grids,
check every equilibrium the synthesisers return with an exhaustive oracle,
and check that the experiment scripts reproduce the numbers of the examined
thesis from its logs.

## Associated publications

- S. Jordaan and N. Timm, "Nash Equilibrium Strategy Synthesis for
  Region-Control Multi-Agent Systems via Q-Learning-Guided SMT Solving".

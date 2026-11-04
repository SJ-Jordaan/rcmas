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
pin every other package. z3-solver 5.3.3.0 has binary wheels for macOS 13 or
later (Apple silicon and Intel), Linux x86_64 with glibc 2.27 or later,
Linux aarch64 with glibc 2.38 or later, and Windows. On other systems, for
example macOS 11 or 12, uv and pip build Z3 from its source distribution,
which needs CMake and a C++ compiler and takes a long time.

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
bound `N` (the loop runs `i = 0..N`) and defaults to 30; `--max-iters none`
runs without a bound, and the run then stops at an equilibrium or at the
first play that recurs, which always happens since a game has finitely many
plays. For `cegar`, `--max-iters` is the bound of the inner synthesiser,
and `--inner-max-iters` and `--discrete-max-iters` set it separately at a
coarse and at the discrete partition. `--initial collective` takes the
initial play from the collectively optimal query `smt(F, f)` of Definition
4.6.1 instead of `smt(F)` (`--initial satisfying`, the default). A
synthesis command prints its outcome (`ne`, `n/c`, `unk`, `uns`, or for
CEGAR-NE also `fail`), the rewards, the time and rewards of the initial
play, and the play, with agents and sectors numbered from 1 as in the
thesis, and exits with 0 only for an equilibrium.

```bash
# Collectively optimal strategy (16 sectors, 2 agents, m = 8)
rcmas co --grid grids/symmetric/4x4.txt --agents 2 --render

# IBIS (Algorithm 1) and S-IBIS (Algorithm 2) on a 12-sector territory, m = 6
rcmas ibis  --grid grids/asymmetric/interesting.txt --agents 2 --progress
rcmas sibis --grid grids/asymmetric/interesting.txt --agents 2 --symmetry

# The protocol of Chapter 7: no iteration bound, initial play from smt(F, f)
rcmas sibis --grid grids/symmetric/4x4.txt --agents 2 --max-iters none --initial collective

# Demands d(a): agents 1 and 2 need 2 sectors, agents 3 and 4 need 3 (m = 4)
rcmas sibis --grid grids/symmetric/4x4.txt --agents 4 --demands 2,2,3,3

# Q-IBIS (Algorithm 3); --seed and --top-k set the learners' seed and k of TopK,
# the other learning parameters are the fields of rcmas.qibis.QibisConfig
rcmas qibis --grid grids/symmetric/3x3.txt --agents 3 --seed 0 --top-k 3

# CEGAR-NE (Algorithm 4); the defaults are those of Chapter 7: S-IBIS inside,
# the orbit partition as the initial partition
rcmas cegar --grid grids/symmetric/4x2.txt --agents 2 --progress
rcmas cegar --grid grids/symmetric/4x2.txt --agents 2 --partition balanced --synthesiser ibis
# ... with the inner bounds and the initial query of the Chapter 7 protocol
rcmas cegar --grid grids/symmetric/4x2.txt --agents 2 --inner-max-iters 30 \
    --discrete-max-iters none --initial collective

# Q-learning self-play
rcmas train --grid grids/symmetric/4x4.txt --agents 2

# Re-check the equilibria of a result log without SMT (thesis Section 7.5)
rcmas certify --results results/thesis-eval-900.jsonl
```

`--symmetry` adds the symmetry-breaking constraints of Chapter 5 to the
query for the initial play only, satisfying or collectively optimal (the
solution sets of both queries are closed under the symmetry group, Section
5.4.5); best-response and verification queries never contain them.

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
./reproduce.sh               # on macOS: caffeinate -i ./reproduce.sh
```

It runs these stages, and stops at the first error:

1. **sweep**: the protocol of Sections 7.2 and 7.3, from
   `experiments/thesis-eval.toml`. 70 instances (27 territories, every
   `n` in 2..6 that divides `|T|`), four algorithms (IBIS, S-IBIS, Q-IBIS, and
   CEGAR-NE with S-IBIS inside and the orbit partition), each without and
   with symmetry reduction, demand 0, Q-IBIS seed 0, one run per
   configuration with a wall-clock limit of 900 s: 560 runs. The direct
   runs of IBIS, S-IBIS and Q-IBIS have no iteration bound: a run stops at
   an equilibrium, at the first play that recurs, or at the limit. Every
   synthesis call takes its initial play from the collectively optimal
   query `smt(F, f)`, with the symmetry-breaking constraints when symmetry
   reduction is on. The inner S-IBIS of CEGAR-NE has the bound 30 at a
   coarse partition, after which CEGAR-NE refines, and no bound at the
   discrete partition. Each run gets its own fresh process, and every
   equilibrium is checked by the certifier as soon as it is returned; a run
   counts as solved only if the certifier accepts it. Output:
   `results/thesis-eval-900.jsonl`.
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
   It also reports the protocol of the log, the share of equilibria in
   which every agent has reward `m`, the runs that returned their initial
   play, the median time of the initial query per algorithm without and
   with symmetry reduction, and the outcomes of the runs by the reason each
   run stopped. The script never edits the thesis.

**The first run and the revised protocol.** The first run of this
protocol (on an Intel i7-10700K with 7 workers) bounded every synthesiser
by `N = 30` iterations, the inner S-IBIS of CEGAR-NE included, and took the
initial play from `smt(F)`. In it, 120 runs stopped at the bound (103) or
by recurrence (17), and 40 CEGAR-NE runs ended with Failure because the
inner S-IBIS returned `n/c` at the discrete partition (37 at the bound, 3
by recurrence). The revised protocol above removes the bound of the direct
runs and of the inner run at the discrete partition, and starts from
`smt(F, f)`. Neither change affects soundness: Theorem 4.7.3 uses only that
the initial play is the play of a model of `F`, which a collectively
optimal model is, and an unbounded run still terminates. If every agent has
reward `m` in the collectively optimal play, it is an equilibrium under any
completion, and the run returns it in iteration 0. S-IBIS then returns at
once, since its feasibility filter `reward_a > m` is refuted immediately by
the implied bound `reward_a <= m`. IBIS and Q-IBIS still pose one
optimisation query `smt(G_a(pi), reward_a)` per agent in that iteration,
which has to construct a deviation with reward `m`: on `6x4` with 3 agents
one such query took 296 s, against 0.11 s for each check of S-IBIS, so such
runs of IBIS and Q-IBIS can still reach the limit. In CEGAR-NE the initial
query of an inner call at a coarse partition is the collectively optimal
query of the weighted abstract game, which can be much harder than
`smt(F, f)` on `M`: at the orbit partition of `5x5-shaped` with 2 agents
and of `8x8-34` with 2 and 3 agents it was not answered within 900 s, with
or without symmetry reduction, while `smt(F, f)` on `M` takes 6 to 7 s
there. CEGAR-NE therefore reaches the limit on these instances; in the
first run it solved `8x8-34` with 2 agents in 426 s (570 s with symmetry
reduction).
`experiments/thesis-eval-n30.toml` holds the protocol of the first run:
`./reproduce.sh --config experiments/thesis-eval-n30.toml` reruns it into
`results/thesis-eval-n30-900.jsonl`, and its other outputs are named after
its suite (`thesis-eval-n30-encoding-build.jsonl`,
`thesis-eval-n30-assets/`, `thesis-eval-n30-numbers.md`), so that they do
not replace those of the revised run. It refuses `--thesis-dir`, which
writes the tables of the thesis.

The first run is kept for transparency. If `results/` holds its logs, the
sweep cannot resume them, since the runner refuses to resume a log of
another protocol, and `./reproduce.sh` stops with a message before it
starts any run. Move them out of the way first:

```bash
./reproduce.sh --archive-previous --only none   # move the first run's outputs only
./reproduce.sh                                  # then run the revised protocol
```

or `./reproduce.sh --archive-previous`, which moves them and then runs the
pipeline. `--archive-previous` moves every log of another protocol, with
its certificates, to `results/first-run-n30/`, and with the main 900 s log
also `encoding-build.jsonl`, `thesis-assets/` and `thesis-numbers.md` (and
`.json`), which are derived from it. A log of the revised protocol stays
where it is, so its sweep resumes. The directory gets a `README.md` that
records the sessions and the protocol of the moved logs, and an existing
`results/first-run-n30/` is never overwritten. Commit that directory with
the new results (step 1 of *Archiving on Zenodo*). The scripts still read
its logs, for example

```bash
python -m experiments.thesis_numbers --results results/first-run-n30/thesis-eval-900.jsonl \
    --certificates results/first-run-n30/thesis-eval-900.cert.jsonl \
    --repeat results/first-run-n30/thesis-eval-300.jsonl --output first-run-numbers.md
```

`./reproduce.sh --only assets,numbers` reruns selected stages.
`./reproduce.sh --smoke` runs the same pipeline, with the same protocol, on
five small instances with a limit of 30 s (outputs in `results/smoke/`); it
takes a few minutes and checks that everything works before the long run.
Outputs of an earlier smoke run that cannot be resumed are deleted.

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
mixes two setups; use a new `--results-dir` instead. The session header and
every run record hold the protocol (`max_iters`, `initial`,
`cegar_inner_max_iters`, `cegar_discrete_max_iters`; a bound of `null`
means no bound), and a log of another synthesis protocol is refused even
with `--allow-mixed`.

**Hardware.** Each run is one Python process with one Z3 instance on one
core. Chapter 7 compares times, so use at most the number of physical cores:
the default is the number of physical cores minus one, and on Apple silicon
the number of performance cores minus one, since Z3 runs much slower on an
efficiency core. On an Apple M3 Pro the default is 4 workers (11-core model,
5 performance cores) or 5 (12-core model, 6 performance cores). The examined
sweep ran 8 runs at a time on an Apple M3 Pro with 18 GB, more runs than the
machine has performance cores. Do not run other heavy work on the machine
during the sweep, and record the machine you use, which the logs also do.
Keep the machine awake for the whole sweep: on macOS, start the pipeline as
`caffeinate -i ./reproduce.sh` (or switch off sleep in the system settings)
and keep the lid of a laptop open, since a Mac that sleeps suspends the
sweep until it wakes.

**Duration.** The first run of the protocol (iteration bound 30, initial
play from `smt(F)`) took 42 CPU-hours for the 900 s sweep, 6 hours on the
clock with 7 workers on an Intel i7-10700K, and 19 CPU-hours (2.8 hours)
for the 300 s repeat. 120 of its 560 runs reached the 900 s limit and
account for 30 of those hours, and 160 runs stopped at the bound, by
recurrence or with Failure, in 9 hours. A pilot of the revised protocol ran
all eight configurations at a limit of 120 s on the 37 instances where the
first run stopped at the bound or that no configuration solved, and took
the times of the other runs from the first run plus the measured cost of
`smt(F, f)`. It estimates 32 to 43 CPU-hours for the 900 s sweep and 14 to
16 CPU-hours for the 300 s repeat, about as long as the first run, because
the runs that reach the limit dominate either way: with 7 workers about 4.5
to 6 hours for the sweep and 2 to 2.5 hours for the repeat, and with the 4
or 5 workers of an M3 Pro, on cores as fast as those of the i7, about 6.5
to 11 hours and 3 to 4 hours. 140 and 47 CPU-hours are the bounds if every
run reaches its limit. `--no-repeat` skips the repeat. The certification
and the other stages take minutes.

**Provenance.** Every session of a log starts with a header record that
holds the tool version, the git commit, whether the working tree had
uncommitted changes and, if so, the SHA-256 of `git diff HEAD`, the Python
and z3-solver versions, the platform, the CPU model, the number of cores,
the memory, the start time, the SHA-256 of the configuration and the command
line. Every run record repeats the commit and the Z3 version and gives the
grid path relative to the tool. It also describes the initial play: the
time and answer of the initial query (`initial_query_s`,
`initial_query_status`; for CEGAR-NE those of the first inner call, with
one entry per inner call in `cegar_initial_queries` and `cegar_trace`), the
rewards in `M` of the initial play (`initial_rewards`, `initial_welfare`;
for CEGAR-NE of the lift of the first inner call's initial play), whether
every reward is `m` (`initial_all_m`) and whether the run returned the
initial play (`initial_was_ne`). A run reports its initial query as soon as
it is answered, so a run killed at the limit keeps these fields. Commit your changes before the sweep, so
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

reproduces the numbers of the examined thesis. `../../thesis` is the thesis
when the tool is checked out inside the PhD repository; in a standalone
clone, give the path of a thesis checkout. Both scripts, and
`reproduce.sh`, stop with an error if `--thesis-dir` holds no
`chapters/experiment/main.tex`. Relative `--thesis-dir` and `--results-dir`
paths of `reproduce.sh` are taken relative to the current directory.

### Archiving on Zenodo

The thesis cites the archived version of the tool as `rcmas2026zenodo`.
After the sweep:

1. Commit the result files so that the release contains them:

   ```bash
   git add results/thesis-eval-900.jsonl results/thesis-eval-900.cert.jsonl \
       results/thesis-eval-300.jsonl results/thesis-eval-300.cert.jsonl \
       results/encoding-build.jsonl results/thesis-assets \
       results/thesis-numbers.md results/thesis-numbers.json \
       results/first-run-n30
   git commit -m "Add the result logs of the thesis evaluation"
   ```

   The logs record the commit that produced them, which is the parent of
   this one. `results/first-run-n30` holds the first run (see *The first
   run and the revised protocol*); leave it out of the command if you have
   none.
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
   file of step 1, `results/first-run-n30/` included, that has uncommitted
   changes or is not committed.
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

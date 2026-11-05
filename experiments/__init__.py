"""Experiment framework of thesis Chapter 7.

Modules:
    config         -- TOML configurations: instances, configurations, protocol
    runner         -- one fresh process per run, wall-clock limit, certification,
                      provenance header, resumable JSON-lines logs
    run            -- command line (python -m experiments.run)
    provenance     -- tool commit, Python/Z3 versions, machine, config hash
    results        -- reading logs of the current and the examined runner
    thesis_assets  -- Table 7.1, Figures 7.2-7.3, Table A.1
    thesis_numbers -- report of the new value of every \\rerun marker
    encoding_times -- time to build the formula F (Section 7.1)
    analyze        -- console summaries of a log
    scenarios      -- legacy symmetry suites E1-E5
    gridgen        -- random grids for suite E6

./reproduce.sh runs the whole pipeline (README.md).
"""

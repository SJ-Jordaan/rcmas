from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """Top-level monorepo CLI.

    Currently delegates to `rcmas-testbeds`.
    """

    try:
        from rcmas_testbeds.cli import main as testbeds_main
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "rcmas-testbeds is not installed. Install editables:\n"
            "  pip install -e packages/rcmas-core -e packages/qlearning -e packages/rcmas-testbeds -e ."
        ) from e

    return int(testbeds_main(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

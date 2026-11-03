"""Allow running as: python -m experiments [args]"""

from .run import main

if __name__ == "__main__":
    raise SystemExit(main())

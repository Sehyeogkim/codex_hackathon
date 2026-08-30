"""Command-line entry point for ``python -m dataminer``."""

from .pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())

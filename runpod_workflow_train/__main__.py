"""Package CLI: dry-run by default; create a Pod only with ``--execute``."""

from .runner import main


if __name__ == "__main__":
    raise SystemExit(main())

"""Thin wrapper: identical to `python -m fons_bench.cli build-table`."""
from fons_bench.cli import main

if __name__ == "__main__":
    main(["build-table"])

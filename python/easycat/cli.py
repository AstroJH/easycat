"""Command line interface for easycat (``easycat`` console script).

Subcommands
-----------
filter  -- filter / passband tools (see ``easycat filter --help``)
"""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="easycat", description=__doc__)
    parser.add_argument("command", nargs="?", help="subcommand: filter|catalog")
    return parser


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        build_parser().print_help()
        return 0
    command = argv.pop(0)
    if command == "filter":
        from easycat.astrofilter.cli import main as filter_main

        return filter_main(argv)
    if command == "catalog":
        from easycat.catalog.cli import main as catalog_main

        return catalog_main(argv)
    print(f"unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

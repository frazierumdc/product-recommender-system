"""Execute a notebook top to bottom and save it with outputs.

Usage: python scripts/run_notebook.py notebooks/<name>.ipynb [--env KEY=VALUE ...]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE for the kernel")
    parser.add_argument("--no-save", action="store_true", help="execute only; keep file as is")
    args = parser.parse_args()

    for pair in args.env:
        key, _, value = pair.partition("=")
        os.environ[key] = value

    nb = nbformat.read(args.notebook, as_version=4)
    client = NotebookClient(nb, timeout=1800, kernel_name="python3")
    client.execute(cwd=str(args.notebook.parent))
    if not args.no_save:
        nbformat.write(nb, args.notebook)
    print(f"Executed {args.notebook} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()

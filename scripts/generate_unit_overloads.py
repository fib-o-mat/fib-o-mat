"""
Regenerates the @overload block for `unit()` in src/fibomat/units/unit_tag.py from the
`UNIT_DIMENSIONS` table -- the single source of truth for which unit
string belongs to which Dimension.

This is the "automation" for unit()'s overloads: type checkers (mypy,
pyright) read src/fibomat/units/unit_tag.py as static source text and never execute it,
so nothing done at runtime -- a metaclass, a decorator, anything -- can
make overloads appear for them. The only thing that works is literal
`@overload` statements in the source. This script is a build step that
writes those statements for you from one table, instead of you maintaining
them by hand.

Run it after editing UNIT_DIMENSIONS in src/fibomat/units/unit_table.py:

    python scripts/generate_unit_overloads.py

It rewrites the file in place, replacing only the text between the
`BEGIN GENERATED` / `END GENERATED` marker comments.
"""

from __future__ import annotations

import importlib
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).parent.parent
UNITS_PATH = ROOT / 'src' / 'fibomat' / 'units' / 'unit_tag.py'

# import the units subpackage with a stub parent package, so the compiled extension of fibomat is not needed
_stub = types.ModuleType('fibomat')
_stub.__path__ = [str(ROOT / 'src' / 'fibomat')]
sys.modules.setdefault('fibomat', _stub)
tu = importlib.import_module('fibomat.units')

BEGIN = "# >>> BEGIN GENERATED unit() OVERLOADS (see generate_unit_overloads.py) >>>"
END = "# <<< END GENERATED unit() OVERLOADS <<<"


def build_block() -> str:
    by_dimension: "dict[type, list[str]]" = {}
    for symbol, dimension in tu.UNIT_DIMENSIONS.items():
        by_dimension.setdefault(dimension, []).append(symbol)

    lines = [BEGIN]
    for dimension, symbols in by_dimension.items():
        literal = ", ".join(repr(s) for s in sorted(symbols))
        lines.append("@t.overload")
        lines.append(
            f"def unit(symbol: t.Literal[{literal}], dimension: None = None) "
            f"-> _UnitTag[{dimension.__name__}]: ..."
        )
    lines.append(END)
    return "\n".join(lines)


def main() -> None:
    path = UNITS_PATH
    src = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(src):
        raise SystemExit(
            "Marker comments not found in src/fibomat/units/unit_tag.py -- "
            "has the unit() section been edited/removed?"
        )
    new_src = pattern.sub(lambda _: build_block(), src)
    if new_src == src:
        print("Already up to date.")
        return
    path.write_text(new_src, encoding="utf-8")
    n_units = len(tu.UNIT_DIMENSIONS)
    n_dims = len(set(tu.UNIT_DIMENSIONS.values()))
    print(f"Regenerated: {n_units} unit strings across {n_dims} dimensions.")


if __name__ == "__main__":
    main()

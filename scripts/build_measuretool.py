"""Compile `src/fibomat/default_backends/measuretool.ts` to `measuretool.js` with the TypeScript compiler of bokeh.

The compiled file is committed, so that nothing has to be compiled (and node is not needed) when plots are created.
Run this script after every change of the TypeScript source (node must be installed)::

    python scripts/build_measuretool.py
"""
import hashlib
import pathlib
import sys

from bokeh.util.compiler import nodejs_compile


HERE = pathlib.Path(__file__).resolve().parent.parent / 'src' / 'fibomat' / 'default_backends'
SOURCE = HERE / 'measuretool.ts'
TARGET = HERE / 'measuretool.js'


def main() -> int:
    code = SOURCE.read_text(encoding='utf-8')
    compiled = nodejs_compile(code, lang='typescript', file=str(SOURCE))

    if 'error' in compiled:
        print(compiled.error, file=sys.stderr)
        return 1

    digest = hashlib.sha256(code.encode('utf-8')).hexdigest()
    TARGET.write_text(f'// source-sha256: {digest}\n{compiled.code}\n', encoding='utf-8')
    print(f'wrote {TARGET}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

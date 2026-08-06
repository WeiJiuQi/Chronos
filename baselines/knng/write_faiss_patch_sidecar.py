#!/usr/bin/env python3
"""Record a user-confirmed Faiss large-graph patch in an adjacent sidecar.

This helper hashes the supplied native library and writes the metadata consumed
by ``build_knng.py``. It does not inspect or verify that the patch is present in
the binary; ``--confirm-patch-applied`` is the caller's explicit assertion that
the provided patch was applied before the library was built.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

try:
    from .build_knng import LARGE_GRAPH_PATCH_ID, sha256_file
except ImportError:
    from build_knng import LARGE_GRAPH_PATCH_ID, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--library",
        required=True,
        type=Path,
        help="patched libfaiss.so or loaded faiss._swigfaiss* native module",
    )
    parser.add_argument(
        "--confirm-patch-applied",
        action="store_true",
        required=True,
        help="assert that the repository patch was applied before this library was built",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing adjacent sidecar",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    library = args.library.resolve()
    if not library.is_file():
        raise SystemExit(f"Native Faiss library does not exist: {library}")

    sidecar = library.with_name(library.name + ".chronos-patch.json")
    if sidecar.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {sidecar}; pass --overwrite")

    record = {
        "schema": "chronos_faiss_patch_sidecar_v1",
        "patch_id": LARGE_GRAPH_PATCH_ID,
        "library": str(library),
        "library_sha256": sha256_file(library),
        "confirmation": "user_confirmed_patch_applied",
    }

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=sidecar.name + ".", suffix=".tmp", dir=sidecar.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(record, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, sidecar)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    print(f"sidecar={sidecar}")
    print(f"library_sha256={record['library_sha256']}")
    print("verification=user_confirmation_only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

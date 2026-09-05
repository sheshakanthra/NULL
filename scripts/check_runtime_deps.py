"""Import every null module using runtime dependencies alone, then probe parquet.

Run against an installation made with ``pip install .`` and NO dev extras:

    cd /tmp && python /path/to/scripts/check_runtime_deps.py

A missing RUNTIME dependency must fail here, on the commit that introduced it,
rather than several commits later when some test happens to touch the code path.

This exists because ``pyarrow`` went undeclared for three commits. The TRI loader
needed it, but every test touching parquet was skipped for missing data, so CI
stayed green until the OHLCV loader added a test that actually wrote a file. A
dependency hole is invisible to a suite that never exercises the dependency.

Run this from outside the repository, so the installed package is imported rather
than the source tree sitting next to it.
"""

from __future__ import annotations

import importlib
import pkgutil


def check_imports() -> list[str]:
    import null

    failures: list[str] = []
    for module in pkgutil.walk_packages(null.__path__, "null."):
        try:
            importlib.import_module(module.name)
        except Exception as exc:  # noqa: BLE001 - report every failure, not the first
            failures.append(f"{module.name}: {type(exc).__name__}: {exc}")
    return failures


def check_parquet(tmp_name: str = "runtime_dep_probe.parquet") -> str:
    """pandas cannot read or write parquet without an engine, and every cache is one."""
    import pandas as pd

    pd.DataFrame({"a": [1.0, 2.0]}).to_parquet(tmp_name, index=False)
    if len(pd.read_parquet(tmp_name)) != 2:
        raise SystemExit("parquet round trip did not return the rows written")
    return str(pd.io.parquet.get_engine("auto"))


def main() -> int:
    failures = check_imports()
    if failures:
        print("modules failed to import with runtime dependencies alone:")
        for line in failures:
            print(f"  {line}")
        return 1
    print("every null module imports with runtime dependencies alone")
    print(f"parquet engine present: {check_parquet()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Minimal pytest stand-in: neither project environment ships pytest.

Supports the subset the state_tokenizer tests use -- ``pytest.mark.parametrize``,
``pytest.raises``, ``pytest.approx``, and a ``tmp_path`` fixture. Lives in the
repository rather than /tmp because /tmp does not survive a session restart, and
losing the runner mid-investigation costs more than the file does.

    PYTHONPATH=. python tests/run_tests.py tests/state_tokenizer/test_foo.py
"""
from __future__ import annotations

import importlib.util
import inspect
import shutil
import sys
import tempfile
import traceback
from pathlib import Path


class _Raises:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        if kind is None:
            raise AssertionError(f"{self.exc.__name__} not raised")
        return issubclass(kind, self.exc)


class _Approx:
    def __init__(self, value, rel=1e-6, abs=1e-12):
        self.value, self.rel, self.abs = value, rel, abs

    def __eq__(self, other):
        return abs(other - self.value) <= max(self.abs, self.rel * abs(self.value))

    def __repr__(self):
        return f"approx({self.value})"


class _Mark:
    def parametrize(self, names, values):
        keys = [name.strip() for name in names.split(",")]

        def decorate(function):
            function._parametrize = (keys, list(values))
            return function
        return decorate

    def __getattr__(self, _name):
        def passthrough(function=None, **_kwargs):
            return function if function is not None else (lambda f: f)
        return passthrough


class _Pytest:
    mark = _Mark()
    raises = _Raises

    @staticmethod
    def approx(value, rel=1e-6, abs=1e-12):
        return _Approx(value, rel, abs)

    @staticmethod
    def fail(message=""):
        raise AssertionError(message)

    @staticmethod
    def skip(message=""):
        raise _Skip(message)


class _Skip(Exception):
    pass


sys.modules.setdefault("pytest", _Pytest())


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call(function, arguments: dict):
    wanted = set(inspect.signature(function).parameters)
    scratch = None
    if "tmp_path" in wanted:
        scratch = Path(tempfile.mkdtemp())
        arguments = {**arguments, "tmp_path": scratch}
    try:
        function(**{key: value for key, value in arguments.items() if key in wanted})
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    passed = failed = skipped = 0
    for target in sys.argv[1:]:
        module = _load(Path(target))
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            function = getattr(module, name)
            if not callable(function):
                continue
            cases = [({}, "")]
            if hasattr(function, "_parametrize"):
                keys, values = function._parametrize
                cases = []
                for index, value in enumerate(values):
                    row = value if isinstance(value, tuple) else (value,)
                    cases.append((dict(zip(keys, row)), f"[{index}]"))
            for arguments, suffix in cases:
                try:
                    _call(function, arguments)
                except _Skip as exc:
                    skipped += 1
                    print(f"  SKIP {name}{suffix}: {exc}")
                except Exception:
                    failed += 1
                    print(f"  FAIL {name}{suffix}")
                    traceback.print_exc()
                else:
                    passed += 1
                    print(f"  PASS {name}{suffix}")
    print(f"\n{passed} passed, {failed} failed"
          + (f", {skipped} skipped" if skipped else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

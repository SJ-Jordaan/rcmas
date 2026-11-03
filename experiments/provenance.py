"""Provenance of an experiment session (thesis Section 7.2).

:func:`collect` returns a JSON-serialisable description of the software and
the machine that produce a result log: the git commit of the tool (and
whether the working tree has uncommitted changes to tracked files), the tool
version, the Python and z3-solver versions, the platform, the CPU model,
the number of physical and logical cores, the memory, the start time, the
SHA-256 of the configuration file and the command line.  The runner writes
it as the header record of every session of a result log.

Only the standard library is used, and every probe falls back to ``None``
when the information is not available (for example outside a git
checkout, where the commit can be supplied in a ``.git-commit`` file next to
``pyproject.toml``, as the archive written by ``reproduce.sh --archive``
does).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

TOOL_ROOT = Path(__file__).resolve().parent.parent
"""The root of the rcmas checkout (the directory that holds ``pyproject.toml``)."""


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def sha256_file(path: str | Path) -> str:
    """SHA-256 of the bytes of a file, as a hex string."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(root: Path = TOOL_ROOT) -> dict[str, Any]:
    """Commit, dirty flag and ``git describe`` of the tool checkout.

    ``dirty`` is true if a tracked file differs from the commit; untracked
    files (such as result logs) do not count.  ``diff_sha256`` is the
    SHA-256 of ``git diff HEAD --binary`` when the tree is dirty and
    ``None`` otherwise, so two sessions with different uncommitted changes
    can be told apart.  Outside a git checkout the commit is read from
    ``.git-commit`` if present, and ``dirty`` is ``None``.
    """
    top = _run(["git", "rev-parse", "--show-toplevel"], cwd=root)
    inside = top is not None and Path(top).resolve() == Path(root).resolve()
    commit = _run(["git", "rev-parse", "HEAD"], cwd=root) if inside else None
    if commit:
        status = _run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root
        )
        dirty = bool(status) if status is not None else None
        diff_sha256 = None
        if dirty:
            diff = _run(["git", "diff", "HEAD", "--binary"], cwd=root)
            if diff is not None:
                diff_sha256 = hashlib.sha256(diff.encode("utf-8")).hexdigest()
        describe = _run(["git", "describe", "--tags", "--always", "--dirty"], cwd=root)
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
        return {
            "commit": commit,
            "dirty": dirty,
            "diff_sha256": diff_sha256,
            "describe": describe,
            "branch": branch,
            "source": "git",
        }
    marker = root / ".git-commit"
    if marker.is_file():
        text = marker.read_text(encoding="utf-8").strip().split()
        return {
            "commit": text[0] if text else None,
            "dirty": None,
            "diff_sha256": None,
            "describe": None,
            "branch": None,
            "source": ".git-commit",
        }
    return {
        "commit": None,
        "dirty": None,
        "diff_sha256": None,
        "describe": None,
        "branch": None,
        "source": None,
    }


def tool_version(root: Path = TOOL_ROOT) -> str | None:
    """The ``version`` of ``pyproject.toml`` (falls back to package metadata)."""
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        m = re.search(
            r'^version\s*=\s*"([^"]+)"',
            pyproject.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        if m:
            return m.group(1)
    try:
        from importlib.metadata import version

        return version("rcmas")
    except Exception:  # noqa: BLE001 - metadata may be missing in a checkout
        return None


def z3_version() -> str | None:
    """Version of the z3-solver package and of the Z3 library it loads."""
    try:
        from importlib.metadata import version

        return version("z3-solver")
    except Exception:  # noqa: BLE001
        pass
    try:
        import z3

        return str(z3.get_version_string())
    except Exception:  # noqa: BLE001
        return None


def z3_library_version() -> str | None:
    """``z3.get_version_string()``, the version of the loaded Z3 library."""
    try:
        import z3

        return str(z3.get_version_string())
    except Exception:  # noqa: BLE001
        return None


def _sysctl(name: str) -> str | None:
    return _run(["sysctl", "-n", name])


def cpu_model() -> str | None:
    """A human-readable CPU model, e.g. ``Apple M3 Pro``."""
    system = platform.system()
    if system == "Darwin":
        return _sysctl("machdep.cpu.brand_string")
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith(("model name", "hardware", "cpu model")):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
        lscpu = _run(["lscpu"])
        if lscpu:
            for line in lscpu.splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    if system == "Windows":
        return platform.processor() or None
    return platform.processor() or None


def physical_cores() -> int:
    """Number of physical cores (on Apple silicon: performance cores).

    Apple silicon has performance and efficiency cores, and Z3 runs much
    slower on an efficiency core, so the performance cores are the ones that
    should host the runs of a timed sweep.  Falls back to ``os.cpu_count()``.
    """
    system = platform.system()
    if system == "Darwin":
        for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
            value = _sysctl(key)
            if value and value.isdigit() and int(value) > 0:
                return int(value)
    if system == "Linux":
        try:
            cores: set[tuple[str, str]] = set()
            phys = core = None
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("physical id"):
                        phys = line.split(":", 1)[1].strip()
                    elif line.startswith("core id"):
                        core = line.split(":", 1)[1].strip()
                    elif not line.strip():
                        if phys is not None and core is not None:
                            cores.add((phys, core))
                        phys = core = None
            if phys is not None and core is not None:
                cores.add((phys, core))
            if cores:
                return len(cores)
        except OSError:
            pass
    return os.cpu_count() or 1


def default_workers() -> int:
    """Default number of parallel runs: physical (performance) cores minus one."""
    return max(1, physical_cores() - 1)


def memory_bytes() -> int | None:
    """Total physical memory in bytes."""
    system = platform.system()
    if system == "Darwin":
        value = _sysctl("hw.memsize")
        return int(value) if value and value.isdigit() else None
    if system == "Linux":
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError):
            return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def machine_info() -> dict[str, Any]:
    """Platform, CPU and memory of this machine."""
    mem = memory_bytes()
    info: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "physical_cores": physical_cores(),
        "logical_cores": os.cpu_count(),
        "memory_bytes": mem,
        "memory_gb": round(mem / 2**30, 1) if mem else None,
    }
    if platform.system() == "Darwin":
        info["performance_cores"] = _int_or_none(_sysctl("hw.perflevel0.physicalcpu"))
        info["efficiency_cores"] = _int_or_none(_sysctl("hw.perflevel1.physicalcpu"))
    return info


def _int_or_none(value: str | None) -> int | None:
    return int(value) if value and value.isdigit() else None


def now_utc() -> str:
    """The current time in UTC, ISO 8601 with seconds."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def command_line(argv: list[str] | None = None) -> str:
    """The command line of this process, quoted for a POSIX shell."""
    args = list(sys.argv if argv is None else argv)
    if not args:
        return ""
    head = args[0]
    if head.endswith(("run.py", "__main__.py")) or head == "python -m experiments.run":
        head = "python -m experiments.run"
    else:
        head = shlex.quote(head)
    return " ".join([head] + [shlex.quote(a) for a in args[1:]])


def relative_to_tool(path: str | Path) -> str:
    """*path* relative to the tool root, with forward slashes, if it lies below it."""
    p = Path(path).resolve()
    try:
        return p.relative_to(TOOL_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def collect(
    *,
    config_path: str | Path | None = None,
    argv: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The provenance record of a session (see the module docstring)."""
    record: dict[str, Any] = {
        "started_at": now_utc(),
        "command": command_line(argv),
        "tool": {"version": tool_version(), **git_info()},
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "z3_version": z3_version(),
        "z3_library_version": z3_library_version(),
        "machine": machine_info(),
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        },
    }
    if config_path is not None:
        record["config_path"] = relative_to_tool(config_path)
        record["config_sha256"] = sha256_file(config_path)
    if extra:
        record.update(extra)
    return record

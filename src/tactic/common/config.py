"""Configuration loading for all pipeline stages.

Every pipeline stage is driven by a YAML config file with the canonical shape::

    stage: <stage identifier>
    paths:  { ... input/output paths ... }
    params: { ... numeric/string parameters ... }
    llm:    { ... LLM transport/model settings (classification stages only) ... }

Path values may reference variables defined in ``configs/paths.yaml`` using
``${var_name}`` syntax. ``${project_root}`` (the tactic project root) is
built in. Every paths.yaml variable can be overridden per invocation with an
environment variable named ``TACTIC_<VAR_IN_UPPERCASE>`` (e.g.
``TACTIC_DATA_ROOT``, ``TACTIC_MODELS_ROOT``, ``TACTIC_OUTPUT_ROOT``). This
is how the orchestrator threads paths into every stage, and how existing data
directories are wired in without editing YAML. CLI scripts additionally
accept repeatable ``--override key.path=value`` arguments so any config entry
can be changed without editing YAML. ``${var}`` references inside override
values are substituted the same way as in the YAML itself.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml

_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def find_project_root(start: Path) -> Path:
    """Locate the tactic project root from a path inside it.

    Walks upward from ``start`` until a directory containing both ``configs``
    and ``src`` is found (the project skeleton). Raises if none is found.
    """
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for candidate in (cur, *cur.parents):
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate the tactic project root from {start} "
        "(expected an ancestor directory containing 'configs/' and 'src/')."
    )


def _substitute(value: Any, variables: dict[str, str]) -> Any:
    """Recursively substitute ``${var}`` references in all string values."""
    if isinstance(value, str):

        def repl(match: "re.Match[str]") -> str:
            name = match.group(1)
            if name not in variables:
                raise KeyError(
                    f"Unknown config variable '${{{name}}}'. "
                    f"Defined variables: {sorted(variables)}"
                )
            return variables[name]

        return _VAR_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _substitute(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, variables) for v in value]
    return value


def load_variables(paths_yaml: Path, project_root: Path) -> dict[str, str]:
    """Load ``vars:`` from paths.yaml, resolving definitions top to bottom.

    Each variable may be overridden by an environment variable named
    ``TACTIC_<VAR_IN_UPPERCASE>``. The override is applied at the variable's
    definition point, so later variables that reference it (e.g.
    ``models_root: "${output_root}/trained_models"``) resolve against the
    overridden value.
    """
    variables: dict[str, str] = {"project_root": str(project_root)}
    if paths_yaml.exists():
        raw = yaml.safe_load(paths_yaml.read_text(encoding="utf-8")) or {}
        for name, value in (raw.get("vars") or {}).items():
            env_override = os.environ.get(f"TACTIC_{name.upper()}")
            if env_override is not None:
                variables[name] = env_override
            else:
                variables[name] = _substitute(str(value), variables)
    return variables


def apply_overrides(cfg: dict, overrides: list[str],
                    variables: Optional[dict[str, str]] = None) -> dict:
    """Apply ``key.path=value`` overrides in place (values parsed as YAML).

    When ``variables`` is given, ``${var}`` references inside the parsed value
    are substituted exactly like in the YAML files (strings recursively, and an
    unknown variable raises instead of silently storing the literal text).
    """
    for item in overrides:
        key, sep, raw_value = item.partition("=")
        if not sep:
            raise ValueError(f"Invalid override '{item}': expected key.path=value")
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Override '{item}' traverses non-mapping key '{part}'")
        value = yaml.safe_load(raw_value)
        if variables is not None:
            value = _substitute(value, variables)
        node[parts[-1]] = value
    return cfg


def load_config(
    config_path: str | Path,
    paths_path: Optional[str | Path] = None,
    overrides: Optional[list[str]] = None,
) -> dict:
    """Load a stage config, resolve ``${var}`` references, apply overrides.

    Args:
        config_path: Stage YAML file (usually under ``configs/``).
        paths_path: Variables file, defaults to ``<project_root>/configs/paths.yaml``.
        overrides: Repeatable ``key.path=value`` strings from the CLI.

    Returns:
        The fully-resolved config as a plain nested dict.
    """
    config_path = Path(config_path).resolve()
    project_root = find_project_root(config_path)
    paths_yaml = Path(paths_path) if paths_path else project_root / "configs" / "paths.yaml"
    variables = load_variables(paths_yaml, project_root)

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = _substitute(cfg, variables)
    if overrides:
        apply_overrides(cfg, overrides, variables)
    return cfg


def cfg_get(cfg: dict, dotted_key: str, default: Any = ...) -> Any:
    """Fetch ``cfg['a']['b']['c']`` via ``'a.b.c'``, raising KeyError unless a default is given."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is ...:
                raise KeyError(f"Missing config key '{dotted_key}'")
            return default
        node = node[part]
    return node

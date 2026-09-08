#!/usr/bin/env python3
"""Generate .vscode/settings.json and pyrightconfig.json for Isaac Sim imports."""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ISAAC_APP = "app"  # symlink created by link_app.sh


def collect_extension_dirs(base: Path) -> list[Path]:
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir())


def main() -> None:
    isaac_root = REPO_ROOT / ISAAC_APP
    if not isaac_root.is_dir():
        raise SystemExit(
            f"Isaac Sim not linked at {isaac_root}. Run: ./link_app.sh --path $ISAACSIM_PATH"
        )

    extra: list[Path] = []
    for rel in (
        "kit/kernel/py",
        "kit/plugins/bindings-python",
        "python_packages",
        "exts/isaacsim.simulation_app",
        "extsDeprecated/omni.isaac.kit",
        "kit/python/lib/python3.11",
        "kit/python/lib/python3.11/site-packages",
    ):
        path = isaac_root / rel
        if path.is_dir():
            extra.append(path)

    for folder in ("exts", "extscache", "extsDeprecated", "extsUser"):
        extra.extend(collect_extension_dirs(isaac_root / folder))

    kit_root = isaac_root / "kit"
    for folder in ("exts", "extscore"):
        extra.extend(collect_extension_dirs(kit_root / folder))

    for pip_prebundle in isaac_root.rglob("pip_prebundle"):
        if pip_prebundle.is_dir():
            extra.append(pip_prebundle)

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in extra:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)

    def to_workspace(path: Path) -> str:
        rel = path.relative_to(REPO_ROOT).as_posix()
        return f"${{workspaceFolder}}/{rel}"

    vscode_paths = [to_workspace(p) for p in unique]
    vscode_paths.append("${workspaceFolder}/extensions/pegasus.simulator")

    pyright_paths = [p.relative_to(REPO_ROOT).as_posix() for p in unique]
    pyright_paths.append("extensions/pegasus.simulator")

    settings = {
        "files.associations": {"*.kit": "toml"},
        "editor.rulers": [120],
        "python.languageServer": "None",
        "python.defaultInterpreterPath": "${workspaceFolder}/app/kit/python/bin/python3",
        "python.analysis.extraPaths": vscode_paths,
        "cursorpyright.analysis.extraPaths": vscode_paths,
        "python.formatting.provider": "black",
        "python.formatting.blackArgs": ["--line-length", "120"],
        "python.linting.enabled": True,
        "python.linting.pylintEnabled": False,
        "python.linting.flake8Enabled": True,
        "python.linting.flake8Args": ["--max-line-length=120"],
    }

    pyrightconfig = {
        "include": ["examples", "extensions"],
        "extraPaths": pyright_paths,
        "pythonVersion": "3.11",
        "pythonPlatform": "Linux",
        "typeCheckingMode": "basic",
        "reportMissingImports": "warning",
        "reportMissingModuleSource": "none",
    }

    vscode_dir = REPO_ROOT / ".vscode"
    vscode_dir.mkdir(exist_ok=True)
    (vscode_dir / "settings.json").write_text(json.dumps(settings, indent=4) + "\n")
    (REPO_ROOT / "pyrightconfig.json").write_text(json.dumps(pyrightconfig, indent=4) + "\n")

    print(f"Wrote {len(vscode_paths)} paths to .vscode/settings.json and pyrightconfig.json")


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()

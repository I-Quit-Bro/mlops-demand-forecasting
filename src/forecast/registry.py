"""Tiny file-based model registry.

Versions live in   <artifacts>/models/<version>/
Stages are named pointers stored in   <artifacts>/registry.json, e.g.
    {"candidate": "20261016-101500-ab12cd3", "production": "...", "previous": "..."}

Promotion and rollback are just pointer moves, which is why they take seconds.
The pointer file is replaced atomically, so readers (the API) never see a half-written file.

CLI:  forecast-registry list | show | set <stage> <version>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from forecast.config import load_config
from forecast.exception import ForecastError, ModelNotFoundError, RegistryError
from forecast.logger import get_logger

log = get_logger(__name__)

STAGES = ("candidate", "production", "previous")


class ModelRegistry:
    def __init__(self, artifacts_dir: str | Path) -> None:
        self.root = Path(artifacts_dir)
        self.models_dir = self.root / "models"
        self.registry_file = self.root / "registry.json"

    def list_versions(self) -> list[str]:
        if not self.models_dir.exists():
            return []
        return sorted(
            p.name
            for p in self.models_dir.iterdir()
            if p.is_dir() and not p.name.endswith(".tmp") and (p / "model.joblib").exists()
        )

    def model_dir(self, version: str) -> Path:
        path = self.models_dir / version
        if not (path / "model.joblib").exists():
            raise ModelNotFoundError("Model version not found", version=version, path=str(path))
        return path

    def stages(self) -> dict[str, str]:
        if not self.registry_file.exists():
            return {}
        try:
            data = json.loads(self.registry_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError("Registry file is unreadable", path=str(self.registry_file)) from exc
        if not isinstance(data, dict):
            raise RegistryError("Registry file has the wrong shape", path=str(self.registry_file))
        return {k: v for k, v in data.items() if isinstance(v, str)}

    def get_stage(self, stage: str) -> str | None:
        return self.stages().get(stage)

    def set_stage(self, stage: str, version: str) -> None:
        if stage not in STAGES:
            raise RegistryError("Unknown stage", stage=stage, allowed=list(STAGES))
        self.model_dir(version)  # raises if the version does not exist
        data = self.stages()
        data[stage] = version
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_file.with_name(self.registry_file.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.registry_file)  # atomic
        log.info("Registry: %s -> %s", stage, version)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and move model registry pointers")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List all saved model versions")
    sub.add_parser("show", help="Show which version each stage points to")
    set_cmd = sub.add_parser("set", help="Point a stage at a version")
    set_cmd.add_argument("stage", choices=STAGES)
    set_cmd.add_argument("version")
    args = parser.parse_args(argv)

    try:
        registry = ModelRegistry(load_config(args.config).artifacts_dir)
        if args.command == "list":
            pointers = {v: k for k, v in registry.stages().items()}
            for version in registry.list_versions():
                print(f"{version}  {pointers.get(version, '')}".rstrip())
        elif args.command == "show":
            stages = registry.stages()
            for stage in STAGES:
                print(f"{stage:<11}{stages.get(stage, '-')}")
        else:
            registry.set_stage(args.stage, args.version)
            print(f"{args.stage} -> {args.version}")
    except ForecastError as exc:
        log.error("Registry command failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reinstall existing editable packages from their migrated Linux directories."""

import argparse
import importlib.metadata as metadata
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--environment", required=True, choices=("env_isaaclab", "soma-x"))
parser.add_argument("--verify_only", action="store_true")
args = parser.parse_args()
expected_prefix = Path("/home/nmt/miniconda3/envs") / args.environment
assert Path(sys.prefix) == expected_prefix
local = Path(__file__).resolve().parent
projects = local.parents[2]
old_root = Path("/mnt/c/Linux")
expected_repo = "IsaacLab" if args.environment == "env_isaaclab" else "SOMA-X"
site_packages = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
installed = {dist.metadata["Name"]: dist for dist in metadata.distributions(path=[str(site_packages)])}
before = {name: dist.version for name, dist in installed.items()}
packages = []
for dist in installed.values():
    direct_url = dist.read_text("direct_url.json")
    if not direct_url:
        continue
    info = json.loads(direct_url)
    if not info.get("dir_info", {}).get("editable"):
        continue
    parsed = urlparse(info["url"])
    source = Path(unquote(parsed.path))
    if parsed.scheme != "file" or not source.is_relative_to(old_root / expected_repo):
        continue
    target = projects / source.relative_to(old_root)
    assert target.is_dir() and (target / "setup.py").is_file(), target
    packages.append({"name": dist.metadata["Name"], "version": dist.version,
                     "old_source": str(source), "new_source": str(target)})
report_path = local / f"{args.environment}_editable_migration.json"
log_path = local / f"{args.environment}_editable_install.log"
if args.verify_only:
    report = json.loads(report_path.read_text())
    packages = report["packages"]
    before = report["versions_before"]
else:
    assert packages, "No old editable installs found; nothing to change."
    assert not report_path.exists(), "Preserve the existing migration report."
    report = {"environment": args.environment, "packages": packages, "versions_before": before}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    command = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--no-index"]
    for package in packages:
        command += ["--editable", package["new_source"]]
    print(f"Relinking {len(packages)} existing packages in {args.environment}; log: {log_path}", flush=True)
    with log_path.open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    report["install_exit_code"] = result.returncode
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    assert result.returncode == 0, log_path
installed = {dist.metadata["Name"]: dist for dist in metadata.distributions(path=[str(site_packages)])}
after = {name: dist.version for name, dist in installed.items()}
report["package_versions_unchanged"] = before == after
report["new_direct_urls"] = {
    package["name"]: json.loads(installed[package["name"]].read_text("direct_url.json"))
    for package in packages
}
report_path.write_text(json.dumps(report, indent=2) + "\n")
print("\n".join(log_path.read_text(errors="replace").splitlines()[-8:]))
print(f"Package versions unchanged: {report['package_versions_unchanged']}")
assert report["package_versions_unchanged"], "Inspect unexpected dependency/version changes."
assert all(info["url"].startswith(projects.as_uri() + "/") for info in report["new_direct_urls"].values())

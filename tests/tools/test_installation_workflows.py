# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_installation_guide_uses_uv_for_infinigen_setup():
    installation = (REPO_ROOT / "docs" / "Installation.md").read_text()

    assert "uv python install 3.11" in installation
    assert "uv sync --extra dev --extra terrain --extra vis" in installation
    assert "conda create --name infinigen" not in installation


def test_dockerfile_uses_uv_managed_python_environment():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()

    assert "ghcr.io/astral-sh/uv:0.8.22" in dockerfile
    assert "python:3.11-bookworm" in dockerfile
    assert "uv sync --frozen --extra dev --python 3.11" in dockerfile
    assert "conda create --name infinigen" not in dockerfile


def test_makefile_exposes_optional_docker_platform_override():
    makefile = (REPO_ROOT / "Makefile").read_text()

    assert "DOCKER_PLATFORM ?=" in makefile
    assert "--platform $(DOCKER_PLATFORM)" in makefile


def test_ci_workflow_installs_and_runs_with_uv():
    workflow = (REPO_ROOT / ".github" / "workflows" / "checks.yml").read_text()

    assert "python -m pip install uv" in workflow
    assert "uv sync --frozen --extra dev" in workflow
    assert "uv run pytest tests -k 'not skip_for_ci'" in workflow

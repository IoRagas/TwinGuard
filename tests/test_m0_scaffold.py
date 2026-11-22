import importlib
from pathlib import Path
import pytest

def test_project_structure():
    """Verify all required top-level directories and files exist."""
    root = Path(__file__).resolve().parent.parent
    expected_files = ["README.md", "AGENTS.md", "requirements.txt", ".gitignore"]
    for fname in expected_files:
        assert (root / fname).exists(), f"Missing file: {fname}"

    expected_packages = ["sim", "attacks", "twin", "detect", "agent", "eval", "demo", "tests"]
    for pkg in expected_packages:
        pkg_dir = root / pkg
        assert pkg_dir.is_dir(), f"Missing directory: {pkg}"
        assert (pkg_dir / "__init__.py").exists(), f"Missing __init__.py in: {pkg}"

def test_imports():
    """Verify that all core project packages can be imported."""
    packages = ["sim", "attacks", "twin", "detect", "agent", "eval", "demo"]
    for pkg in packages:
        mod = importlib.import_module(pkg)
        assert mod is not None

def test_dependencies():
    """Verify required core third-party dependencies are available."""
    dependencies = ["numpy", "scipy", "pandas", "matplotlib", "plotly", "streamlit", "pytest"]
    for dep in dependencies:
        mod = importlib.import_module(dep)
        assert mod is not None

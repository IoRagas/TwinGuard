"""Tests for Milestone 6: Streamlit Demo and Pre-Recorded Fallback."""
import time
from pathlib import Path
import pytest

from sim.simulator import Simulator
from attacks.spoof_drift import SpoofDriftAttack
from agent.state_machine import RecoveryAction


def test_fallback_gif_exists():
    """Verify pre-recorded fallback GIF exists and is non-empty."""
    root = Path(__file__).resolve().parent.parent
    gif_path = root / "demo" / "fallback_demo.gif"
    assert gif_path.exists(), "demo/fallback_demo.gif is missing"
    assert gif_path.stat().st_size > 100_000, "demo/fallback_demo.gif is unexpectedly small (<100KB)"


def test_demo_execution_time_under_60s():
    """Milestone 6 core requirement: Full attack-to-response demo runs in under 60 s."""
    start_t = time.time()
    sim = Simulator(seed=1042, total_time=130.0)
    attack = SpoofDriftAttack(start_time=50.0, drift_rate=0.8, stealth_velocity=True)
    res = sim.run_mission(attack=attack, preferred_response=RecoveryAction.RETURN)
    elapsed = time.time() - start_t

    print(f"\n[M6 Demo Execution Time] Completed full 130s flight simulation in {elapsed:.2f} s")
    assert elapsed < 60.0, f"Demo simulation took {elapsed:.2f} s, which exceeds 60s limit"
    assert res.is_safe
    assert res.detection_time is not None


def test_app_syntax_and_imports():
    """Verify demo/app.py compiles cleanly without syntax errors."""
    root = Path(__file__).resolve().parent.parent
    app_path = root / "demo" / "app.py"
    assert app_path.exists()
    with open(app_path, "r", encoding="utf-8") as f:
        code = f.read()
    compile(code, str(app_path), "exec")

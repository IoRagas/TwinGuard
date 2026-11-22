"""Tests for Milestone 4: Safety policy state machine, recovery, and explainer."""
import numpy as np
import pytest

from sim.simulator import Simulator
from attacks.spoof_jump import SpoofJumpAttack
from attacks.spoof_drift import SpoofDriftAttack
from attacks.jamming import GPSJammingAttack
from agent.state_machine import DroneSafetyState, RecoveryAction


def test_spoof_jump_safe_recovery():
    """Verify drone detects spoof jump, decouples GPS, returns, and lands safely."""
    sim = Simulator(seed=42, total_time=150.0)
    attack = SpoofJumpAttack(start_time=50.0, jump_distance=50.0)
    res = sim.run_mission(attack=attack, preferred_response=RecoveryAction.RETURN)

    # 1. State machine transitioned to UNDER_ATTACK
    attack_transitions = [tr for tr in res.fsm_transitions if tr.to_state == DroneSafetyState.UNDER_ATTACK]
    assert len(attack_transitions) > 0, "Drone never confirmed UNDER_ATTACK"

    # 2. Drone transitioned to RESPONSE (RETURN)
    return_transitions = [tr for tr in res.fsm_transitions if tr.action == RecoveryAction.RETURN]
    assert len(return_transitions) > 0, "Drone never initiated RETURN recovery"

    # 3. Transitions are chronologically logged
    timestamps = [tr.timestamp for tr in res.fsm_transitions]
    assert timestamps == sorted(timestamps)

    # 4. Drone ends in safe state (LAND or near home)
    assert res.is_safe, f"Drone was not safe at mission end: final speed={np.linalg.norm(res.true_vel[-1]):.2f}"
    assert res.states[-1] in [DroneSafetyState.LAND, DroneSafetyState.RESPONSE]

    # 5. Explainer produced incident report
    assert "TWINGUARD INCIDENT RESPONSE SUMMARY" in res.incident_report
    assert "UNDER_ATTACK" in res.incident_report
    print(f"\n[M4 Spoof Jump Recovery] Final distance to home: {res.final_distance_to_home:.2f} m")


def test_spoof_drift_safe_recovery():
    """Verify stealthy spoof drift triggers CUSUM and drone safely recovers."""
    sim = Simulator(seed=42, total_time=150.0)
    attack = SpoofDriftAttack(start_time=50.0, drift_rate=0.8, stealth_velocity=True)
    res = sim.run_mission(attack=attack, preferred_response=RecoveryAction.RETURN)

    attack_transitions = [tr for tr in res.fsm_transitions if tr.to_state == DroneSafetyState.UNDER_ATTACK]
    assert len(attack_transitions) > 0, "Drone never detected spoof drift"

    assert res.is_safe
    print(f"\n[M4 Spoof Drift Recovery] Final distance to home: {res.final_distance_to_home:.2f} m")


def test_jamming_safe_recovery():
    """Verify jamming dropout triggers failsafe return/landing."""
    sim = Simulator(seed=42, total_time=150.0)
    attack = GPSJammingAttack(start_time=50.0, duration=20.0, mode="dropout")
    res = sim.run_mission(attack=attack, preferred_response=RecoveryAction.RETURN)

    attack_transitions = [tr for tr in res.fsm_transitions if tr.to_state == DroneSafetyState.UNDER_ATTACK]
    assert len(attack_transitions) > 0, "Drone never detected GPS Jamming"

    assert res.is_safe
    print(f"\n[M4 Jamming Recovery] Final distance to home: {res.final_distance_to_home:.2f} m")


def test_hold_recovery_policy():
    """Verify drone can execute HOLD recovery maneuver on demand."""
    sim = Simulator(seed=42, total_time=100.0)
    attack = SpoofJumpAttack(start_time=40.0, jump_distance=30.0)
    res = sim.run_mission(attack=attack, preferred_response=RecoveryAction.HOLD)

    hold_transitions = [tr for tr in res.fsm_transitions if tr.action == RecoveryAction.HOLD]
    assert len(hold_transitions) > 0, "Drone failed to execute HOLD action"
    assert res.is_safe

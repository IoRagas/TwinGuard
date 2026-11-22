"""Deterministic plain-language incident explainer for flight security events."""
from typing import Optional, List
import numpy as np

from agent.state_machine import StateTransition, DroneSafetyState, RecoveryAction


class IncidentExplainer:
    """Generates human-auditable, plain-language explanations of security incidents."""

    def explain_transition(self, transition: StateTransition) -> str:
        """Generate clear incident report for a specific state machine transition."""
        t_sec = transition.timestamp
        state_str = f"{transition.from_state.value} -> {transition.to_state.value}"
        act_str = transition.action.value

        lines = [
            f"[SECURITY EVENT @ t={t_sec:.2f}s]",
            f"Transition: {state_str} | Active Action: {act_str}",
            f"Root Cause: {transition.reason}",
        ]

        if transition.details:
            lines.append("Telemetry Evidence Block (Untrusted Data Isolation):")
            lines.append("```json")
            lines.append(f"{transition.details}")
            lines.append("```")

        return "\n".join(lines)

    def explain_flight_summary(self, transitions: List[StateTransition], final_pos: np.ndarray) -> str:
        """Generate an executive summary of the entire flight incident log."""
        if not transitions:
            return "Flight completed normally with ZERO security anomalies detected."

        attack_transitions = [t for t in transitions if t.to_state == DroneSafetyState.UNDER_ATTACK]
        final_state = transitions[-1].to_state.value
        final_dist = float(np.linalg.norm(final_pos))

        report = [
            "=== TWINGUARD INCIDENT RESPONSE SUMMARY ===",
            f"Total State Transitions: {len(transitions)}",
            f"Final Drone State: {final_state}",
            f"Final Distance to Home Base: {final_dist:.2f} m",
        ]

        if attack_transitions:
            first_attack = attack_transitions[0]
            report.append(f"Attack Confirmed At: t={first_attack.timestamp:.2f} s")
            report.append(f"Primary Detection Reason: {first_attack.reason}")

        report.append("\nDetailed Transition Chronology:")
        for idx, tr in enumerate(transitions):
            report.append(f"  {idx + 1}. [t={tr.timestamp:6.2f}s] {tr.from_state.value:<12} -> {tr.to_state.value:<12} | {tr.reason}")

        return "\n".join(report)

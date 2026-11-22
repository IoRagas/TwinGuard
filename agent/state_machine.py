"""Deterministic safety policy state machine for autonomous cyber-physical recovery."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from detect.fusion import DetectionStatus
from twin.estimator import TwinEstimator


class DroneSafetyState(Enum):
    NORMAL = "NORMAL"
    SUSPECT = "SUSPECT"
    UNDER_ATTACK = "UNDER_ATTACK"
    RESPONSE = "RESPONSE"
    LAND = "LAND"
    RECOVERED = "RECOVERED"


class RecoveryAction(Enum):
    NONE = "NONE"
    HOLD = "HOLD"
    RETURN = "RETURN"
    LAND = "LAND"


@dataclass
class StateTransition:
    """Immutable audit record for a state machine transition."""
    timestamp: float
    from_state: DroneSafetyState
    to_state: DroneSafetyState
    action: RecoveryAction
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


class SafetyStateMachine:
    """Governs autonomous response transitions based on statistical detection evidence."""

    def __init__(
        self,
        confirmation_time_sec: float = 1.0,     # Time in SUSPECT before confirming UNDER_ATTACK
        max_uncertainty_limit: float = 25.0,     # 3-sigma pos uncertainty limit before forced LAND
        home_arrival_threshold: float = 12.0,   # Proximity to home (0,0) to execute landing
        preferred_response: RecoveryAction = RecoveryAction.RETURN,
    ):
        self.state = DroneSafetyState.NORMAL
        self.action = RecoveryAction.NONE
        self.confirmation_time_sec = confirmation_time_sec
        self.max_uncertainty_limit = max_uncertainty_limit
        self.home_arrival_threshold = home_arrival_threshold
        self.preferred_response = preferred_response

        self.suspect_start_time: Optional[float] = None
        self.hold_position: Optional[np.ndarray] = None
        self.transition_history: List[StateTransition] = []

    def reset(self):
        self.state = DroneSafetyState.NORMAL
        self.action = RecoveryAction.NONE
        self.suspect_start_time = None
        self.hold_position = None
        self.transition_history.clear()

    def update(
        self,
        timestamp: float,
        detection: DetectionStatus,
        estimator: TwinEstimator,
    ) -> Tuple[DroneSafetyState, RecoveryAction]:
        """Evaluate latest detection status and twin state to advance state machine."""
        pos_uncertainty = estimator.pos_uncertainty
        twin_pos = estimator.pos

        # Check for emergency landing condition if uncertainty explodes while in dead-reckoning
        if self.state in [DroneSafetyState.UNDER_ATTACK, DroneSafetyState.RESPONSE]:
            if pos_uncertainty > self.max_uncertainty_limit:
                self._transition_to(
                    timestamp,
                    DroneSafetyState.LAND,
                    RecoveryAction.LAND,
                    f"Dead-reckoning 3-sigma uncertainty ({pos_uncertainty:.1f}m) exceeded safety limit ({self.max_uncertainty_limit:.1f}m)",
                    {"pos_uncertainty": pos_uncertainty, "twin_pos": twin_pos.tolist()},
                )
                return self.state, self.action

        # State dispatch
        if self.state == DroneSafetyState.NORMAL:
            if detection.jamming_alarm:
                # Jamming / complete GPS dropout immediately escalates
                estimator.decouple_gps()
                self._transition_to(
                    timestamp,
                    DroneSafetyState.UNDER_ATTACK,
                    RecoveryAction.NONE,
                    "Immediate GPS Jamming/Signal Loss detected",
                    {"jamming": True},
                )
            elif detection.chi2_alarm and detection.cusum_alarm:
                # Dual detector trigger: immediate confirmation
                estimator.decouple_gps()
                self._transition_to(
                    timestamp,
                    DroneSafetyState.UNDER_ATTACK,
                    RecoveryAction.NONE,
                    f"Concurrent Chi-Square ({detection.chi2_stat:.1f}) and CUSUM ({detection.cusum_stat:.1f}) alarm trigger",
                    {"chi2_stat": detection.chi2_stat, "cusum_stat": detection.cusum_stat},
                )
            elif detection.chi2_alarm or detection.cusum_alarm:
                self.suspect_start_time = timestamp
                det_name = "Chi-Square" if detection.chi2_alarm else "CUSUM"
                val = detection.chi2_stat if detection.chi2_alarm else detection.cusum_stat
                self._transition_to(
                    timestamp,
                    DroneSafetyState.SUSPECT,
                    RecoveryAction.NONE,
                    f"Initial anomaly detected by {det_name} (statistic={val:.1f}); elevating surveillance",
                    {"detector": det_name, "statistic": val},
                )

        elif self.state == DroneSafetyState.SUSPECT:
            if detection.jamming_alarm or (detection.chi2_alarm and detection.cusum_alarm):
                estimator.decouple_gps()
                self._transition_to(
                    timestamp,
                    DroneSafetyState.UNDER_ATTACK,
                    RecoveryAction.NONE,
                    "Anomaly escalated to confirmed attack: multiple detector triggers",
                    {"detection": str(detection)},
                )
            elif detection.chi2_alarm or detection.cusum_alarm:
                elapsed = timestamp - (self.suspect_start_time or timestamp)
                if elapsed >= self.confirmation_time_sec:
                    estimator.decouple_gps()
                    self._transition_to(
                        timestamp,
                        DroneSafetyState.UNDER_ATTACK,
                        RecoveryAction.NONE,
                        f"Anomaly sustained for {elapsed:.2f}s >= confirmation threshold ({self.confirmation_time_sec:.1f}s)",
                        {"sustained_time": elapsed},
                    )
            else:
                # Anomaly disappeared before confirmation threshold
                self.suspect_start_time = None
                self._transition_to(
                    timestamp,
                    DroneSafetyState.NORMAL,
                    RecoveryAction.NONE,
                    "Transient anomaly resolved; resuming normal mission",
                    {},
                )

        elif self.state == DroneSafetyState.UNDER_ATTACK:
            # Transition to chosen safe RESPONSE maneuver
            if self.preferred_response == RecoveryAction.RETURN:
                self._transition_to(
                    timestamp,
                    DroneSafetyState.RESPONSE,
                    RecoveryAction.RETURN,
                    "Confirmed cyber attack. GPS decoupled. Executing dead-reckoning RETURN to home (0,0)",
                    {"twin_pos": twin_pos.tolist()},
                )
            elif self.preferred_response == RecoveryAction.HOLD:
                self.hold_position = twin_pos.copy()
                self._transition_to(
                    timestamp,
                    DroneSafetyState.RESPONSE,
                    RecoveryAction.HOLD,
                    f"Confirmed cyber attack. GPS decoupled. Executing HOLD hover at [{twin_pos[0]:.1f}, {twin_pos[1]:.1f}]",
                    {"hold_pos": twin_pos.tolist()},
                )
            else:
                self._transition_to(
                    timestamp,
                    DroneSafetyState.LAND,
                    RecoveryAction.LAND,
                    "Confirmed cyber attack. Executing emergency vertical LAND",
                    {},
                )

        elif self.state == DroneSafetyState.RESPONSE:
            if self.action == RecoveryAction.RETURN:
                # Check proximity to home (0,0)
                dist_to_home = float(np.linalg.norm(twin_pos))
                if dist_to_home <= self.home_arrival_threshold:
                    self._transition_to(
                        timestamp,
                        DroneSafetyState.LAND,
                        RecoveryAction.LAND,
                        f"Home waypoint reached (distance {dist_to_home:.1f}m <= {self.home_arrival_threshold:.1f}m); landing",
                        {"dist_to_home": dist_to_home},
                    )

        return self.state, self.action

    def _transition_to(
        self,
        timestamp: float,
        new_state: DroneSafetyState,
        action: RecoveryAction,
        reason: str,
        details: Dict[str, Any],
    ):
        transition = StateTransition(
            timestamp=timestamp,
            from_state=self.state,
            to_state=new_state,
            action=action,
            reason=reason,
            details=details,
        )
        self.transition_history.append(transition)
        self.state = new_state
        self.action = action

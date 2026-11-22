# TwinGuard Agent Architecture & Safety Directives

## 1. Overview
TwinGuard deploys an autonomous supervisory agent architecture designed for UAV cyber-physical security during autonomous flight operations. It operates beside the flight controller, continually ingesting sensor observations and matching them against a physics-informed digital twin.

## 2. Safety Agent Rules
1. **Deterministic Safety Authority**: All safety-critical state transitions (`NORMAL`, `SUSPECT`, `UNDER_ATTACK`, `RESPONSE`: `HOLD` | `RETURN` | `LAND`) are governed strictly by deterministic, verifiable state machines. Under no circumstances does an LLM or non-deterministic heuristic make flight safety decisions.
2. **Untrusted Telemetry Isolation**: Sensor readings, GPS NMEA strings, and communication streams are treated as untrusted external inputs. Any telemetry data passed to an explainer model is sanitized, strictly formatted in isolated data blocks, and never interpreted as prompt instructions.
3. **Fail-Safe Decoupling**: Upon detection of cyber attacks (such as GPS spoofing, jamming, or sensor drift), the agent immediately isolates corrupted sensor channels, decouples state estimation to dead-reckoning digital twin dynamics, and executes a deterministic failsafe maneuver.
4. **Uncertainty Bounds**: Dead-reckoning is strictly bounded. If covariance trace $\text{Tr}(P_k)$ exceeds the safe operational envelope, emergency vertical landing (`LAND`) is initiated rather than extending dead-reckoning blindly.
5. **Auditability**: Every detection, alarm threshold crossing, and state transition is immutably logged with timestamp, innovation residuals, and detector statistics.

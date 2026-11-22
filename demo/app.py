"""TwinGuard Interactive Streamlit Demo Application (NICAT AeroHack 2026)."""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sim.simulator import Simulator
from attacks.spoof_jump import SpoofJumpAttack
from attacks.spoof_drift import SpoofDriftAttack
from attacks.jamming import GPSJammingAttack
from agent.state_machine import RecoveryAction, DroneSafetyState
from detect.baseline import BaselineDetector


st.set_page_config(
    page_title="TwinGuard | Cyber-Physical Digital Twin",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ TwinGuard: Autonomous Drone Cyber-Physical Defense")
st.caption("NICAT AeroHack 2026 | Physics-Based Digital Twin with Statistical Detection & Deterministic Recovery")


# ==========================================
# Sidebar Configuration
# ==========================================
st.sidebar.header("🕹️ Flight & Attack Controls")

attack_category = st.sidebar.selectbox(
    "Attack Scenario",
    [
        "Spoof Drift (Stealthy Ramp) [Core Showcase]",
        "Spoof Jump (Abrupt Step)",
        "GPS Jamming (Signal Dropout)",
        "GPS Jamming (High-Noise Burst)",
        "Clean Flight (Zero Attacks)",
    ],
    index=0,
)

if "Spoof Drift" in attack_category:
    drift_rate = st.sidebar.slider("Drift Speed (m/s)", min_value=0.2, max_value=2.0, value=0.8, step=0.1)
    attack_start = st.sidebar.slider("Attack Onset Time (s)", min_value=20.0, max_value=80.0, value=50.0, step=5.0)
    attack_obj = SpoofDriftAttack(start_time=attack_start, drift_rate=drift_rate, stealth_velocity=True)
elif "Spoof Jump" in attack_category:
    jump_dist = st.sidebar.slider("Jump Offset Distance (m)", min_value=20.0, max_value=100.0, value=40.0, step=10.0)
    attack_start = st.sidebar.slider("Attack Onset Time (s)", min_value=20.0, max_value=80.0, value=50.0, step=5.0)
    attack_obj = SpoofJumpAttack(start_time=attack_start, jump_distance=jump_dist)
elif "Signal Dropout" in attack_category:
    jam_dur = st.sidebar.slider("Dropout Duration (s)", min_value=5.0, max_value=30.0, value=15.0, step=5.0)
    attack_start = st.sidebar.slider("Attack Onset Time (s)", min_value=20.0, max_value=80.0, value=50.0, step=5.0)
    attack_obj = GPSJammingAttack(start_time=attack_start, duration=jam_dur, mode="dropout")
elif "High-Noise Burst" in attack_category:
    jam_dur = st.sidebar.slider("Burst Duration (s)", min_value=5.0, max_value=30.0, value=15.0, step=5.0)
    attack_start = st.sidebar.slider("Attack Onset Time (s)", min_value=20.0, max_value=80.0, value=50.0, step=5.0)
    attack_obj = GPSJammingAttack(start_time=attack_start, duration=jam_dur, mode="noise_burst")
else:
    attack_obj = None
    attack_start = 999.0

seed = st.sidebar.number_input("Deterministic Simulation Seed", min_value=0, max_value=9999, value=1042, step=1)
total_time = st.sidebar.slider("Simulation Total Duration (s)", min_value=60.0, max_value=160.0, value=130.0, step=10.0)

response_choice = st.sidebar.selectbox(
    "TwinGuard Recovery Policy",
    ["RETURN (Dead-Reckon Home)", "HOLD (Hover in Place)", "LAND (Immediate Descent)"],
    index=0,
)
pref_action = RecoveryAction.RETURN if "RETURN" in response_choice else (RecoveryAction.HOLD if "HOLD" in response_choice else RecoveryAction.LAND)

show_baseline_comp = st.sidebar.checkbox("Compare with Naive Fixed-Threshold Alarm", value=True)


# ==========================================
# Run Simulation
# ==========================================
@st.cache_data(show_spinner=False)
def simulate(seed_val, dur_val, attack_type_str, strength_val, action_str):
    sim = Simulator(seed=seed_val, total_time=dur_val)
    res = sim.run_mission(attack=attack_obj, preferred_response=pref_action)
    return res

strength_val = getattr(attack_obj, "drift_rate", getattr(attack_obj, "offset", 0.0)) if attack_obj else 0.0
res = simulate(seed, total_time, attack_category, str(strength_val), response_choice)


# Baseline evaluation on the same flight
baseline = BaselineDetector()
base_alarms = []
for m in res.gps_measurements:
    fired, _ = baseline.update(m)
    if fired and m.timestamp >= attack_start:
        base_alarms.append(m.timestamp)
base_fired = len(base_alarms) > 0
base_first_time = base_alarms[0] if base_fired else None


# ==========================================
# Status Banners & Metrics
# ==========================================
col_tg, col_base, col_stat = st.columns([1.5, 1.5, 1.5])

final_state = res.states[-1].value
tg_detected = res.detection_time is not None

with col_tg:
    if tg_detected:
        st.error(f"🚨 **TwinGuard Status**: UNDER ATTACK confirmed at t={res.detection_time:.2f}s\n\nActive State: `{final_state}` | Action: `{res.actions[-1].value}`")
    else:
        st.success(f"✅ **TwinGuard Status**: NORMAL\n\nState: `{final_state}` | No Anomaly Detected")

with col_base:
    if show_baseline_comp:
        if base_fired:
            st.error(f"⚠️ **Naive Baseline**: ALARM fired at t={base_first_time:.2f}s\n\nDiscontinuity detected on raw GPS")
        else:
            st.warning("⚠️ **Naive Baseline**: BLIND (No Alarm)\n\nFailed to detect attack! (Smooth drift fooled delta-jump threshold)")

with col_stat:
    dist_home = res.final_distance_to_home
    max_unc = float(np.max(res.twin_uncertainty))
    st.info(f"📍 **Final Status**\n\nDist to Home: `{dist_home:.1f} m` | Max 3σ Unc: `{max_unc:.1f} m` | Safe: `{res.is_safe}`")


# ==========================================
# Interactive Replay Time Scrubber
# ==========================================
scrub_time = st.slider("⏱️ Flight Scrub Time (s)", min_value=0.0, max_value=float(total_time), value=float(total_time), step=0.5)
scrub_idx = min(len(res.times) - 1, int(round(scrub_time / 0.02)))


# ==========================================
# 2D Map Trajectory
# ==========================================
st.subheader("🗺️ 2D Drone Trajectory: Real vs. Spoofed vs. Twin")

fig_map = go.Figure()

# 1. True path
fig_map.add_trace(go.Scatter(
    x=res.true_pos[:scrub_idx, 0],
    y=res.true_pos[:scrub_idx, 1],
    mode="lines",
    name="True Ground Truth Path",
    line=dict(color="#2ca02c", width=3),
))

# 2. GPS Reported path
fig_map.add_trace(go.Scatter(
    x=res.gps_reported_pos[:scrub_idx, 0],
    y=res.gps_reported_pos[:scrub_idx, 1],
    mode="lines",
    name="GPS Reported Path (Spoofed/Corrupted)",
    line=dict(color="#d62728", width=2, dash="dash"),
))

# 3. Twin Estimated path
fig_map.add_trace(go.Scatter(
    x=res.twin_est_pos[:scrub_idx, 0],
    y=res.twin_est_pos[:scrub_idx, 1],
    mode="lines",
    name="Digital Twin Path (IMU + Decoupled Dead-Reckoning)",
    line=dict(color="#1f77b4", width=2.5, dash="dot"),
))

# Current Drone Position
fig_map.add_trace(go.Scatter(
    x=[res.true_pos[scrub_idx, 0]],
    y=[res.true_pos[scrub_idx, 1]],
    mode="markers",
    name="Current Drone Position",
    marker=dict(size=14, color="#2ca02c", symbol="cross"),
))

# Waypoints & Home
wp_x = [0.0, 400.0, 400.0, 0.0, 0.0]
wp_y = [0.0, 0.0, 300.0, 300.0, 0.0]
fig_map.add_trace(go.Scatter(
    x=wp_x,
    y=wp_y,
    mode="lines+markers",
    name="Patrol Waypoints (400x300m)",
    line=dict(color="grey", width=1, dash="longdash"),
    marker=dict(size=8, symbol="square", color="black"),
))

fig_map.update_layout(
    xaxis_title="X Position (meters)",
    yaxis_title="Y Position (meters)",
    height=550,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
)
st.plotly_chart(fig_map, use_container_width=True)


# ==========================================
# Diagnostic Plots: Residuals, CUSUM, Uncertainty
# ==========================================
col_diag1, col_diag2 = st.columns(2)

with col_diag1:
    st.subheader("📈 Statistical Anomaly Detectors")
    fig_det = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Normalized Innovation Squared (NIS)", "Vector CUSUM Accumulator"])

    # NIS trace
    nis_t = [t for t in res.nis_times if t <= scrub_time]
    nis_v = res.nis_values[:len(nis_t)]
    fig_det.add_trace(go.Scatter(x=nis_t, y=nis_v, mode="lines", name="NIS", line=dict(color="#1f77b4")), row=1, col=1)
    fig_det.add_hline(y=48.0, line_dash="dash", line_color="red", annotation_text="Chi2 Threshold (48.0)", row=1, col=1)

    # CUSUM trace
    cusum_t = [s.timestamp for s in res.detection_statuses if s.timestamp <= scrub_time]
    cusum_v = [s.cusum_stat for s in res.detection_statuses if s.timestamp <= scrub_time]
    fig_det.add_trace(go.Scatter(x=cusum_t, y=cusum_v, mode="lines", name="CUSUM", line=dict(color="#ff7f0e")), row=2, col=1)
    fig_det.add_hline(y=9.0, line_dash="dash", line_color="red", annotation_text="CUSUM Threshold (9.0)", row=2, col=1)

    fig_det.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
    st.plotly_chart(fig_det, use_container_width=True)

with col_diag2:
    st.subheader("🔍 State Uncertainty & Ground Speed")
    fig_aux = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Dead-Reckon 3-Sigma Position Uncertainty (m)", "Ground Speed (m/s)"])

    times_scrub = res.times[:scrub_idx]
    unc_scrub = res.twin_uncertainty[:scrub_idx]
    speed_scrub = np.linalg.norm(res.true_vel[:scrub_idx], axis=1)

    fig_aux.add_trace(go.Scatter(x=times_scrub, y=unc_scrub, mode="lines", name="Uncertainty", line=dict(color="#9467bd")), row=1, col=1)
    fig_aux.add_hline(y=25.0, line_dash="dash", line_color="red", annotation_text="Safety Envelope Limit (25m)", row=1, col=1)

    fig_aux.add_trace(go.Scatter(x=times_scrub, y=speed_scrub, mode="lines", name="Speed", line=dict(color="#2ca02c")), row=2, col=1)
    fig_aux.add_hline(y=10.0, line_dash="dash", line_color="grey", annotation_text="Max Speed (10 m/s)", row=2, col=1)

    fig_aux.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
    st.plotly_chart(fig_aux, use_container_width=True)


# ==========================================
# Plain-Language Explainer & Audit Trail
# ==========================================
st.subheader("📜 Plain-Language Incident Report & Audit Chronology")
st.code(res.incident_report, language="yaml")

# TwinGuard Rigorous Empirical Evaluation Results

> **Evaluation Protocol**:
> - Detector Tuning Seeds: `0 - 99` (Used strictly for detector calibration).
> - Reporting Seeds: `1000 - 1199` (Total 200 evaluation runs; zero data leakage).
> - Physics Simulation: 2D Quadrotor with aerodynamic drag, 50 Hz IMU, 5 Hz GPS, random walk wind disturbance.

| Metric | Naive Fixed-Threshold Alarm | TwinGuard Digital Twin (Chi-Square + CUSUM) |
| :--- | :---: | :---: |
| **Spoof Jump Detection Rate (20-100m)** | 100.0% | **100.0%** |
| **Spoof Jump Median Detection Latency** | 0.00 s | **0.00 s** |
| **Spoof Drift Detection Rate (0.2-2.0 m/s)** | 0.0% | **68.6%** |
| **Spoof Drift Median Detection Latency** | N/A (Never Detected) | **0.20 s** |
| **GPS Jamming Detection Rate** | 100.0% | **100.0%** |
| **GPS Jamming Median Detection Latency** | 2.60 s | **0.00 s** |
| **False Alarm Rate (Clean Flights)** | 0.0% (0/50) | **0.0% (0/50)** |
| **Safe Recovery Rate (Attacked Flights)** | 0.0% (No Recovery Logic) | **100.0%** |
| **Mean Final Distance to Home After Safe Land** | N/A (Drone flies away) | **168.46 m** |

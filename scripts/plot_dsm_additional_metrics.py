import matplotlib.pyplot as plt
import numpy as np

np.random.seed(2026)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

# ---------------------------------------------------------
# Subplot 1: Mode Selection Ratio vs. Tau_bh
# ---------------------------------------------------------
tau_bh = np.linspace(0.5, 15, 300)
k = 0.65
tau_th = 6.0
p_offload = 100.0 / (1.0 + np.exp(k * (tau_bh - tau_th)))
p_local = 100.0 - p_offload

ax1.plot(tau_bh, p_offload, color='#d62728', linestyle='-', linewidth=2.2, label='Central Offloading Ratio (%)')
ax1.plot(tau_bh, p_local, color='#1f77b4', linestyle='-', linewidth=2.2, label='Local VQ Processing Ratio (%)')
ax1.axvline(x=tau_th, color='gray', linestyle='--', alpha=0.7, label=r'Decision Boundary ($\tau_{th} = 6.0$)')

ax1.set_title(r"Mode Selection Ratio vs. Backhaul Threshold ($\tau_{bh}$)", fontsize=11, fontweight='bold', pad=10)
ax1.set_xlabel(r"Backhaul Quality Threshold ($\tau_{bh}$)", fontsize=10)
ax1.set_ylabel("Selection Probability (%)", fontsize=10)
ax1.set_xlim(0.5, 15)
ax1.set_ylim(0, 105)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(fontsize=8.5, loc='center right', framealpha=0.9)

# ---------------------------------------------------------
# Subplot 2: Dynamic Tau Tracking & Traffic Fluctuation (Dual Y-Axis)
# ---------------------------------------------------------
time = np.linspace(0, 100, 250)
traffic_load = 50.0 + 30.0 * np.sin(2 * np.pi * time / 35.0) + 12.0 * np.sin(2 * np.pi * time / 12.0) + np.random.normal(0, 2.5, size=len(time))
traffic_load = np.clip(traffic_load, 10.0, 95.0)

# Optimal tau dynamically tracking traffic load
tau_opt_dynamic = 1.5 + 0.11 * traffic_load + np.random.normal(0, 0.25, size=len(time))
tau_fixed_baseline = np.full_like(time, 6.0)

ax2_twin = ax2.twinx()

# Plot Traffic Load on right axis
l1 = ax2_twin.plot(time, traffic_load, color='gray', linestyle=':', linewidth=1.3, alpha=0.65, label='Network Traffic Load (%)')
ax2_twin.set_ylabel("Traffic Load (%)", fontsize=10, color='gray')
ax2_twin.tick_params(axis='y', labelcolor='gray')
ax2_twin.set_ylim(0, 110)

# Plot Tau tracking on left axis
l2 = ax2.plot(time, tau_opt_dynamic, color='#2ca02c', linestyle='-', linewidth=2.0, label=r'Proposed Dynamic DSM ($\tau^*(t)$)')
l3 = ax2.plot(time, tau_fixed_baseline, color='#d62728', linestyle='--', linewidth=1.8, label=r'Fixed Baseline ($\tau = 6.0$)')

ax2.set_title(r"Real-time Dynamic $\tau^*(t)$ Tracking under Fluctuation", fontsize=11, fontweight='bold', pad=10)
ax2.set_xlabel("Time Step (t)", fontsize=10)
ax2.set_ylabel(r"Backhaul Threshold $\tau^*(t)$", fontsize=10)
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 15)
ax2.grid(True, linestyle=':', alpha=0.6)

# Combine legends
lines = l2 + l3 + l1
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, fontsize=8.5, loc='upper right', framealpha=0.9)

plt.tight_layout()
output_path = "dsm_additional_metrics_analysis_v2.png"
plt.savefig(output_path, dpi=300)
plt.close()
print(f"Refined graph saved as {output_path}")
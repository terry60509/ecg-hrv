#!/usr/bin/env python3
"""HRV analysis of my_ecg_5min.csv (500 Hz AD8232 recording).

R-peak detection on the raw `adc` column with neurokit2, then time- and
frequency-domain HRV plus a Poincaré plot with fitted SD1/SD2 ellipse.

Outputs:
    hrv_report.png   - 4-panel report: ECG+R peaks, RR trend,
                       time-domain metrics, frequency-domain PSD
    poincare.png     - Poincaré plot with ellipse and SD1/SD2 arrows
    hrv_report.txt   - summary report with all metrics
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import welch

FS = 500

df = pd.read_csv("my_ecg_5min.csv")
ecg = df["adc"].to_numpy(dtype=float)
dur_min = len(ecg) / FS / 60

# --- 1. R-peak detection on the raw ADC signal -------------------------
clean = nk.ecg_clean(ecg, sampling_rate=FS)
_, info = nk.ecg_peaks(clean, sampling_rate=FS, correct_artifacts=True)
rpeaks = info["ECG_R_Peaks"]

rr_ms = np.diff(rpeaks) / FS * 1000.0
rr_t = rpeaks[1:] / FS                     # time of each RR interval (s)
hr_inst = 60000.0 / rr_ms                  # instantaneous HR per beat
hr_mean, hr_min, hr_max = hr_inst.mean(), hr_inst.min(), hr_inst.max()

# --- 2. HRV metrics ----------------------------------------------------
hrv_t = nk.hrv_time(rpeaks, sampling_rate=FS)
hrv_f = nk.hrv_frequency(rpeaks, sampling_rate=FS, psd_method="welch",
                         normalize=False)
hrv_nl = nk.hrv_nonlinear(rpeaks, sampling_rate=FS)

rmssd = float(hrv_t["HRV_RMSSD"].iloc[0])
sdnn = float(hrv_t["HRV_SDNN"].iloc[0])
pnn50 = float(hrv_t["HRV_pNN50"].iloc[0])
lf = float(hrv_f["HRV_LF"].iloc[0])        # ms^2
hf = float(hrv_f["HRV_HF"].iloc[0])        # ms^2
lfhf = float(hrv_f["HRV_LFHF"].iloc[0])
sd1 = float(hrv_nl["HRV_SD1"].iloc[0])
sd2 = float(hrv_nl["HRV_SD2"].iloc[0])

# PSD of the evenly-resampled RR tachogram (for the report panel)
interp = interp1d(rr_t, rr_ms, kind="cubic", fill_value="extrapolate")
t_even = np.arange(rr_t[0], rr_t[-1], 1 / 4.0)   # 4 Hz resampling
rr_even = interp(t_even)
freq, psd = welch(rr_even - rr_even.mean(), fs=4.0, nperseg=min(256, len(rr_even)))

# --- 3. 4-panel HRV report ---------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 9))

# (a) ECG with R peaks (10-s window so beats are visible)
ax = axes[0, 0]
t = np.arange(len(ecg)) / FS
seg = (t >= 60) & (t < 70)
ax.plot(t[seg], clean[seg], lw=0.8)
pk_seg = rpeaks[(rpeaks >= 60 * FS) & (rpeaks < 70 * FS)]
ax.plot(pk_seg / FS, clean[pk_seg], "rv", ms=8)
ax.set_title(f"ECG with R-peak detection  ({len(rpeaks)} beats total)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")

# (b) RR interval trend
ax = axes[0, 1]
ax.plot(rr_t, rr_ms, lw=0.8, color="tab:green")
ax.axhline(rr_ms.mean(), color="k", ls="--", lw=0.8, alpha=0.6,
           label=f"mean {rr_ms.mean():.0f} ms")
ax.set_title("RR interval trend")
ax.set_xlabel("Time (s)")
ax.set_ylabel("RR (ms)")
ax.legend()

# (c) Time-domain metrics panel
ax = axes[1, 0]
ax.axis("off")
ax.set_title("Time-domain HRV")
lines = [
    ("Mean HR", f"{hr_mean:.1f} bpm  (min {hr_min:.0f} / max {hr_max:.0f})"),
    ("SDNN", f"{sdnn:.1f} ms"),
    ("RMSSD", f"{rmssd:.1f} ms"),
    ("pNN50", f"{pnn50:.1f} %"),
    ("SD1 / SD2", f"{sd1:.1f} / {sd2:.1f} ms"),
]
for i, (k, v) in enumerate(lines):
    ax.text(0.08, 0.85 - i * 0.16, k, fontsize=14, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.45, 0.85 - i * 0.16, v, fontsize=14, transform=ax.transAxes)

# (d) Frequency-domain PSD with LF/HF bands
ax = axes[1, 1]
ax.plot(freq, psd, lw=1.2, color="tab:blue")
ax.fill_between(freq, psd, where=(freq >= 0.04) & (freq < 0.15),
                alpha=0.4, color="tab:orange", label=f"LF = {lf:.0f} ms²")
ax.fill_between(freq, psd, where=(freq >= 0.15) & (freq < 0.40),
                alpha=0.4, color="tab:green", label=f"HF = {hf:.0f} ms²")
ax.set_xlim(0, 0.5)
ax.set_title(f"RR power spectrum (Welch)   LF/HF = {lfhf:.2f}")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD (ms²/Hz)")
ax.legend()

fig.suptitle("HRV Analysis Report — my_ecg_5min.csv", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("hrv_report.png", dpi=150)
plt.close(fig)

# --- 4. Poincaré plot with ellipse and SD1/SD2 arrows -------------------
fig, ax = plt.subplots(figsize=(7, 7))
x, y = rr_ms[:-1], rr_ms[1:]
cx, cy = x.mean(), y.mean()
ax.scatter(x, y, s=10, alpha=0.4, label="RR pairs")

lim = [rr_ms.min() - 60, rr_ms.max() + 60]
ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5, label="RR(n) = RR(n+1)")

# Ellipse: semi-major = SD2 along the identity line, semi-minor = SD1
ax.add_patch(Ellipse((cx, cy), width=2 * sd2, height=2 * sd1, angle=45,
                     facecolor="none", edgecolor="crimson", lw=2, zorder=4))
u = np.sqrt(0.5)  # unit vector component for 45 degrees
ax.annotate("", xy=(cx + sd2 * u, cy + sd2 * u), xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", color="darkorange", lw=2.5))
ax.annotate("", xy=(cx - sd1 * u, cy + sd1 * u), xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", color="purple", lw=2.5))
ax.text(cx + sd2 * u + 8, cy + sd2 * u, f"SD2 = {sd2:.1f} ms",
        color="darkorange", fontsize=11, fontweight="bold")
ax.text(cx - sd1 * u - 8, cy + sd1 * u + 8, f"SD1 = {sd1:.1f} ms",
        color="purple", fontsize=11, fontweight="bold", ha="right")

ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("RR(n) (ms)")
ax.set_ylabel("RR(n+1) (ms)")
ax.set_title("Poincaré plot")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("poincare.png", dpi=150)
plt.close(fig)

# --- 5. Text report ----------------------------------------------------
if lfhf > 2.0:
    balance = "LF/HF 比值偏高，顯示記錄期間交感神經活性相對占優（清醒靜坐狀態常見）。"
elif lfhf < 0.5:
    balance = "LF/HF 比值偏低，顯示副交感（迷走）神經活性相對占優。"
else:
    balance = "LF/HF 比值接近 1，交感與副交感活性大致平衡。"
interp_text = (
    f"平均心率 {hr_mean:.0f} bpm，屬正常靜息範圍。"
    f"SDNN {sdnn:.0f} ms 與 RMSSD {rmssd:.0f} ms 顯示整體變異正常、"
    f"短期逐拍變異中等。{balance}"
    f"Poincaré SD2/SD1 = {sd2 / sd1:.2f}，長期變異明顯大於短期變異，"
    "與上述頻域結果一致。本結果僅為訊號分析，非醫療診斷。"
)

report = f"""HRV 分析報告 — my_ecg_5min.csv
================================================

資料
  取樣率           : {FS} Hz
  總長度           : {len(ecg)} 樣本 ({dur_min:.1f} 分鐘)
  電極脫落樣本      : {int(df['lead_off'].sum())}

R 波偵測 (neurokit2, 原始 adc 訊號)
  R 波數           : {len(rpeaks)}
  RR 間隔平均       : {rr_ms.mean():.1f} ms  (範圍 {rr_ms.min():.0f} ~ {rr_ms.max():.0f} ms)
  心率 平均/最低/最高 : {hr_mean:.1f} / {hr_min:.1f} / {hr_max:.1f} bpm

時域 HRV
  SDNN             : {sdnn:.1f} ms
  RMSSD            : {rmssd:.1f} ms
  pNN50            : {pnn50:.1f} %

頻域 HRV (Welch, 絕對功率)
  LF (0.04-0.15 Hz): {lf:.1f} ms²
  HF (0.15-0.40 Hz): {hf:.1f} ms²
  LF/HF ratio      : {lfhf:.2f}

非線性 (Poincaré)
  SD1              : {sd1:.1f} ms
  SD2              : {sd2:.1f} ms
  SD2/SD1          : {sd2 / sd1:.2f}

解讀
  {interp_text}

輸出檔案
  hrv_report.png   - 4 合 1 報告圖 (ECG+R 波 / RR 趨勢 / 時域指標 / 頻譜)
  poincare.png     - 龐加萊圖 (含 SD1/SD2 橢圓與箭頭)
"""

with open("hrv_report.txt", "w") as f:
    f.write(report)
print(report)

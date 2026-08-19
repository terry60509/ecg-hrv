#!/usr/bin/env python3
"""HRV analysis of my_ecg_5min.csv (500 Hz AD8232 recording).

R-peak detection on the raw `adc` column with neurokit2, then time- and
frequency-domain HRV plus a Poincaré plot.

Outputs:
    ecg_rpeaks.png   - ECG overview with detected R peaks
    poincare.png     - Poincaré plot RR(n) vs RR(n+1)
    hrv_report.txt   - summary report with all metrics
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd

FS = 500

df = pd.read_csv("my_ecg_5min.csv")
ecg = df["adc"].to_numpy(dtype=float)
dur_min = len(ecg) / FS / 60

# --- 1. R-peak detection on the raw ADC signal -------------------------
clean = nk.ecg_clean(ecg, sampling_rate=FS)
_, info = nk.ecg_peaks(clean, sampling_rate=FS, correct_artifacts=True)
rpeaks = info["ECG_R_Peaks"]

rr_ms = np.diff(rpeaks) / FS * 1000.0
mean_hr = 60000.0 / rr_ms.mean()

# --- 2. HRV metrics ----------------------------------------------------
hrv_t = nk.hrv_time(rpeaks, sampling_rate=FS)
hrv_f = nk.hrv_frequency(rpeaks, sampling_rate=FS)
hrv_nl = nk.hrv_nonlinear(rpeaks, sampling_rate=FS)

rmssd = float(hrv_t["HRV_RMSSD"].iloc[0])
sdnn = float(hrv_t["HRV_SDNN"].iloc[0])
pnn50 = float(hrv_t["HRV_pNN50"].iloc[0])
lf = float(hrv_f["HRV_LF"].iloc[0])
hf = float(hrv_f["HRV_HF"].iloc[0])
lfhf = float(hrv_f["HRV_LFHF"].iloc[0])
sd1 = float(hrv_nl["HRV_SD1"].iloc[0])
sd2 = float(hrv_nl["HRV_SD2"].iloc[0])

# --- 3. ECG overview with R peaks --------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 7))
t = np.arange(len(ecg)) / FS
axes[0].plot(t, clean, lw=0.4)
axes[0].plot(rpeaks / FS, clean[rpeaks], "r.", ms=4)
axes[0].set_title(f"ECG (cleaned) with {len(rpeaks)} R peaks — full recording")
axes[0].set_xlabel("Time (s)")

seg = (t >= 60) & (t < 70)  # 10-second zoom
axes[1].plot(t[seg], clean[seg], lw=0.8)
pk_seg = rpeaks[(rpeaks >= 60 * FS) & (rpeaks < 70 * FS)]
axes[1].plot(pk_seg / FS, clean[pk_seg], "rv", ms=8)
axes[1].set_title("Zoom: 60-70 s")
axes[1].set_xlabel("Time (s)")
fig.tight_layout()
fig.savefig("ecg_rpeaks.png", dpi=150)
plt.close(fig)

# --- 4. Poincaré plot --------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(rr_ms[:-1], rr_ms[1:], s=10, alpha=0.5)
lim = [rr_ms.min() - 50, rr_ms.max() + 50]
ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)  # identity line
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("RR(n) (ms)")
ax.set_ylabel("RR(n+1) (ms)")
ax.set_title(f"Poincaré plot  (SD1={sd1:.1f} ms, SD2={sd2:.1f} ms)")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("poincare.png", dpi=150)
plt.close(fig)

# --- 5. Text report ----------------------------------------------------
report = f"""HRV 分析報告 — my_ecg_5min.csv
================================================

資料
  取樣率           : {FS} Hz
  總長度           : {len(ecg)} 樣本 ({dur_min:.1f} 分鐘)
  電極脫落樣本      : {int(df['lead_off'].sum())}

R 波偵測 (neurokit2, 原始 adc 訊號)
  R 波數           : {len(rpeaks)}
  RR 間隔平均       : {rr_ms.mean():.1f} ms
  RR 間隔範圍       : {rr_ms.min():.0f} ~ {rr_ms.max():.0f} ms
  平均心率          : {mean_hr:.1f} bpm

時域 HRV
  RMSSD            : {rmssd:.1f} ms
  SDNN             : {sdnn:.1f} ms
  pNN50            : {pnn50:.1f} %

頻域 HRV
  LF (0.04-0.15 Hz): {lf:.4f} (normalized)
  HF (0.15-0.40 Hz): {hf:.4f} (normalized)
  LF/HF ratio      : {lfhf:.2f}

非線性 (Poincaré)
  SD1              : {sd1:.1f} ms
  SD2              : {sd2:.1f} ms
  SD2/SD1          : {sd2 / sd1:.2f}

輸出檔案
  ecg_rpeaks.png   - ECG 全程與 60-70 秒放大圖，R 波標記
  poincare.png     - 龐加萊圖
"""

with open("hrv_report.txt", "w") as f:
    f.write(report)
print(report)

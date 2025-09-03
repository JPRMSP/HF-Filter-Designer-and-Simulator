import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

st.set_page_config(page_title="HF Filter Designer", layout="wide")

st.title("📡 HF Filter Designer and Simulator")
st.markdown("An interactive tool to design and visualize High-Frequency Filters in real-time.")

# Sidebar Controls
st.sidebar.header("🔧 Filter Parameters")
filter_type = st.sidebar.selectbox("Select Filter Type", ["Lowpass", "Highpass", "Bandpass", "Bandstop"])
order = st.sidebar.slider("Filter Order (N)", 1, 10, 3)
fc = st.sidebar.slider("Cutoff Frequency (Hz)", 100, 5000, 1000, step=100)

if filter_type in ["Bandpass", "Bandstop"]:
    bandwidth = st.sidebar.slider("Bandwidth (Hz)", 100, 2000, 500, step=50)
else:
    bandwidth = None

approx_type = st.sidebar.selectbox("Approximation Method", ["Butterworth", "Chebyshev I", "Chebyshev II", "Elliptic"])

# Normalize frequency (digital prototype)
fs = 20000  # sampling freq for digital sim
nyq = fs / 2

if filter_type == "Lowpass":
    Wn = fc / nyq
elif filter_type == "Highpass":
    Wn = fc / nyq
elif filter_type in ["Bandpass", "Bandstop"]:
    Wn = [ (fc - bandwidth/2)/nyq , (fc + bandwidth/2)/nyq ]

# Filter Design
if approx_type == "Butterworth":
    b, a = signal.butter(order, Wn, btype=filter_type.lower())
elif approx_type == "Chebyshev I":
    b, a = signal.cheby1(order, 0.5, Wn, btype=filter_type.lower())
elif approx_type == "Chebyshev II":
    b, a = signal.cheby2(order, 40, Wn, btype=filter_type.lower())
elif approx_type == "Elliptic":
    b, a = signal.ellip(order, 0.5, 40, Wn, btype=filter_type.lower())

# Frequency Response
w, h = signal.freqz(b, a, worN=1024, fs=fs)

# Compute Phase Delay
phase = np.unwrap(np.angle(h))
group_delay = -np.diff(phase) / np.diff(w)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Magnitude Response")
    fig1, ax1 = plt.subplots()
    ax1.plot(w, 20*np.log10(np.abs(h)))
    ax1.set_title("Frequency Response (dB)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True)
    st.pyplot(fig1)

with col2:
    st.subheader("Phase Response")
    fig2, ax2 = plt.subplots()
    ax2.plot(w, phase)
    ax2.set_title("Phase Response")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (radians)")
    ax2.grid(True)
    st.pyplot(fig2)

st.subheader("Group Delay")
fig3, ax3 = plt.subplots()
ax3.plot(w[1:], group_delay)
ax3.set_title("Group Delay")
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Delay (samples)")
ax3.grid(True)
st.pyplot(fig3)

# Simple S11, S21 Parameters (magnitude only)
s11 = 1 - np.abs(h)**2
s21 = np.abs(h)

st.subheader("S-Parameters")
fig4, ax4 = plt.subplots()
ax4.plot(w, 20*np.log10(s11), label="S11 (Reflection)")
ax4.plot(w, 20*np.log10(s21), label="S21 (Transmission)")
ax4.set_title("Scattering Parameters")
ax4.set_xlabel("Frequency (Hz)")
ax4.set_ylabel("Magnitude (dB)")
ax4.legend()
ax4.grid(True)
st.pyplot(fig4)

st.markdown("✅ Designed using classical filter theory (no datasets, no ML).")

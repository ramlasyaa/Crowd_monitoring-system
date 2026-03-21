import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os
import time
from datetime import datetime, timedelta

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="CrowdSentry AI — Intelligent Crowd Monitoring",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "CrowdSentry AI | Crowd Monitoring Robot System | Final Year Project"
    }
)

# ============================================================
# GLOBAL DARK THEME CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* === GLOBAL RESET === */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1120 50%, #0a0f1e 100%);
    min-height: 100vh;
}

/* === SIDEBAR === */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1120 0%, #111827 100%) !important;
    border-right: 1px solid rgba(99, 179, 237, 0.15);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* === HERO BANNER === */
.hero-container {
    background: linear-gradient(135deg, rgba(30,41,59,0.9) 0%, rgba(15,23,42,0.95) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(99,179,237,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #63b3ed;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #e2e8f0 0%, #63b3ed 50%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 12px;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    font-weight: 400;
    line-height: 1.7;
    max-width: 680px;
}
.tech-stack {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 24px;
}
.tech-pill {
    background: rgba(167,139,250,0.1);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    color: #a78bfa;
}

/* === METRIC CARDS === */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 28px 0;
}
.metric-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.8) 0%, rgba(15,23,42,0.9) 100%);
    border-radius: 16px;
    padding: 24px 20px;
    border: 1px solid rgba(99,179,237,0.12);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.metric-card.blue::before   { background: linear-gradient(90deg, #63b3ed, #3182ce); }
.metric-card.green::before  { background: linear-gradient(90deg, #68d391, #38a169); }
.metric-card.red::before    { background: linear-gradient(90deg, #fc8181, #e53e3e); }
.metric-card.purple::before { background: linear-gradient(90deg, #a78bfa, #7c3aed); }

.metric-icon {
    font-size: 1.8rem;
    margin-bottom: 14px;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #e2e8f0;
    line-height: 1;
    margin-bottom: 6px;
    font-variant-numeric: tabular-nums;
}
.metric-delta {
    font-size: 0.8rem;
    font-weight: 500;
}
.metric-delta.up   { color: #fc8181; }
.metric-delta.down { color: #68d391; }
.metric-delta.neu  { color: #63b3ed; }

/* === ALERT BANNERS === */
.alert-critical {
    background: linear-gradient(135deg, rgba(229,62,62,0.15) 0%, rgba(197,48,48,0.1) 100%);
    border: 1px solid rgba(229,62,62,0.4);
    border-left: 4px solid #e53e3e;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 20px 0;
    display: flex;
    align-items: center;
    gap: 16px;
}
.alert-warning {
    background: linear-gradient(135deg, rgba(237,137,54,0.15) 0%, rgba(221,107,32,0.1) 100%);
    border: 1px solid rgba(237,137,54,0.35);
    border-left: 4px solid #ed8936;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 20px 0;
    display: flex;
    align-items: center;
    gap: 16px;
}
.alert-safe {
    background: linear-gradient(135deg, rgba(72,187,120,0.12) 0%, rgba(56,161,105,0.08) 100%);
    border: 1px solid rgba(72,187,120,0.3);
    border-left: 4px solid #48bb78;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 20px 0;
    display: flex;
    align-items: center;
    gap: 16px;
}
.alert-icon  { font-size: 2rem; }
.alert-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 2px; }
.alert-msg   { font-size: 0.88rem; color: #94a3b8; }
.alert-critical .alert-title { color: #fc8181; }
.alert-warning .alert-title  { color: #fbd38d; }
.alert-safe .alert-title     { color: #68d391; }

/* === SECTION HEADERS === */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 36px 0 20px 0;
}
.section-header-icon {
    width: 40px; height: 40px;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
}
.section-header-title { font-size: 1.25rem; font-weight: 700; color: #e2e8f0; }
.section-header-subtitle { font-size: 0.82rem; color: #64748b; margin-top: 1px; }

/* === CHART WRAPPER === */
.chart-wrapper {
    background: linear-gradient(135deg, rgba(30,41,59,0.6) 0%, rgba(15,23,42,0.7) 100%);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
}

/* === INCIDENT LOG === */
.incident-row {
    background: rgba(229,62,62,0.07);
    border: 1px solid rgba(229,62,62,0.2);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.incident-row .dot {
    width: 8px; height: 8px;
    background: #e53e3e;
    border-radius: 50%;
    flex-shrink: 0;
    box-shadow: 0 0 8px #e53e3e;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.4); }
}
.incident-frame { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #e53e3e; font-weight: 600; }
.incident-count { font-size: 0.85rem; color: #94a3b8; }

/* === ABOUT CARD === */
.about-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.7) 0%, rgba(15,23,42,0.8) 100%);
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 16px;
    padding: 28px;
}
.about-card h4 { color: #63b3ed; font-weight: 700; margin-bottom: 12px; }
.about-card p  { color: #94a3b8; font-size: 0.9rem; line-height: 1.7; }

/* === DATAFRAME OVERRIDES === */
[data-testid="stDataFrame"] {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(99,179,237,0.1) !important;
    border-radius: 12px !important;
}

/* === TAB STYLING === */
[data-testid="stTabs"] button {
    color: #64748b !important;
    font-weight: 500;
    font-size: 0.88rem;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #63b3ed !important;
    border-bottom: 2px solid #63b3ed !important;
}

/* === PROGRESS BAR === */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #63b3ed, #a78bfa) !important;
    border-radius: 10px !important;
}

/* === SIDEBAR SLIDERS === */
[data-testid="stSlider"] [role="slider"] {
    background-color: #63b3ed !important;
}

/* === FOOTER === */
.footer {
    text-align: center;
    padding: 40px 20px 20px;
    border-top: 1px solid rgba(99,179,237,0.08);
    margin-top: 60px;
}
.footer p { color: #475569; font-size: 0.82rem; }

/* === SCROLLBAR === */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1120; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* Hide default streamlit elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MATPLOTLIB DARK THEME HELPER
# ============================================================
def apply_dark_style(fig, ax_list=None):
    """Apply consistent dark theme to matplotlib figures."""
    fig.patch.set_facecolor('#0d1728')
    if ax_list is None:
        ax_list = fig.get_axes()
    for ax in ax_list:
        ax.set_facecolor('#0d1728')
        ax.tick_params(colors='#64748b', labelsize=9)
        ax.xaxis.label.set_color('#64748b')
        ax.yaxis.label.set_color('#64748b')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e293b')
        ax.grid(color='#1e293b', linestyle='-', linewidth=0.6, alpha=0.8)
    return fig


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
        <div style='font-size:3rem;'>🤖</div>
        <div style='font-size:1.1rem; font-weight:800; color:#e2e8f0; margin-top:8px;'>CrowdSentry AI</div>
        <div style='font-size:0.75rem; color:#64748b; margin-top:4px; letter-spacing:0.05em;'>CROWD MONITORING SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em; font-weight:600;'>⚙️ Alert Configuration</p>", unsafe_allow_html=True)
    alert_threshold = st.slider(
        "Crowd Alert Threshold",
        min_value=1, max_value=30, value=10,
        help="Alert fires when crowd count exceeds this value in any frame."
    )

    st.markdown("<p style='font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em; font-weight:600; margin-top:20px;'>📂 Data Source</p>", unsafe_allow_html=True)
    data_source = st.radio(
        "Input Mode",
        ["Use Project Data", "Upload Custom CSV"],
        label_visibility="collapsed"
    )

    uploaded_file = None
    if data_source == "Upload Custom CSV":
        uploaded_file = st.file_uploader(
            "Upload crowd_metrics.csv",
            type=["csv"],
            help="Must contain: Frame Number, Total Detected, Alert Triggered"
        )

    st.markdown("---")
    st.markdown("<p style='font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em; font-weight:600;'>📊 Display Options</p>", unsafe_allow_html=True)
    show_smoothed = st.toggle("Show Smoothed Trend", value=True)
    show_density_zones = st.toggle("Show Density Zones", value=True)
    chart_style = st.selectbox("Chart Style", ["Area Fill", "Line Only", "Bar Chart"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#475569; text-align:center; line-height:1.8;'>
        <strong style='color:#64748b;'>Model:</strong> YOLOv8s + Roboflow<br>
        <strong style='color:#64748b;'>Tracking:</strong> Head-Detection v1<br>
        <strong style='color:#64748b;'>Frame Skip:</strong> 1 of 3 frames<br>
        <strong style='color:#64748b;'>Alert Beep:</strong> macOS Ping.aiff
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(ttl=10)
def load_metrics(file_obj=None):
    if file_obj is not None:
        df = pd.read_csv(file_obj)
    else:
        # Try loading the project's actual CSV
        possible_paths = [
            "crowd_metrics.csv",
            "/Users/ramlasya/Documents/Crowd_monitoring_robot/crowd_robot/crowd_metrics.csv",
            "../crowd_metrics.csv"
        ]
        df = None
        for p in possible_paths:
            if os.path.exists(p):
                df = pd.read_csv(p)
                break
        if df is None:
            # fallback synthetic data shaped like real data
            frames = np.arange(3, 843, 3)
            n = len(frames)
            base = np.clip(np.random.normal(13, 4, n), 0, 25).astype(int)
            base[50:80] += 7
            inside = np.clip((base * 0.6).astype(int), 0, base)
            alert = inside >= 10
            df = pd.DataFrame({
                'Frame Number': frames,
                'Total Detected': base,
                'In Alert Area': inside,
                'Alert Triggered': alert
            })
    return df


df = load_metrics(uploaded_file)

# Recompute dynamic alert column based on sidebar slider
if 'Total Detected' not in df.columns:
    st.error("❌ CSV must contain a 'Total Detected' column.")
    st.stop()

df['Dynamic Alert'] = df['Total Detected'] >= alert_threshold

# Compute summary statistics
total_frames = len(df)
current_count = int(df['Total Detected'].iloc[-1]) if not df.empty else 0
max_count = int(df['Total Detected'].max())
avg_count = float(df['Total Detected'].mean())
alert_frames = int(df['Dynamic Alert'].sum())
safe_percent = round(100 * (1 - alert_frames / max(total_frames, 1)), 1)
peak_frame = int(df.loc[df['Total Detected'].idxmax(), 'Frame Number'])

# Check if 'In Alert Area' exists
has_alert_area = 'In Alert Area' in df.columns


# ============================================================
# HERO SECTION
# ============================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🤖 &nbsp; AI-Powered Surveillance System</div>
    <div class="hero-title">CrowdSentry AI<br>Monitoring Dashboard</div>
    <div class="hero-subtitle">
        Real-time crowd density analysis powered by <strong>YOLOv8</strong> object detection and the <strong>Roboflow</strong>
        Head Detection model. Monitors crowd behaviour, triggers alerts, and generates actionable analytics from video footage.
    </div>
    <div class="tech-stack">
        <span class="tech-pill">YOLOv8s</span>
        <span class="tech-pill">Roboflow API</span>
        <span class="tech-pill">Head Detection v1</span>
        <span class="tech-pill">OpenCV</span>
        <span class="tech-pill">DeepSORT Tracking</span>
        <span class="tech-pill">Python 3.10+</span>
        <span class="tech-pill">Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# KPI METRIC CARDS
# ============================================================
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card blue">
        <div class="metric-icon">👥</div>
        <div class="metric-label">Latest Crowd Count</div>
        <div class="metric-value">{current_count}</div>
        <div class="metric-delta neu">persons detected</div>
    </div>
    <div class="metric-card red">
        <div class="metric-icon">🔺</div>
        <div class="metric-label">Peak Crowd (Frame {peak_frame})</div>
        <div class="metric-value">{max_count}</div>
        <div class="metric-delta up">maximum recorded</div>
    </div>
    <div class="metric-card green">
        <div class="metric-icon">📊</div>
        <div class="metric-label">Average Density</div>
        <div class="metric-value">{avg_count:.1f}</div>
        <div class="metric-delta neu">persons / frame</div>
    </div>
    <div class="metric-card purple">
        <div class="metric-icon">🚨</div>
        <div class="metric-label">Security Breaches</div>
        <div class="metric-value">{alert_frames}</div>
        <div class="metric-delta {'up' if alert_frames > 0 else 'down'}">{safe_percent}% of time within limits</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# ALERT BANNER
# ============================================================
if current_count >= alert_threshold:
    st.markdown(f"""
    <div class="alert-critical">
        <div class="alert-icon">🚨</div>
        <div>
            <div class="alert-title">CRITICAL — Capacity Limit Exceeded!</div>
            <div class="alert-msg">{current_count} individuals detected in the last processed frame. Threshold: {alert_threshold} persons. Immediate action required.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
elif avg_count >= alert_threshold * 0.8:
    st.markdown(f"""
    <div class="alert-warning">
        <div class="alert-icon">⚠️</div>
        <div>
            <div class="alert-title">WARNING — Approaching Capacity Limit</div>
            <div class="alert-msg">Average density ({avg_count:.1f} persons/frame) is above 80% of the threshold ({alert_threshold}). Monitor closely.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="alert-safe">
        <div class="alert-icon">✅</div>
        <div>
            <div class="alert-title">ALL CLEAR — System Operating Normally</div>
            <div class="alert-msg">Crowd levels are within safe limits. Average: {avg_count:.1f} persons/frame. No immediate concerns detected.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# CAPACITY PROGRESS BAR
# ============================================================
capacity_pct = min(1.0, current_count / max(1, alert_threshold))
st.markdown(f"<p style='font-size:0.82rem; color:#64748b; margin-bottom:6px;'>🔒 Current Occupancy vs. Threshold  —  <strong style='color:#e2e8f0;'>{current_count} / {alert_threshold}</strong></p>", unsafe_allow_html=True)
st.progress(capacity_pct)

st.markdown("---")


# ============================================================
# MAIN TABS
# ============================================================
tab_analytics, tab_video, tab_incidents, tab_about = st.tabs([
    "📈 Analytics & Insights",
    "🎥 Video Evidence",
    "🚨 Incident Log",
    "ℹ️ About This System"
])


# ─────────────────────────────────────────────────────────
# TAB 1: ANALYTICS & INSIGHTS
# ─────────────────────────────────────────────────────────
with tab_analytics:

    col_main, col_side = st.columns([3, 1], gap="large")

    with col_main:
        # ── Chart 1: Crowd Density Timeline ──────────────────
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">📈</div>
            <div>
                <div class="section-header-title">Crowd Density Timeline</div>
                <div class="section-header-subtitle">Total persons detected per frame across entire video duration</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        plt.style.use('dark_background')
        fig1, ax1 = plt.subplots(figsize=(10, 4.5))
        fig1 = apply_dark_style(fig1, [ax1])

        x = df['Frame Number'].values
        y = df['Total Detected'].values

        if chart_style == "Area Fill":
            ax1.fill_between(x, y, alpha=0.18, color='#63b3ed')
            ax1.plot(x, y, color='#63b3ed', linewidth=2.2, label='Crowd Count', zorder=3)
        elif chart_style == "Line Only":
            ax1.plot(x, y, color='#63b3ed', linewidth=2.2, label='Crowd Count', zorder=3)
        else:
            ax1.bar(x, y, color='#3182ce', alpha=0.7, label='Crowd Count')

        if show_smoothed and len(y) > 10:
            from numpy.lib.stride_tricks import sliding_window_view
            win = min(15, len(y) // 5)
            if win >= 2:
                pad = win // 2
                y_smooth = np.convolve(y, np.ones(win)/win, mode='same')
                ax1.plot(x, y_smooth, color='#a78bfa', linewidth=2,
                         linestyle='--', alpha=0.9, label='Smoothed Trend', zorder=4)

        # Threshold line
        ax1.axhline(y=alert_threshold, color='#e53e3e', linestyle='--',
                    linewidth=1.8, label=f'Alert Threshold ({alert_threshold})', zorder=5)

        # Fill alert zones
        if show_density_zones:
            ax1.fill_between(x, y, alert_threshold,
                             where=(y >= alert_threshold),
                             interpolate=True, color='#e53e3e', alpha=0.22,
                             label='Breach Zone')

        ax1.set_xlabel('Frame Number', fontsize=10)
        ax1.set_ylabel('People Detected', fontsize=10)
        ax1.legend(loc='upper right', frameon=False, fontsize=9,
                   labelcolor='#94a3b8')
        ax1.set_xlim(x[0], x[-1])
        ax1.set_ylim(bottom=0)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

    with col_side:
        # ── Distribution Chart ────────────────────────────────
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">📊</div>
            <div>
                <div class="section-header-title">Distribution</div>
                <div class="section-header-subtitle">Crowd count frequency</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(4, 4.5))
        fig2 = apply_dark_style(fig2, [ax2])
        ax2.hist(df['Total Detected'], bins=20, color='#63b3ed', alpha=0.8,
                 edgecolor='#0d1728', linewidth=0.8)
        ax2.axvline(avg_count, color='#a78bfa', linestyle='--', linewidth=1.8, label=f'Mean: {avg_count:.1f}')
        ax2.axvline(alert_threshold, color='#e53e3e', linestyle='--', linewidth=1.8, label=f'Threshold: {alert_threshold}')
        ax2.set_xlabel('Count', fontsize=9)
        ax2.set_ylabel('Frequency', fontsize=9)
        ax2.legend(frameon=False, fontsize=8, labelcolor='#94a3b8')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # ── Status summary box ────────────────────────────────
        status_color = "#e53e3e" if alert_frames > total_frames * 0.25 else "#48bb78"
        st.markdown(f"""
        <div style='background:rgba(15,23,42,0.8); border:1px solid rgba(99,179,237,0.12);
                    border-radius:14px; padding:20px; margin-top:8px;'>
            <p style='font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:12px;'>Session Summary</p>
            <div style='margin-bottom:8px;'>
                <span style='color:#94a3b8; font-size:0.85rem;'>Total Frames</span>
                <span style='float:right; font-weight:700; color:#e2e8f0;'>{total_frames}</span>
            </div>
            <div style='margin-bottom:8px;'>
                <span style='color:#94a3b8; font-size:0.85rem;'>Alert Frames</span>
                <span style='float:right; font-weight:700; color:#fc8181;'>{alert_frames}</span>
            </div>
            <div style='margin-bottom:8px;'>
                <span style='color:#94a3b8; font-size:0.85rem;'>Safe Frames</span>
                <span style='float:right; font-weight:700; color:#68d391;'>{total_frames - alert_frames}</span>
            </div>
            <div style='margin-bottom:8px;'>
                <span style='color:#94a3b8; font-size:0.85rem;'>Peak Frame</span>
                <span style='float:right; font-weight:700; color:#63b3ed;'>#{peak_frame}</span>
            </div>
            <div>
                <span style='color:#94a3b8; font-size:0.85rem;'>Safety Rate</span>
                <span style='float:right; font-weight:700; color:{status_color};'>{safe_percent}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Row 2: Two charts side by side ───────────────────────
    st.markdown("---")
    r2col1, r2col2 = st.columns(2, gap="large")

    with r2col1:
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">🎯</div>
            <div>
                <div class="section-header-title">Alert Event Heatmap</div>
                <div class="section-header-subtitle">When and how intense alert events occurred</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        n_bins = 20
        frame_bins = np.array_split(df, n_bins)
        alert_counts = [g['Dynamic Alert'].sum() for g in frame_bins]
        avg_crowding  = [g['Total Detected'].mean() for g in frame_bins]
        bin_labels    = [f"F{int(g['Frame Number'].iloc[0])}" for g in frame_bins]

        heatmap_data = np.array([alert_counts]).reshape(4, 5)

        fig3, ax3 = plt.subplots(figsize=(6.5, 3.5))
        fig3 = apply_dark_style(fig3, [ax3])

        cmap = plt.cm.get_cmap('RdYlGn_r')
        im = ax3.imshow(heatmap_data, aspect='auto', cmap=cmap, interpolation='bilinear')

        ax3.set_xticks([])
        ax3.set_yticks([])
        fig3.colorbar(im, ax=ax3, label='Alert Events per Segment', shrink=0.8)

        for i in range(4):
            for j in range(5):
                text_val = heatmap_data[i, j]
                color = 'white' if text_val > heatmap_data.max() / 2 else '#e2e8f0'
                ax3.text(j, i, f'{text_val}', ha='center', va='center',
                         fontsize=12, fontweight='bold', color=color)

        ax3.set_title('Alert Events Across 20 Video Segments', color='#94a3b8', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with r2col2:
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">📉</div>
            <div>
                <div class="section-header-title">Crowd Roll-Average (10-frame)</div>
                <div class="section-header-subtitle">Rolling mean showing overall crowd dynamics</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig4, ax4 = plt.subplots(figsize=(6.5, 3.5))
        fig4 = apply_dark_style(fig4, [ax4])

        roll = df['Total Detected'].rolling(window=10, min_periods=1).mean()
        roll_std = df['Total Detected'].rolling(window=10, min_periods=1).std().fillna(0)

        ax4.plot(df['Frame Number'], roll, color='#a78bfa', linewidth=2.2, label='10-frame rolling avg')
        ax4.fill_between(df['Frame Number'],
                         roll - roll_std, roll + roll_std,
                         alpha=0.15, color='#a78bfa', label='±1 Std Dev')
        ax4.axhline(alert_threshold, color='#e53e3e', linestyle='--',
                    linewidth=1.5, label=f'Threshold ({alert_threshold})')
        ax4.set_xlabel('Frame Number', fontsize=9)
        ax4.set_ylabel('Persons (rolling avg)', fontsize=9)
        ax4.legend(frameon=False, fontsize=8, labelcolor='#94a3b8')
        ax4.set_ylim(bottom=0)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    # ── Row 3: Alert Area chart (only if data has it) ────────
    if has_alert_area:
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">🗺️</div>
            <div>
                <div class="section-header-title">Alert Zone vs Full Frame — Detection Split</div>
                <div class="section-header-subtitle">Comparison of heads detected in the monitored area vs total in frame</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig5, ax5 = plt.subplots(figsize=(12, 4))
        fig5 = apply_dark_style(fig5, [ax5])

        ax5.fill_between(df['Frame Number'], df['Total Detected'],
                         alpha=0.25, color='#63b3ed', label='Total Detected')
        ax5.plot(df['Frame Number'], df['Total Detected'],
                 color='#63b3ed', linewidth=1.8, zorder=3)

        ax5.fill_between(df['Frame Number'], df['In Alert Area'],
                         alpha=0.35, color='#fc8181', label='In Monitored Zone')
        ax5.plot(df['Frame Number'], df['In Alert Area'],
                 color='#fc8181', linewidth=1.8, zorder=4)

        ax5.axhline(alert_threshold, color='#fbd38d', linestyle='--',
                    linewidth=1.5, label=f'Alert Threshold ({alert_threshold})')
        ax5.set_xlabel('Frame Number', fontsize=10)
        ax5.set_ylabel('Persons Detected', fontsize=10)
        ax5.legend(frameon=False, fontsize=9, labelcolor='#94a3b8')
        ax5.set_xlim(df['Frame Number'].min(), df['Frame Number'].max())
        ax5.set_ylim(bottom=0)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

    # ── Raw data table ────────────────────────────────────────
    with st.expander("📋 View Raw Analytics Data", expanded=False):
        display_df = df.copy()
        display_df['Status'] = display_df['Dynamic Alert'].apply(
            lambda x: "🔴 ALERT" if x else "🟢 Safe"
        )
        st.dataframe(
            display_df[['Frame Number', 'Total Detected'] +
                       (['In Alert Area'] if has_alert_area else []) +
                       ['Status']].reset_index(drop=True),
            use_container_width=True,
            height=300
        )


# ─────────────────────────────────────────────────────────
# TAB 2: VIDEO EVIDENCE
# ─────────────────────────────────────────────────────────
with tab_video:
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">🎥</div>
        <div>
            <div class="section-header-title">Processed Video Output</div>
            <div class="section-header-subtitle">YOLOv8 detection with bounding boxes and crowd count overlay</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    vc1, vc2 = st.columns(2, gap="large")

    with vc1:
        st.markdown("<p style='font-size:0.78rem; color:#64748b; text-transform:uppercase; letter-spacing:0.08em; font-weight:600; margin-bottom:12px;'>🎬 AI Detection Output</p>", unsafe_allow_html=True)
        video_paths = [
            "/Users/ramlasya/Documents/Crowd_monitoring_robot/crowd_robot/output_video_h264.mp4",
            "/Users/ramlasya/Documents/Crowd_monitoring_robot/crowd_robot/output_video.mp4",
            "/Users/ramlasya/Documents/Crowd_monitoring_robot/crowd_robot/output_video_with_audio.mp4",
            "output_video_h264.mp4",
            "output_video.mp4",
            "output_video_with_audio.mp4"
        ]
        video_shown = False
        for vp in video_paths:
            if os.path.exists(vp):
                st.video(vp)
                video_shown = True
                break
        if not video_shown:
            st.info("📂 Video file not found. Run `process_video.py` to generate the processed output.")

    with vc2:
        st.markdown("<p style='font-size:0.78rem; color:#64748b; text-transform:uppercase; letter-spacing:0.08em; font-weight:600; margin-bottom:12px;'>🔥 Crowd Heatmap</p>", unsafe_allow_html=True)
        heatmap_paths = [
            "/Users/ramlasya/Documents/Crowd_monitoring_robot/crowd_robot/heatmap_result.png",
            "heatmap_result.png"
        ]
        heatmap_shown = False
        for hp in heatmap_paths:
            if os.path.exists(hp):
                st.image(hp, caption="Aggregated Movement Heatmap — warmer zones = higher density", use_container_width=True)
                heatmap_shown = True
                break
        if not heatmap_shown:
            st.info("🔥 Heatmap not available. Run `main.py` to generate movement tracking overlay.")

    # ── Security incident snapshots ───────────────────────────
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">📸</div>
        <div>
            <div class="section-header-title">Incident Frame Captures</div>
            <div class="section-header-subtitle">Auto-saved frames when crowd count exceeded the threshold</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    incident_dir = "/Users/ramlasya/Documents/Crowd_monitoring_robot/crowd_robot/security_incidents"
    if os.path.exists(incident_dir):
        incident_images = sorted([
            f for f in os.listdir(incident_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if incident_images:
            img_cols = st.columns(min(len(incident_images), 3))
            for idx, img_name in enumerate(incident_images[:3]):
                img_path = os.path.join(incident_dir, img_name)
                parts = img_name.replace('.jpg','').replace('.jpeg','').split('_')
                try:
                    frame_no = parts[2]
                    count_no = parts[4]
                    caption_text = f"Frame {frame_no} — {count_no} persons detected"
                except:
                    caption_text = img_name
                with img_cols[idx]:
                    st.image(img_path, caption=caption_text, use_container_width=True)
        else:
            st.info("📂 No incident frames saved yet.")
    else:
        st.info("📂 Security incidents directory not found.")


# ─────────────────────────────────────────────────────────
# TAB 3: INCIDENT LOG
# ─────────────────────────────────────────────────────────
with tab_incidents:
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">🚨</div>
        <div>
            <div class="section-header-title">Security Incident Log</div>
            <div class="section-header-subtitle">All frames where crowd count exceeded the configured threshold</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    incident_df = df[df['Dynamic Alert'] == True].copy()

    il1, il2, il3 = st.columns(3)
    with il1:
        st.markdown(f"""
        <div style='background:rgba(229,62,62,0.1); border:1px solid rgba(229,62,62,0.25);
                    border-radius:12px; padding:18px; text-align:center;'>
            <div style='font-size:2rem; font-weight:800; color:#fc8181;'>{len(incident_df)}</div>
            <div style='font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em; margin-top:4px;'>Total Incidents</div>
        </div>
        """, unsafe_allow_html=True)
    with il2:
        peak_in_alert = int(incident_df['Total Detected'].max()) if not incident_df.empty else 0
        st.markdown(f"""
        <div style='background:rgba(237,137,54,0.1); border:1px solid rgba(237,137,54,0.25);
                    border-radius:12px; padding:18px; text-align:center;'>
            <div style='font-size:2rem; font-weight:800; color:#fbd38d;'>{peak_in_alert}</div>
            <div style='font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em; margin-top:4px;'>Max Count in Alert</div>
        </div>
        """, unsafe_allow_html=True)
    with il3:
        breach_pct = round(100 * len(incident_df) / max(total_frames, 1), 1)
        st.markdown(f"""
        <div style='background:rgba(99,179,237,0.1); border:1px solid rgba(99,179,237,0.25);
                    border-radius:12px; padding:18px; text-align:center;'>
            <div style='font-size:2rem; font-weight:800; color:#63b3ed;'>{breach_pct}%</div>
            <div style='font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em; margin-top:4px;'>Breach Rate</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not incident_df.empty:
        # Scrollable incident timeline
        st.markdown("<p style='font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em; font-weight:600; margin-bottom:12px;'>🔴 Incident Timeline (latest first)</p>", unsafe_allow_html=True)
        for _, row in incident_df.tail(30).iloc[::-1].iterrows():
            count_ = int(row['Total Detected'])
            frame_ = int(row['Frame Number'])
            severity = "CRITICAL" if count_ >= alert_threshold * 1.5 else "HIGH"
            sev_color = "#e53e3e" if severity == "CRITICAL" else "#fc8181"
            st.markdown(f"""
            <div class="incident-row">
                <div class="dot"></div>
                <div style='flex:1;'>
                    <div class="incident-frame">Frame #{frame_:04d}</div>
                    <div class="incident-count">{count_} persons detected &nbsp;·&nbsp; {severity}</div>
                </div>
                <div style='background:rgba(229,62,62,0.15); border:1px solid rgba(229,62,62,0.3);
                            border-radius:8px; padding:5px 12px; font-size:0.8rem; font-weight:700; color:{sev_color};'>
                    +{count_ - alert_threshold} over limit
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Incident chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.78rem; color:#64748b; text-transform:uppercase; letter-spacing:0.08em; font-weight:600; margin-bottom:12px;'>📊 Breach Magnitude per Incident</p>", unsafe_allow_html=True)
        fig_inc, ax_inc = plt.subplots(figsize=(12, 3))
        fig_inc = apply_dark_style(fig_inc, [ax_inc])
        breach_magnitude = incident_df['Total Detected'] - alert_threshold
        ax_inc.bar(incident_df['Frame Number'], breach_magnitude, color='#e53e3e', alpha=0.8)
        ax_inc.set_xlabel('Frame Number', fontsize=9)
        ax_inc.set_ylabel('Persons Over Limit', fontsize=9)
        ax_inc.set_xlim(df['Frame Number'].min(), df['Frame Number'].max())
        ax_inc.set_ylim(bottom=0)
        plt.tight_layout()
        st.pyplot(fig_inc)
        plt.close()

    else:
        st.success("✅ No incidents recorded at current threshold setting. Try lowering the slider in the sidebar.")


# ─────────────────────────────────────────────────────────
# TAB 4: ABOUT THIS SYSTEM
# ─────────────────────────────────────────────────────────
with tab_about:

    ab1, ab2 = st.columns([2, 1], gap="large")

    with ab1:
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">🤖</div>
            <div>
                <div class="section-header-title">Project Overview</div>
                <div class="section-header-subtitle">Crowd Monitoring Robot — Final Year Project</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h4>🎯 Project Objective</h4>
            <p>
                This system is designed to provide intelligent, real-time crowd monitoring using
                computer vision and deep learning. The robot leverages the <strong style='color:#63b3ed;'>YOLOv8s</strong> object detection
                model in combination with the <strong style='color:#63b3ed;'>Roboflow Head Detection v1</strong> API to accurately
                count and track individuals in crowded environments.
            </p>
            <br>
            <h4>⚙️ System Pipeline</h4>
            <p>
                <strong style='color:#a78bfa;'>1. Input Stage:</strong> Video footage is fed from a camera or pre-recorded file.<br>
                <strong style='color:#a78bfa;'>2. Detection Stage:</strong> Every 3rd frame is sent to the Roboflow Head Detection API via HTTP POST (base64 encoded JPEG), which returns bounding box predictions with confidence scores.<br>
                <strong style='color:#a78bfa;'>3. Tracking Stage:</strong> DeepSORT multi-object tracker assigns stable IDs across frames to prevent double-counting.<br>
                <strong style='color:#a78bfa;'>4. Alert Stage:</strong> If the detected head count exceeds the configured threshold, an alert is logged, a visual overlay is drawn on the frame, and an audible beep is triggered.<br>
                <strong style='color:#a78bfa;'>5. Analytics Stage:</strong> Per-frame metrics (frame number, total detected, alert status) are streamed to a CSV file for live dashboard monitoring.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="about-card">
            <h4>📡 Detection Model Details</h4>
            <p>
                <strong style='color:#68d391;'>Model:</strong> head-detection-gun9q-mah4d/1 (Roboflow)<br>
                <strong style='color:#68d391;'>Base Architecture:</strong> YOLOv8s (small, fast)<br>
                <strong style='color:#68d391;'>Target Class:</strong> "head" (human heads in overhead/side view)<br>
                <strong style='color:#68d391;'>API:</strong> Roboflow Hosted Inference (REST HTTP)<br>
                <strong style='color:#68dြgreen;'>Confidence Threshold:</strong> API default (0.4+)<br>
                <strong style='color:#68d391;'>Frame Processing:</strong> 1-in-3 sampling for real-time performance<br>
                <strong style='color:#68d391;'>Output:</strong> MP4 video with bounding boxes + CSV analytics
            </p>
        </div>
        """, unsafe_allow_html=True)

    with ab2:
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">📋</div>
            <div>
                <div class="section-header-title">Tech Stack</div>
                <div class="section-header-subtitle"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tech_items = [
            ("🤖", "YOLOv8s", "Primary detection model", "#a78bfa"),
            ("📡", "Roboflow API", "Cloud inference endpoint", "#63b3ed"),
            ("👤", "DeepSORT", "Multi-object tracking", "#68d391"),
            ("🎥", "OpenCV", "Video processing engine", "#fbd38d"),
            ("🧮", "NumPy / Pandas", "Data processing", "#fc8181"),
            ("📊", "Matplotlib", "Visualization engine", "#f6ad55"),
            ("🌐", "Streamlit", "Web dashboard framework", "#63b3ed"),
            ("🐍", "Python 3.10+", "Core language", "#68d391"),
        ]

        for icon, name, desc, color in tech_items:
            st.markdown(f"""
            <div style='background:rgba(15,23,42,0.8); border:1px solid rgba(99,179,237,0.1);
                        border-radius:10px; padding:12px 16px; margin-bottom:8px; display:flex; align-items:center; gap:14px;'>
                <div style='font-size:1.3rem;'>{icon}</div>
                <div>
                    <div style='font-size:0.88rem; font-weight:700; color:{color};'>{name}</div>
                    <div style='font-size:0.75rem; color:#64748b;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Pie chart: frame breakdown
        safe_f  = total_frames - alert_frames
        alert_f = alert_frames
        fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
        fig_pie = apply_dark_style(fig_pie, [ax_pie])
        sizes = [safe_f, alert_f]
        colors_ = ['#48bb78', '#e53e3e']
        labels_ = [f'Safe ({safe_f})', f'Alert ({alert_f})']
        wedges, texts, autotexts = ax_pie.pie(
            sizes, colors=colors_, labels=labels_,
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(linewidth=2, edgecolor='#0d1728'),
            textprops=dict(color='#94a3b8', fontsize=9)
        )
        for at in autotexts:
            at.set_color('#e2e8f0')
            at.set_fontsize(9)
        ax_pie.set_title('Frame Safety Breakdown', color='#94a3b8', fontsize=9, pad=10)
        plt.tight_layout()
        st.pyplot(fig_pie)
        plt.close()


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    <p>🤖 <strong>CrowdSentry AI</strong> — Crowd Monitoring Robot | Final Year Project</p>
    <p style='margin-top:4px;'>Built with YOLOv8 · Roboflow · Streamlit · OpenCV &nbsp;|&nbsp; Head Detection Model v1</p>
</div>
""", unsafe_allow_html=True)

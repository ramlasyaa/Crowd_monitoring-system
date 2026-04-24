# CrowdSentry - AI-Powered Crowd Monitoring System

> Real-time crowd density analysis, anomaly detection, and incident reporting powered by Computer Vision and Deep Learning.

---

## Overview

CrowdSentry is an intelligent crowd monitoring system designed for public safety applications. It processes live CCTV footage to detect crowd density, identify abnormal behaviors (surges, falls, stampede patterns), generate heatmap visualizations, and serve a real-time dashboard for security personnel.

Built as part of a team project at Amrita Vishwa Vidyapeetham, and submitted for publication in a Springer journal (under review).

---

## Features

- Real-time density estimation - Counts and classifies crowd density levels per zone
- Heatmap visualization - Color-coded density overlays on CCTV frames
- Anomaly detection - Flags unusual motion patterns, surges, and high-risk areas
- Incident frame capture - Automatically saves flagged frames with timestamps
- Streamlit Dashboard - Live monitoring UI with alerts and video playback
- Incident log export - Downloadable CSV of detected events

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Computer Vision | OpenCV, YOLOv8 |
| Dashboard | Streamlit |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## Getting Started

```bash
git clone https://github.com/ramlasyaa/Crowd_monitoring-system.git
cd Crowd_monitoring-system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

---

## Publication

> "CrowdSentry: An AI-Based Real-Time Crowd Monitoring and Anomaly Detection System"
> Submitted to Springer - Under Review

---

*Connect via LinkedIn: https://www.linkedin.com/in/ram-lasya-405035336*

#!/bin/bash
# ============================================================
# CrowdSentry AI — App Launcher
# Just double-click this file or run: bash run_app.sh
# ============================================================

echo "🤖 Starting CrowdSentry AI Dashboard..."
echo "-------------------------------------------"
echo "Opening in your browser at: http://localhost:8501"
echo "Press Ctrl+C in this window to stop the app"
echo "-------------------------------------------"

cd "$(dirname "$0")/crowd_robot"
/opt/anaconda3/bin/streamlit run app.py --server.port 8501

import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import plotly.graph_objects as go
from utils import DrowsinessDetector
from collections import deque
try:
    import winsound
except ImportError:
    winsound = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

import threading
from fpdf import FPDF
import os
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="NEURAL-DRIVE AI", layout="wide", page_icon="🚗")

# --- Custom Styling (Techy/Dynamic) ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0e1117; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    .stMetric {
        background: rgba(17, 25, 40, 0.7);
        padding: 15px;
        border-radius: 10px;
        color: #00ffcc !important;
        border: 1px solid rgba(0,255,204,0.2);
    }
    h1, h2, h3 {
        color: #00ffcc !important;
        font-family: 'Roboto Mono', monospace;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .alert-banner {
        padding: 20px;
        border-radius: 10px;
        background-color: #ff4b4b;
        color: white;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        margin-bottom: 10px;
        box-shadow: 0 0 20px rgba(255, 75, 75, 0.5);
    }
    .glow-text {
        text-shadow: 0 0 10px #00ffcc, 0 0 20px #00ffcc;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialization ---
if 'detector' not in st.session_state:
    st.session_state.detector = DrowsinessDetector()
if 'running' not in st.session_state:
    st.session_state.running = False
if 'drowsy_event_count' not in st.session_state:
    st.session_state.drowsy_event_count = 0
if 'yawn_count' not in st.session_state:
    st.session_state.yawn_count = 0
if 'event_log' not in st.session_state:
    st.session_state.event_log = []
if 'history' not in st.session_state:
    st.session_state.history = {
        'ear': deque(maxlen=300),
        'mar': deque(maxlen=300),
        'bpm': deque(maxlen=300),
        'times': deque(maxlen=300)
    }

# --- Functions ---
def speak(text):
    if pyttsx3 is None:
        return
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    threading.Thread(target=_speak).start()

def log_event(event):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.event_log.append({"Timestamp": ts, "Event": event})

def generate_pdf_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="NEURAL-DRIVE: AI Fatigue & Hardware Analysis Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Session Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    
    # Performance Stats
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Driver Performance Metrics", ln=True)
    pdf.set_font("Arial", size=11)
    
    d_ev = st.session_state.drowsy_event_count
    y_ev = st.session_state.yawn_count
    score = max(0, 100 - (d_ev * 30 + y_ev * 20))
    avg_bpm = np.mean(list(st.session_state.history['bpm'])) if st.session_state.history['bpm'] else 0
    
    pdf.cell(200, 10, txt=f"Total Safety Score: {score}%", ln=True)
    pdf.cell(200, 10, txt=f"Prolonged Eye Closure Hits (1s): {d_ev}", ln=True)
    pdf.cell(200, 10, txt=f"Yawning Detection Hits: {y_ev}", ln=True)
    pdf.cell(200, 10, txt=f"Average Heart Rate (BPM): {int(avg_bpm)}", ln=True)
    pdf.ln(10)
    
    # Log Table
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Chronological Event Log", ln=True)
    pdf.set_font("Arial", size=9)
    for entry in st.session_state.event_log[-40:]: 
        pdf.cell(200, 7, txt=f"[{entry['Timestamp']}] {entry['Event']}", ln=True)
    
    # Graphs
    if len(st.session_state.history['ear']) > 5:
        plt.figure(figsize=(10, 8))
        plt.subplot(3, 1, 1); plt.plot(list(st.session_state.history['ear']), color='cyan'); plt.title("Ear Aspect (EAR)")
        plt.subplot(3, 1, 2); plt.plot(list(st.session_state.history['mar']), color='magenta'); plt.title("Mouth Aspect (MAR)")
        plt.subplot(3, 1, 3); plt.plot(list(st.session_state.history['bpm']), color='red'); plt.title("Heart Rate (BPM)")
        plt.tight_layout()
        plt.savefig("session_summary.png")
        plt.close()
        pdf.add_page()
        pdf.image("session_summary.png", x=10, y=20, w=190)
        os.remove("session_summary.png")

    return pdf.output()

# --- SIDEBAR ---
st.sidebar.title("🛠️ NEURAL CONTROL")
st.sidebar.markdown("---")
EAR_THRESH = st.sidebar.slider("EAR Limit", 0.15, 0.30, 0.20)
MAR_THRESH = st.sidebar.slider("MAR Limit", 0.30, 0.80, 0.42) # Optimized Sensitivity
st.sidebar.markdown("---")

if not st.session_state.running:
    if st.sidebar.button("🚀 IGNITION (START)", use_container_width=True):
        st.session_state.running = True
        st.session_state.drowsy_event_count = 0
        st.session_state.yawn_count = 0
        log_event("SESSION_STARTED")
        st.rerun()
else:
    if st.sidebar.button("🛑 STOP ENGINE", use_container_width=True):
        st.session_state.running = False
        log_event("SESSION_STOPPED")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.write(f"**Fatigue Alarms:** {st.session_state.drowsy_event_count}/2")
st.sidebar.write(f"**Yawn Alarms:** {st.session_state.yawn_count}/2")

# --- MAIN UI ---
st.markdown("<h1 class='glow-text'>NEURAL-DRIVE: AI Fatigue Monitor</h1>", unsafe_allow_html=True)

if not st.session_state.running:
    st.info("System Offline. Click IGNITION to begin real-time hardware & neural monitoring.")
    if st.session_state.event_log:
        st.subheader("📋 Final Session Analytics")
        c1, c2, c3 = st.columns(3)
        p_score = max(0, 100 - (st.session_state.drowsy_event_count * 30 + st.session_state.yawn_count * 20))
        c1.metric("Safety Score", f"{p_score}%")
        c2.metric("Total Hits", f"{st.session_state.drowsy_event_count + st.session_state.yawn_count}")
        av_bpm = int(np.mean(list(st.session_state.history['bpm']))) if st.session_state.history['bpm'] else 0
        c3.metric("Avg BPM", f"{av_bpm}")
        
        pdf_bytes = bytes(generate_pdf_report())
        st.download_button("📥 Download PDF Master Report", data=pdf_bytes, file_name="neural_drive_master_report.pdf", mime='application/pdf')
        st.dataframe(pd.DataFrame(st.session_state.event_log), use_container_width=True)
else:
    col_v, col_a = st.columns([2, 1])
    with col_v:
        st.subheader("📡 NEURAL FEED")
        frame_placeholder = st.empty()
        alert_placeholder = st.empty()
    with col_a:
        st.subheader("📊 REAL-TIME METRICS")
        m1, m2 = st.columns(2)
        ear_m = m1.empty()
        mar_m = m2.empty()
        bpm_m = st.empty()
        pose_m = st.empty()
        chart_m = st.empty()

    cap = cv2.VideoCapture(0)
    eyes_closed_frames = 0
    yawn_active = False
    tamper_frames = 0
    face_missing_frames = 0
    last_voice_t = 0
    
    fps = 30
    THRESHOLD_1S = 1 * fps
    FACE_MISSING_TIMEOUT = 2 * fps # 2-Second Timeout

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        intensity = np.mean(gray)
        
        # Hardware Tamper Detection
        if intensity < 8:
            tamper_frames += 1
            if tamper_frames > 25:
                now = time.time()
                if now - last_voice_t > 4:
                    speak("dont cover camera")
                    log_event("HARDWARE_ALERT: CAMERA_COVERED")
                    last_voice_t = now
                tamper_frames = 0
        else:
            tamper_frames = 0

        data, results = st.session_state.detector.process_frame(frame)
        
        if data["face_detected"]:
            face_missing_frames = 0
            # Eyes Monitoring (1s Sensitivity)
            if data["ear"] < EAR_THRESH:
                eyes_closed_frames += 1
            else:
                eyes_closed_frames = 0
            
            if eyes_closed_frames > THRESHOLD_1S:
                st.session_state.drowsy_event_count += 1
                log_event(f"DROWSY_HIT (Eyes closed 1s, Total: {st.session_state.drowsy_event_count})")
                if st.session_state.drowsy_event_count >= 2:
                    speak("Stop driving you are sleepy or drowsy")
                    log_event("CRITICAL_VOICE_ALERT: PROLONGED_FATIGUE")
                eyes_closed_frames = 0

            # Yawn Monitoring (Hyper-Sensitive)
            if data["mar"] > MAR_THRESH:
                if not yawn_active:
                    st.session_state.yawn_count += 1
                    yawn_active = True
                    log_event(f"YAWN_HIT (Total: {st.session_state.yawn_count})")
                    if st.session_state.yawn_count >= 2:
                        speak("Stop driving you are drowsy from yawning")
                        log_event("CRITICAL_VOICE_ALERT: REPEATED_YAWNING")
            else:
                yawn_active = False
            
            # Displays
            ear_m.metric("👁️ EAR", f"{data['ear']:.2f}")
            mar_m.metric("👄 MAR", f"{data['mar']:.2f}")
            bpm_m.metric("💓 BPM", f"{int(data['bpm'])}")
            pose_m.metric("📐 POSE", f"P: {int(data['pitch'])}° | Y: {int(data['yaw'])}°")
            
            # Active Alerts
            if eyes_closed_frames > 15:
                alert_placeholder.warning(f"EYE CLOSURE: {eyes_closed_frames/fps:.1f}s")
            elif st.session_state.drowsy_event_count >= 2 or st.session_state.yawn_count >= 2:
                 alert_placeholder.markdown("<div class='alert-banner'>🚨 CRITICAL ALERT: PULL OVER IMMEDIATELY! 🚨</div>", unsafe_allow_html=True)
            else:
                alert_placeholder.empty()

        else:
            face_missing_frames += 1
            if face_missing_frames > FACE_MISSING_TIMEOUT: # 2 Seconds
                alert_placeholder.error("🚨 WARNING: FACE NOT VISIBLE! 🚨")
                if face_missing_frames % 120 == 0:
                    log_event("FEED_ALERT: FACE_NOT_VISIBLE")
            
            if intensity < 8:
                alert_placeholder.error("🚨 HARDWARE ERROR: CAMERA OBSTRUCTED! 🚨")

        # History Tracking
        if data["face_detected"]:
            st.session_state.history['ear'].append(data["ear"])
            st.session_state.history['mar'].append(data["mar"])
            st.session_state.history['bpm'].append(data["bpm"])
            st.session_state.history['times'].append(time.strftime("%H:%M:%S"))

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        if len(st.session_state.history['times']) % 10 == 0 and data["face_detected"]:
            df_h = pd.DataFrame({
                'Time': list(st.session_state.history['times']), 
                'EAR': list(st.session_state.history['ear']), 'MAR': list(st.session_state.history['mar']), 'BPM': list(st.session_state.history['bpm'])
            })
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_h['Time'], y=df_h['EAR'], name='EAR', line=dict(color='#00ffcc')))
            fig.add_trace(go.Scatter(x=df_h['Time'], y=df_h['MAR'], name='MAR', line=dict(color='#ff00ff')))
            fig.add_trace(go.Scatter(x=df_h['Time'], y=df_h['BPM'], name='BPM', line=dict(color='#ff4b4b'), yaxis='y2'))
            fig.update_layout(template="plotly_dark", height=280, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              yaxis2=dict(overlaying='y', side='right', showgrid=False))
            chart_m.plotly_chart(fig, use_container_width=True)

        time.sleep(0.01)
    cap.release()

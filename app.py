"""
VitalCue - Streamlit Demo App
Contactless respiratory-rate monitoring + GenAI-guided stress intervention.
"""
import time
import uuid

import cv2
import streamlit as st
import pandas as pd
import altair as alt

from core.vision import VisionProcessor
from core.signal_processing import RespiratoryProcessor
from core.engine import BaselineEngine, VitalState
from core.genai_agent import RelaxationAgent
from core.database import DatabaseManager

st.set_page_config(page_title="VitalCue", layout="wide", initial_sidebar_state="collapsed")
st.title("VitalCue — Multi-Modal Stress Tracking")

if "vision" not in st.session_state:
    st.session_state.vision = VisionProcessor()
    st.session_state.resp = RespiratoryProcessor()
    st.session_state.engine = BaselineEngine()
    st.session_state.agent = RelaxationAgent()
    st.session_state.db = DatabaseManager()
    st.session_state.last_cue_time = 0
    st.session_state.cue_text = ""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_start = 0

# Top Bar
col_ctx, col_btn, col_timer = st.columns([2, 1, 1])
with col_ctx:
    context = st.selectbox("Scenario", ["workplace", "driving", "exam"])
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.toggle("Start Session", value=False)
with col_timer:
    st.markdown("<br>", unsafe_allow_html=True)
    timer_placeholder = st.empty()

if run and st.session_state.session_start == 0:
    st.session_state.session_start = time.time()
elif not run:
    st.session_state.session_start = 0

main_col1, main_col2 = st.columns([1, 2])

with main_col1:
    st.markdown("### Camera Feed")
    frame_placeholder = st.empty()

with main_col2:
    alignment_placeholder = st.empty()
    st.markdown("### Live Tracking")
    
    # 5 columns for metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    bpm_placeholder = col1.empty()
    state_placeholder = col2.empty()
    confidence_placeholder = col3.empty()
    expr_placeholder = col4.empty()
    gest_placeholder = col5.empty()
    
    st.markdown("### Respiratory Waveform")
    st.caption("Taking a deep breath (shoulders rising) should push the graph UP.")
    graph_placeholder = st.empty()
    
    st.markdown("### Agent Interventions")
    cue_placeholder = st.empty()
    progress_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    while run and cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            st.warning("Camera not available.")
            break

        # Timer
        elapsed = int(time.time() - st.session_state.session_start)
        mins, secs = divmod(elapsed, 60)
        timer_placeholder.markdown(f"**Session Time:** `{mins:02d}:{secs:02d}`")

        annotated, shoulder_y, _face_landmarks, alignment_feedback, expression, gesture = st.session_state.vision.process_frame(frame)
        
        if alignment_feedback != "ALIGNED":
            alignment_placeholder.warning(f"⚠️ **Positioning:** {alignment_feedback}")
        else:
            alignment_placeholder.empty()

        st.session_state.resp.add_sample(shoulder_y, time.time())
        bpm, confidence = st.session_state.resp.estimate_rate()

        state = st.session_state.engine.update(bpm, confidence) if bpm else st.session_state.engine.state

        # Log to Database once per second
        now = time.time()
        if not hasattr(st.session_state, 'last_db_log'):
            st.session_state.last_db_log = 0
        if now - st.session_state.last_db_log > 1.0:
            st.session_state.db.log_vital(
                session_id=st.session_state.session_id,
                bpm=bpm if bpm else 0.0,
                state=state.value,
                confidence=confidence,
                expression=expression,
                gesture=gesture
            )
            st.session_state.last_db_log = now

        frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
        bpm_placeholder.metric("Breaths/min", f"{bpm:.0f}" if bpm else "—")
        state_placeholder.metric("State", state.value)
        confidence_placeholder.metric("Confidence", f"{confidence:.0%}")
        expr_placeholder.metric("Expression", expression)
        gest_placeholder.metric("Gesture", gesture)
        
        waveform_data = st.session_state.resp.get_waveform()
        if waveform_data:
            df = pd.DataFrame({"y": waveform_data, "x": range(len(waveform_data))})
            chart = alt.Chart(df).mark_line(color="#00ff00", size=2).encode(
                x=alt.X("x", axis=None),
                y=alt.Y("y", axis=None, scale=alt.Scale(zero=False))
            ).properties(height=150)
            chart = chart.configure_view(strokeOpacity=0).configure_axis(grid=False)
            graph_placeholder.altair_chart(chart, use_container_width=True)

        if state == VitalState.CALIBRATING:
            if hasattr(st.session_state.engine, 'calibration_progress'):
                progress_placeholder.progress(st.session_state.engine.calibration_progress())
            cue_placeholder.info("Establishing your personal baseline — sit back so your shoulders are visible.")
        elif state == VitalState.STRESS:
            if now - st.session_state.last_cue_time > 12:  # don't re-cue every single frame
                repeat = st.session_state.last_cue_time > 0
                cue = st.session_state.agent.generate_cue(
                    bpm, st.session_state.engine.baseline_mean, repeat=repeat
                )
                st.session_state.cue_text = cue
                st.session_state.last_cue_time = now
                st.session_state.agent.speak(cue)
            cue_placeholder.warning(f"Stress pattern detected — {st.session_state.cue_text}")
            progress_placeholder.empty()
        elif state == VitalState.ESCALATE:
            msg = RelaxationAgent.escalation_message(context)
            if "last_escalate_msg" not in st.session_state or st.session_state.last_escalate_msg != msg:
                st.session_state.agent.speak(msg)
                st.session_state.last_escalate_msg = msg
            cue_placeholder.error(f"Escalation — {msg}")
            progress_placeholder.empty()
        else:
            cue_placeholder.success("Monitoring — signals within your normal range.")
            progress_placeholder.empty()

    cap.release()
    st.session_state.vision.close()
else:
    st.info("Click 'Start Session' to begin tracking and recording your vitals.")

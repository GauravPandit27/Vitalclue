"""
VitalCue - Streamlit Demo App
Contactless respiratory-rate monitoring + GenAI-guided breathing intervention.

The frame loop is the performance-critical part of this file. Streamlit redraws by
shipping each widget update over a websocket, so anything called per frame is paid for
at camera frame rate. Only the video feed genuinely needs that; metrics and charts are
throttled, and the pulse network runs on a worker thread inside HeartRateProcessor.
"""
import sys
import time
import uuid

import cv2
import streamlit as st
import pandas as pd
import altair as alt

from core.vision import VisionProcessor
from core.signal_processing import RespiratoryProcessor
from core.engine import (
    EFFORT_SUSTAIN_SECONDS,
    EFFORT_Z_THRESHOLD,
    STRESS_Z_THRESHOLD,
    BaselineEngine,
    VitalState,
)
from core.genai_agent import RelaxationAgent
from core.database import DatabaseManager
from core.heart_rate import HeartRateProcessor

# Capture at a fixed, modest resolution. MediaPipe downsamples internally anyway, so a
# 1080p feed costs decode and copy time for detail that is thrown away.
CAPTURE_WIDTH, CAPTURE_HEIGHT = 640, 480
DISPLAY_WIDTH = 400          # the frame is re-encoded every redraw; keep it small
UI_REFRESH_SECONDS = 0.2     # metrics and charts, not the video feed
DB_LOG_SECONDS = 1.0
CUE_INTERVAL_SECONDS = 12

st.set_page_config(page_title="VitalCue", layout="wide", initial_sidebar_state="collapsed")

# Streamlit's default vertical padding costs roughly 150px of screen, which is the
# difference between the whole dashboard fitting on a laptop display and not.
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 0rem;}
      [data-testid="stMetricValue"] {font-size: 1.6rem;}
      [data-testid="stMetricLabel"] {font-size: 0.75rem;}
      h3 {margin-bottom: 0.2rem; font-size: 1rem;}
      [data-testid="stCaptionContainer"] {font-size: 0.7rem; margin-top: -0.3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

head_title, col_ctx, col_mode, col_btn, col_timer = st.columns([3, 1.4, 1.4, 1, 1.2])
with head_title:
    st.markdown("#### VitalCue — Contactless Breathing & Guided Recovery")
with col_ctx:
    context = st.selectbox("Scenario", ["driving", "workplace"])
with col_mode:
    demo_mode = st.toggle("Demo mode", value=True, help="Shortens calibration for a live demo.")
    persist = st.toggle("Save session", value=False, help="Off by default — nothing is written to disk.")
with col_btn:
    show_hr = st.toggle("Heart rate", value=True,
                        help="PhysNet on facial video. Runs on a worker thread.")
    run = st.toggle("Start Session", value=False)
with col_timer:
    timer_placeholder = st.empty()

if "vision" not in st.session_state or st.session_state.get("demo_mode") != demo_mode:
    st.session_state.vision = VisionProcessor()
    st.session_state.resp = RespiratoryProcessor()
    st.session_state.engine = BaselineEngine(demo_mode=demo_mode)
    st.session_state.agent = RelaxationAgent()
    st.session_state.db = DatabaseManager(persist=persist)
    st.session_state.hr = HeartRateProcessor()
    st.session_state.demo_mode = demo_mode
    st.session_state.last_cue_time = 0
    st.session_state.cue_text = ""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_start = 0
    st.session_state.logged_recoveries = 0

if run and st.session_state.session_start == 0:
    st.session_state.session_start = time.time()
elif not run:
    st.session_state.session_start = 0

main_col1, main_col2 = st.columns([1, 2])

with main_col1:
    frame_placeholder = st.empty()
    alignment_placeholder = st.empty()

with main_col2:
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    bpm_placeholder = col1.empty()
    effort_placeholder = col2.empty()
    hr_placeholder = col3.empty()
    state_placeholder = col4.empty()
    confidence_placeholder = col5.empty()
    baseline_placeholder = col6.empty()

    wave_col, pulse_col = st.columns(2)
    with wave_col:
        st.markdown("### Respiratory waveform")
        st.caption("A deep breath (chest rising) should push the line UP.")
        graph_placeholder = st.empty()
    with pulse_col:
        st.markdown("### Pulse waveform (rPPG)")
        st.caption("PhysNet over 4.3s. Wellness signal only — sit still and well lit."
                   if show_hr else "Heart rate is off.")
        hr_graph_placeholder = st.empty()

    cue_placeholder = st.empty()
    progress_placeholder = st.empty()
    recovery_placeholder = st.empty()


def waveform_chart(values, colour, height):
    df = pd.DataFrame({"y": values, "x": range(len(values))})
    chart = alt.Chart(df).mark_line(color=colour, size=2).encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None, scale=alt.Scale(zero=False)),
    ).properties(height=height)
    return chart.configure_view(strokeOpacity=0).configure_axis(grid=False)


def open_camera():
    # DirectShow avoids the multi-second MSMF startup stall on Windows.
    cap = (cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == "win32"
           else cv2.VideoCapture(0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    # Without this the driver queues frames and the feed drifts seconds behind reality
    # whenever processing briefly falls behind capture.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


if run:
    cap = open_camera()
    last_ui_refresh = 0.0
    while run and cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            st.warning("Camera not available.")
            break

        annotated, shoulder_y, alignment_feedback = st.session_state.vision.process_frame(frame)

        st.session_state.resp.add_sample(shoulder_y, time.time())
        bpm, confidence = st.session_state.resp.estimate_rate()
        effort = st.session_state.resp.estimate_effort()

        heart_bpm, heart_conf = (None, 0.0)
        if show_hr:
            st.session_state.hr.add_frame(frame)
            heart_bpm, heart_conf = st.session_state.hr.estimate_rate()

        engine = st.session_state.engine
        state = engine.update(bpm, confidence, effort) if bpm else engine.state

        now = time.time()
        if now - st.session_state.get("last_db_log", 0) > DB_LOG_SECONDS:
            st.session_state.db.log_vital(
                session_id=st.session_state.session_id,
                bpm=bpm if bpm else 0.0,
                state=state.value,
                confidence=confidence,
            )
            st.session_state.last_db_log = now

        # Flush any newly completed stress episodes - this is the outcome metric.
        while st.session_state.logged_recoveries < len(engine.recovery_events):
            event = engine.recovery_events[st.session_state.logged_recoveries]
            st.session_state.db.log_recovery(
                session_id=st.session_state.session_id,
                duration_seconds=event["duration_seconds"],
                resolved=event["resolved"],
            )
            st.session_state.logged_recoveries += 1

        # The video feed redraws every frame; everything else would only flicker.
        frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                channels="RGB", width=DISPLAY_WIDTH)

        if now - last_ui_refresh < UI_REFRESH_SECONDS:
            continue
        last_ui_refresh = now

        elapsed = int(now - st.session_state.session_start)
        mins, secs = divmod(elapsed, 60)
        timer_placeholder.metric("Session", f"{mins:02d}:{secs:02d}")

        if alignment_feedback != "ALIGNED":
            alignment_placeholder.warning(f"**Positioning:** {alignment_feedback}")
        else:
            alignment_placeholder.empty()

        # Both triggers show their z-score, because "1.8x effort" means nothing without
        # knowing how far that is from this person's normal or from the threshold.
        bpm_placeholder.metric(
            "Breaths/min", f"{bpm:.0f}" if bpm else "—",
            delta=(f"z {engine.last_z:+.1f} / {STRESS_Z_THRESHOLD:.2f}"
                   if engine.baseline_mean else None),
            delta_color="off", help="Rapid shallow breathing is what moves this.")

        effort_ratio = engine.effort_ratio(effort)
        effort_placeholder.metric(
            "Effort", f"{effort_ratio:.1f}x" if effort_ratio else "—",
            delta=(f"z {engine.last_effort_z:+.1f} / {EFFORT_Z_THRESHOLD:.1f}"
                   if engine.last_effort_z is not None else None),
            delta_color="off",
            help="Shoulder excursion per breath, relative to your calibrated normal. "
                 f"Must stay above threshold for {EFFORT_SUSTAIN_SECONDS:.0f}s to trigger.")
        if not show_hr:
            hr_placeholder.metric("Heart rate", "off")
        elif heart_bpm:
            # Confidence is how much of the in-band spectrum sits at the peak. Shown
            # next to the number because a pulse reading without it invites more trust
            # than a 4.3s camera estimate has earned.
            hr_placeholder.metric("Heart rate", f"{heart_bpm:.0f} bpm",
                                  delta=f"{heart_conf:.0%} quality", delta_color="off",
                                  help="rPPG estimate. Not a medical measurement.")
        else:
            hr_placeholder.metric("Heart rate", "—",
                                  delta=f"{st.session_state.hr.progress():.0%} buffered",
                                  delta_color="off")
        state_placeholder.metric("State", state.value)
        confidence_placeholder.metric("Breath quality", f"{confidence:.0%}")
        baseline_placeholder.metric(
            "Your baseline",
            f"{engine.baseline_mean:.0f}" if engine.baseline_mean else "—",
        )

        waveform_data = st.session_state.resp.get_waveform()
        if waveform_data:
            graph_placeholder.altair_chart(
                waveform_chart(waveform_data, "#00ff00", 130), width="stretch")

        if show_hr:
            pulse = st.session_state.hr.get_waveform()
            if pulse:
                hr_graph_placeholder.altair_chart(
                    waveform_chart(pulse, "#ff4b6e", 130), width="stretch")

        if state == VitalState.CALIBRATING:
            progress_placeholder.progress(engine.calibration_progress())
            cue_placeholder.info("Learning your personal baseline — sit back so your chest is visible.")
        elif state == VitalState.STRESS:
            if now - st.session_state.last_cue_time > CUE_INTERVAL_SECONDS:
                repeat = st.session_state.last_cue_time > 0
                cue = st.session_state.agent.generate_cue(
                    bpm, engine.baseline_mean, repeat=repeat
                )
                st.session_state.cue_text = cue
                st.session_state.last_cue_time = now
                st.session_state.agent.speak(cue)
            # Name the signal that fired, so the intervention is explainable rather
            # than the system just asserting that you are stressed.
            by_effort = (engine.last_effort_z is not None
                         and engine.last_effort_z >= EFFORT_Z_THRESHOLD)
            trigger = "Breathing harder" if by_effort else "Breathing faster"
            cue_placeholder.warning(
                f"{trigger} than your baseline — {st.session_state.cue_text}")
            progress_placeholder.empty()
        elif state == VitalState.ESCALATE:
            msg = RelaxationAgent.escalation_message(context)
            if st.session_state.get("last_escalate_msg") != msg:
                st.session_state.agent.speak(msg)
                st.session_state.last_escalate_msg = msg
            cue_placeholder.error(f"Escalation — {msg}")
            progress_placeholder.empty()
        else:
            cue_placeholder.success("Monitoring — breathing within your normal range.")
            progress_placeholder.empty()

        summary = engine.recovery_summary()
        if summary:
            mean_recovery = summary["mean_recovery_seconds"]
            recovery_placeholder.caption(
                f"Intervention outcome: {summary['resolved']}/{summary['episodes']} episodes "
                f"returned to baseline"
                + (f", mean {mean_recovery:.0f}s" if mean_recovery else "")
            )

    cap.release()
    st.session_state.vision.close()
    st.session_state.hr.close()
else:
    st.info("Click 'Start Session' to begin.")

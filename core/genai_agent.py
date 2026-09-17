"""
VitalCue - GenAI Relaxation Agent
Generates short, spoken calming guidance for the STRESS state.
Never called for ESCALATE - escalation is a hard gate the LLM does not soften.
"""
try:
    import anthropic
except ImportError:
    anthropic = None

import threading
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

SYSTEM_PROMPT = """You are a calm, brief voice-guidance assistant for a real-time stress-support tool.
You will be given a structured state (never raw video or personal data) describing a detected
respiratory-rate spike relative to a person's own baseline. Respond with ONLY the spoken script,
under 10 seconds when read aloud (roughly 25 words max). Use simple, warm, unhurried language.
Guide a paced breathing exercise (e.g. "in for 4, hold for 4, out for 6") on the first cue.
If told this is a repeat cue (tension persisting), slow down and simplify further - do not repeat
the same script. Never mention medical diagnosis. Never use urgent or alarming language."""


class RelaxationAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        # Reads ANTHROPIC_API_KEY from the environment automatically; falls back to
        # a scripted cue below if no key is configured, so the app still runs offline.
        self.client = anthropic.Anthropic() if anthropic else None
        
        # TTS Engine Setup
        self.tts_engine = pyttsx3.init() if pyttsx3 else None
        if self.tts_engine:
            self.tts_engine.setProperty('rate', 150)
        self.is_speaking = False

    def speak(self, text: str):
        if not self.tts_engine or self.is_speaking:
            return
        def _speak():
            self.is_speaking = True
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            self.is_speaking = False
        threading.Thread(target=_speak).start()

    def generate_cue(self, bpm: float, baseline_bpm: float, repeat: bool = False) -> str:
        """Generate a short spoken calming cue. Only ever called for VitalState.STRESS."""
        if self.client is None or baseline_bpm is None:
            return self._fallback_cue(repeat)

        context = (
            f"Current respiratory rate: {bpm:.0f} breaths/min. "
            f"Personal baseline: {baseline_bpm:.0f} breaths/min. "
            f"This is a {'repeat' if repeat else 'first'} cue for this stress episode."
        )
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": context}],
            )
            return response.content[0].text.strip()
        except Exception:
            return self._fallback_cue(repeat)

    def _fallback_cue(self, repeat: bool) -> str:
        if repeat:
            return "Let's slow down even more. In for four. Hold. Out for six. You're doing fine."
        return "Notice your breath. In for four, hold for four, out for six. Let's do that together."

    @staticmethod
    def escalation_message(context: str) -> str:
        """
        Fixed, non-generated escalation text - deliberately not routed through the
        relaxation agent's generative path. Escalation copy should never vary or
        be softened by an LLM.
        """
        messages = {
            "driving": "Your signals suggest significant distress. Please consider pulling over safely.",
            "exam": "This has been flagged for the instructor. You may pause the exam if needed.",
            "workplace": "Take a break. This is just work, not your life. Rest now to reduce your stress, and you will be more productive later.",
        }
        return messages.get(context, "This has been flagged for follow-up.")

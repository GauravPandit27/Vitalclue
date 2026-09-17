import threading
import pyttsx3

class GenAIAgent:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.is_speaking = False
        
    def generate_intervention(self, hr, state):
        script = f"I notice your heart rate has elevated to {hr}. Let's take a deep breath together. Inhale deeply... and exhale slowly."
        return script
        
    def speak(self, text):
        if self.is_speaking:
            return
            
        def _speak():
            self.is_speaking = True
            self.engine.say(text)
            self.engine.runAndWait()
            self.is_speaking = False
            
        threading.Thread(target=_speak).start()

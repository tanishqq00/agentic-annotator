# session_service.py
import uuid
from time import time

class InMemorySessionService:
    def __init__(self):
        self.sessions = {}

    def create_session(self, user="default"):
        sid = str(uuid.uuid4())
        self.sessions[sid] = {"user": user, "start": time(), "events": []}
        return sid

    def add_event(self, sid, event):
        self.sessions[sid]["events"].append({"ts": time(), "event": event})

    def get_session(self, sid):
        return self.sessions.get(sid)

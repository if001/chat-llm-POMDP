from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo
from app.ports.clock import Clock


class SystemClock(Clock):
    def now_iso(self) -> str:
        return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()

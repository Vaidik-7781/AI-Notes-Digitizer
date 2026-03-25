"""
modules/session_manager.py
===========================
Lightweight JSON-based session persistence.

Sessions are stored as individual JSON files in the sessions directory.
Each file is named <job_id>.json and contains all page data, paths, settings.

Auto-cleanup: sessions older than Config.SESSION_TTL_HOURS are deleted
on the next list_recent() call.
"""

from __future__ import annotations

import os
import json
import time
import glob
from pathlib import Path
from typing import Optional

from config import Config


class SessionManager:

    def __init__(self, sessions_dir: str):
        self.sessions_dir = sessions_dir
        os.makedirs(sessions_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────

    def save(self, session_id: str, data: dict) -> bool:
        """
        Persist session data to disk.
        Strips word_boxes (too large) and ensures all values are JSON-serialisable.
        """
        path = self._path(session_id)
        slim_pages = []
        for page in data.get("pages", []):
            slim = {k: v for k, v in page.items() if k != "word_boxes"}
            slim_pages.append(slim)

        payload = {**data, "pages": slim_pages, "saved_at": time.time()}

        def _safe_default(obj):
            """Convert any non-serialisable object to string."""
            try:
                return str(obj)
            except Exception:
                return ""

        try:
            json_str = json.dumps(
                payload,
                ensure_ascii=True,   # ASCII-safe — avoids any encoding issues
                indent=2,
                default=_safe_default
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
            return True
        except Exception as e:
            print(f"[Session] Save failed for {session_id}: {e}")
            return False

    def load(self, session_id: str) -> Optional[dict]:
        """Load a session by ID. Returns None if not found."""
        path = self._path(session_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[Session] Load failed for {session_id}: {e} — deleting corrupt file")
            try:
                os.remove(path)
            except Exception:
                pass
            return None
        except Exception as e:
            print(f"[Session] Load failed for {session_id}: {e}")
            return None

    def delete(self, session_id: str) -> bool:
        """Delete a session file."""
        path = self._path(session_id)
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"[Session] Delete failed for {session_id}: {e}")
            return False

    def list_recent(self, limit: int = 10) -> list[dict]:
        """
        Return the most recent N sessions, sorted by creation time.
        Also prunes sessions older than SESSION_TTL_HOURS.
        """
        pattern = os.path.join(self.sessions_dir, "*.json")
        files   = glob.glob(pattern)
        now     = time.time()
        ttl_sec = Config.SESSION_TTL_HOURS * 3600
        records = []

        for fp in files:
            try:
                # Skip if file is too small to be valid JSON (< 10 bytes)
                if os.path.getsize(fp) < 10:
                    continue

                mtime = os.path.getmtime(fp)
                if now - mtime > ttl_sec:
                    os.remove(fp)
                    continue

                # Peek at first byte — if 0xFF or 0x89 it's a binary file, skip it
                with open(fp, "rb") as bf:
                    first_byte = bf.read(1)
                if first_byte and first_byte[0] in (0xFF, 0x89, 0x25, 0x50):
                    print(f"[Session] Skipping binary file in sessions dir: {fp}")
                    continue

                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
                records.append(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                print(f"[Session] Corrupt session file skipped: {fp}")
                continue
            except Exception:
                continue

        records.sort(key=lambda r: r.get("created_at", 0), reverse=True)
        return records[:limit]

    def exists(self, session_id: str) -> bool:
        return os.path.isfile(self._path(session_id))

    # ──────────────────────────────────────────────────────────────────────────

    def _path(self, session_id: str) -> str:
        # Sanitise ID to prevent directory traversal
        safe_id = "".join(c for c in session_id if c.isalnum() or c == "-")
        return os.path.join(self.sessions_dir, f"{safe_id}.json")
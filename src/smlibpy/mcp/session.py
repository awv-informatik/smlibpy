"""One modeling session: a worker process, the scripts that built its state, checkpoints.

The state of a session is the sequence of scripts that ran without an error. When the
worker has to be replaced (a script ran too long, the kernel crashed) or a checkpoint is
restored, a new worker runs that sequence again.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
from typing import Any

from . import worker

DEFAULT_TIMEOUT = 120.0
MAX_TIMEOUT = 900.0
REPLAY_TIMEOUT = 900.0
MAX_CHECKPOINTS = 20
SESSION_FORMAT = "smlibpy-session"
SESSION_FORMAT_VERSION = 1


class WorkerLost(Exception):
    pass


class Session:
    def __init__(self) -> None:
        self._mp = multiprocessing.get_context("spawn")
        self._lock = threading.RLock()
        self._proc: Any = None
        self._conn: Any = None
        self.history: list[tuple[str, str | None]] = []
        self.checkpoints: list[tuple[str, int]] = []
        self.version = 0  # grows whenever what the viewer shows may have changed
        self._changed = threading.Condition()

    # -- worker -------------------------------------------------------------------
    def _start(self) -> None:
        parent, child = self._mp.Pipe()
        proc = self._mp.Process(target=worker.main, args=(child,), daemon=True, name="smlibpy-session")
        proc.start()
        child.close()
        self._proc, self._conn = proc, parent

    def _stop(self) -> None:
        if self._proc is not None:
            if self._proc.is_alive():
                self._proc.kill()
            self._proc.join(5)
        if self._conn is not None:
            self._conn.close()
        self._proc = self._conn = None

    def _call(self, timeout: float, request: dict[str, Any]) -> dict[str, Any]:
        if self._proc is None or not self._proc.is_alive():
            self._stop()
            self._start()
        self._conn.send(request)
        if not self._conn.poll(timeout):
            self._stop()
            raise WorkerLost(f"it did not finish within {timeout:.0f} s")
        try:
            return self._conn.recv()
        except (EOFError, OSError):
            code = self._proc.exitcode if self._proc is not None else None
            self._stop()
            raise WorkerLost("the kernel process crashed" + (f" (exit code {code})" if code is not None else ""))

    def call(self, op: str, timeout: float = DEFAULT_TIMEOUT, **args: Any) -> dict[str, Any]:
        """A request that does not change the model (scene, snapshot, inspect, save)."""
        with self._lock:
            try:
                return self._call(timeout, {"op": op, **args})
            except WorkerLost as e:
                note = self._rebuild()
                return {"ok": False, "error": f"The request was stopped: {e}. {note}"}

    # -- scripts ------------------------------------------------------------------
    def run(self, code: str, label: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
        timeout = max(1.0, min(float(timeout), MAX_TIMEOUT))
        with self._lock:
            try:
                answer = self._call(timeout, {"op": "run", "code": code, "label": label})
            except WorkerLost as e:
                note = self._rebuild()
                return {"ok": False, "error": f"The script was stopped: {e}. {note}"}
            if answer.get("ok"):
                self.history.append((code, label))
                self._bump()
            return answer

    def load(self, path: str, name: str | None) -> dict[str, Any]:
        """Loading a file is part of the session's state: it is kept as a script."""
        with self._lock:
            try:
                answer = self._call(REPLAY_TIMEOUT, {"op": "load", "path": path, "name": name})
            except WorkerLost as e:
                note = self._rebuild()
                return {"ok": False, "error": f"Loading was stopped: {e}. {note}"}
            if answer.get("ok"):
                self.history.append((f"{answer['name']} = load({path!r})", f"load {path}"))
                self._bump()
            return answer

    def _rebuild(self, upto: int | None = None) -> str:
        """A new worker, with the scripts of the history run again."""
        self._stop()
        scripts = self.history[: len(self.history) if upto is None else upto]
        self.history = []
        self._start()
        for i, (code, label) in enumerate(scripts):
            try:
                answer = self._call(REPLAY_TIMEOUT, {"op": "run", "code": code, "label": label})
            except WorkerLost as e:
                self._stop()
                self._start()
                self.history = []
                self._replay_quietly(scripts[:i])
                self._bump()
                return (f"A new kernel process was started; {i} earlier scripts ran again, script {i + 1} "
                        f"did not ({e}), so the session holds what the first {i} built.")
            if not answer.get("ok"):
                self._bump()
                return (f"A new kernel process was started; {i} earlier scripts ran again, script {i + 1} failed "
                        f"this time: {answer.get('error', '')[:300]}. The session holds what the first {i} built.")
            self.history.append((code, label))
        self._bump()
        n = len(scripts)
        if n == 0:
            return "A new kernel process was started; the session is empty."
        return f"A new kernel process was started and the {n} earlier scripts ran again, so nothing was lost."

    def _replay_quietly(self, scripts: list[tuple[str, str | None]]) -> None:
        for code, label in scripts:
            try:
                answer = self._call(REPLAY_TIMEOUT, {"op": "run", "code": code, "label": label})
            except WorkerLost:
                self._start()
                return
            if not answer.get("ok"):
                return
            self.history.append((code, label))

    # -- checkpoints ----------------------------------------------------------------
    def checkpoint(self, label: str | None = None) -> str:
        with self._lock:
            label = label or f"checkpoint {len(self.checkpoints) + 1}"
            self.checkpoints = [(k, n) for k, n in self.checkpoints if k != label]
            self.checkpoints.append((label, len(self.history)))
            self.checkpoints = self.checkpoints[-MAX_CHECKPOINTS:]
            return label

    def restore(self, label: str | None = None) -> str:
        with self._lock:
            if not self.checkpoints:
                raise KeyError("no checkpoint yet")
            if label is None:
                label, n = self.checkpoints[-1]
            else:
                found = [n for k, n in self.checkpoints if k == label]
                if not found:
                    raise KeyError(f"no checkpoint {label!r}; there are: {', '.join(k for k, _ in self.checkpoints)}")
                n = found[-1]
            note = self._rebuild(upto=n)
            return f"Restored {label!r} ({n} scripts). {note}"

    def clear(self) -> None:
        with self._lock:
            self._stop()
            self.history = []
            self.checkpoints = []
            self._start()
            self._bump()

    # -- persistence ---------------------------------------------------------------
    def save_to_file(self, path: str) -> str:
        """Write the session's scripts and checkpoints to a JSON file.

        The file holds what a session is: the sequence of scripts that built its
        state, and the checkpoint labels with their positions in that sequence.
        ``load_from_file`` replays the scripts to restore the session.
        """
        with self._lock:
            data = {
                "format": SESSION_FORMAT,
                "format_version": SESSION_FORMAT_VERSION,
                "history": [{"code": code, "label": label} for code, label in self.history],
                "checkpoints": [{"label": label, "index": n} for label, n in self.checkpoints],
            }
            path = os.path.abspath(os.path.expanduser(path))
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1)
            return path

    def load_from_file(self, path: str) -> str:
        """Restore a session saved by ``save_to_file``: clear, then replay its scripts.

        Checkpoints are restored too, when their positions still hold (a checkpoint
        past a script that failed to replay is dropped).
        """
        path = os.path.abspath(os.path.expanduser(path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("format") != SESSION_FORMAT:
            raise ValueError(f"{path}: not a {SESSION_FORMAT} file")
        if data.get("format_version") != SESSION_FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported format version {data.get('format_version')!r}")
        with self._lock:
            self.clear()
            scripts = data.get("history", [])
            replayed = 0
            for i, entry in enumerate(scripts):
                code, label = entry["code"], entry.get("label")
                try:
                    answer = self._call(REPLAY_TIMEOUT, {"op": "run", "code": code, "label": label})
                except WorkerLost as e:
                    note = self._rebuild()
                    return (f"Restored {replayed} of {len(scripts)} scripts; the session is as it was "
                            f"before script {i + 1} ({e}). {note}")
                if not answer.get("ok"):
                    self._bump()
                    return (f"Restored {replayed} of {len(scripts)} scripts; script {i + 1} failed "
                            f"this time: {answer.get('error', '')[:200]}")
                self.history.append((code, label))
                replayed += 1
            kept = []
            for c in data.get("checkpoints", []):
                if isinstance(c.get("label"), str) and isinstance(c.get("index"), int) \
                        and 0 <= c["index"] <= replayed:
                    kept.append((c["label"], c["index"]))
            self.checkpoints = kept[-MAX_CHECKPOINTS:]
            self._bump()
            n_ckpt = len(self.checkpoints)
            return f"Restored {replayed} scripts and {n_ckpt} checkpoints."

    # -- change notification --------------------------------------------------------
    def _bump(self) -> None:
        with self._changed:
            self.version += 1
            self._changed.notify_all()

    def wait_for_change(self, seen: int, timeout: float) -> int:
        with self._changed:
            if self.version == seen:
                self._changed.wait(timeout)
            return self.version

    def close(self) -> None:
        with self._lock:
            self._stop()

"""Alert lifecycle: create / update / resolve / dismiss, without duplicates.

Rules (all PRELIMINARY indicators until validated against field observations):
- Actionable categories are watch / elevated / high.
- At most one ACTIVE alert per (AOI, kind); enforced by a unique partial index in the database.
- Processing an assessment whose ``as_of`` is not newer than the alert's last one is a no-op (idempotent re-runs).
- Insufficient data / baseline never creates, updates or resolves anything (unknown is not "OK").
- A clear (informational) assessment must repeat ``resolve_after_clear`` times before the alert is resolved.
- A dismissed alert suppresses re-creation at the same or lower severity for ``dismiss_suppress_days``;
  a genuine escalation above the dismissed severity creates a new alert.
- Every state change is recorded in ``alert_events``.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable

from src.alerts.risk_scoring import CATEGORIES, RiskAssessment

log = logging.getLogger(__name__)

ACTIONABLE = ("watch", "elevated", "high")
RANK = {c: i for i, c in enumerate(CATEGORIES)}          # informational=0 ... high=3
RESOLVE_AFTER_CLEAR = 2
DISMISS_SUPPRESS_DAYS = 30
KIND = "crop_stress"


@dataclass(frozen=True)
class AlertResult:
    action: str                  # created | updated | escalated | de-escalated | clear_pending | resolved | unchanged | suppressed | no_change
    alert_id: str | None
    category: str | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AlertManager:
    def __init__(self, conn: sqlite3.Connection, *, clock: Callable[[], str] = _now,
                 resolve_after_clear: int = RESOLVE_AFTER_CLEAR, dismiss_suppress_days: int = DISMISS_SUPPRESS_DAYS) -> None:
        self.conn = conn
        self.clock = clock
        self.resolve_after_clear = resolve_after_clear
        self.dismiss_suppress_days = dismiss_suppress_days

    # ---- helpers ------------------------------------------------------------------------------------
    def _event(self, alert_id: str, kind: str, frm: str | None, to: str | None, note: str | None = None) -> None:
        self.conn.execute("INSERT INTO alert_events (alert_id, event_type, at, from_category, to_category, note) "
                          "VALUES (?,?,?,?,?,?)", (alert_id, kind, self.clock(), frm, to, note))

    def get_active(self, aoi_id: str, kind: str = KIND) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM alerts WHERE aoi_id=? AND kind=? AND status='active'", (aoi_id, kind)).fetchone()

    def get(self, alert_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()

    def _last_dismissed(self, aoi_id: str, kind: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM alerts WHERE aoi_id=? AND kind=? AND status='dismissed' "
                                 "ORDER BY dismissed_at DESC LIMIT 1", (aoi_id, kind)).fetchone()

    # ---- main entry point ---------------------------------------------------------------------------
    def process(self, a: RiskAssessment, kind: str = KIND) -> AlertResult:
        """Apply one assessment to the alert state of its AOI."""
        if a.aoi_id is None:
            raise ValueError("assessment has no aoi_id")
        if a.category not in RANK:                                     # insufficient_data / insufficient_baseline
            return AlertResult("no_change", None, a.category)

        active = self.get_active(a.aoi_id, kind)
        payload = (json.dumps(a.evidence), json.dumps(a.to_dict()))
        now = self.clock()

        if a.category in ACTIONABLE:
            if active is None:
                prior = self._last_dismissed(a.aoi_id, kind)
                if prior is not None and self._suppressed(prior, a):
                    return AlertResult("suppressed", prior["id"], a.category)
                return self._create(a, kind, payload, now, after_dismissal=prior is not None)
            if a.as_of <= active["last_as_of"]:
                return AlertResult("unchanged", active["id"], active["category"])
            old, new = active["category"], a.category
            action = "escalated" if RANK[new] > RANK[old] else "de-escalated" if RANK[new] < RANK[old] else "updated"
            self.conn.execute(
                "UPDATE alerts SET category=?, score=?, confidence=?, confidence_label=?, last_seen=?, last_as_of=?, updated_at=?, "
                "clear_streak=0, evidence_json=?, assessment_json=? WHERE id=?",
                (new, a.score, a.confidence, a.confidence_label, a.as_of, a.as_of, now, *payload, active["id"]))
            self._event(active["id"], action, old, new, f"score {a.score}")
            self.conn.commit()
            return AlertResult(action, active["id"], new)

        # informational: counts toward resolution of an open alert
        if active is None:
            return AlertResult("no_change", None, a.category)
        if a.as_of <= active["last_as_of"]:
            return AlertResult("unchanged", active["id"], active["category"])
        streak = active["clear_streak"] + 1
        if streak >= self.resolve_after_clear:
            self.conn.execute("UPDATE alerts SET status='resolved', resolved_at=?, updated_at=?, last_as_of=?, clear_streak=? WHERE id=?",
                              (now, now, a.as_of, streak, active["id"]))
            self._event(active["id"], "resolved", active["category"], a.category, f"{streak} consecutive clear observations")
            self.conn.commit()
            return AlertResult("resolved", active["id"], a.category)
        self.conn.execute("UPDATE alerts SET clear_streak=?, last_as_of=?, updated_at=? WHERE id=?",
                          (streak, a.as_of, now, active["id"]))
        self._event(active["id"], "updated", active["category"], active["category"],
                    f"clear observation {streak}/{self.resolve_after_clear}")
        self.conn.commit()
        return AlertResult("clear_pending", active["id"], active["category"])

    def _suppressed(self, prior: sqlite3.Row, a: RiskAssessment) -> bool:
        age = (date.fromisoformat(a.as_of) - date.fromisoformat(prior["dismissed_at"][:10])).days
        return age <= self.dismiss_suppress_days and RANK[a.category] <= RANK[prior["dismissed_category"]]

    def _create(self, a: RiskAssessment, kind: str, payload: tuple[str, str], now: str, after_dismissal: bool) -> AlertResult:
        alert_id = "alert_" + hashlib.sha1(f"{a.aoi_id}|{kind}|{a.as_of}|{now}".encode()).hexdigest()[:12]
        self.conn.execute(
            "INSERT INTO alerts (id, aoi_id, kind, status, category, score, confidence, confidence_label, first_seen, last_seen, "
            "last_as_of, created_at, updated_at, clear_streak, evidence_json, assessment_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,?)",
            (alert_id, a.aoi_id, kind, "active", a.category, a.score, a.confidence, a.confidence_label, a.as_of, a.as_of,
             a.as_of, now, now, *payload))
        self._event(alert_id, "created", None, a.category,
                    "re-created after escalation above a dismissed alert" if after_dismissal else f"score {a.score}")
        self.conn.commit()
        return AlertResult("created", alert_id, a.category)

    # ---- user actions ---------------------------------------------------------------------------------
    def dismiss(self, alert_id: str, note: str | None = None) -> None:
        alert = self._require_active(alert_id)
        now = self.clock()
        self.conn.execute("UPDATE alerts SET status='dismissed', dismissed_at=?, dismissed_category=?, updated_at=? WHERE id=?",
                          (now, alert["category"], now, alert_id))
        self._event(alert_id, "dismissed", alert["category"], None, note)
        self.conn.commit()

    def resolve(self, alert_id: str, note: str | None = None) -> None:
        alert = self._require_active(alert_id)
        now = self.clock()
        self.conn.execute("UPDATE alerts SET status='resolved', resolved_at=?, updated_at=? WHERE id=?", (now, now, alert_id))
        self._event(alert_id, "resolved", alert["category"], None, note or "resolved manually")
        self.conn.commit()

    def _require_active(self, alert_id: str) -> sqlite3.Row:
        alert = self.get(alert_id)
        if alert is None:
            raise KeyError(f"No such alert: {alert_id}")
        if alert["status"] != "active":
            raise ValueError(f"Alert {alert_id} is already {alert['status']}")
        return alert

    # ---- queries and export ---------------------------------------------------------------------------
    def list(self, *, status: str | None = None, aoi_id: str | None = None, category: str | None = None,
             since: str | None = None) -> list[sqlite3.Row]:
        q, args = "SELECT a.*, i.name AS aoi_name FROM alerts a JOIN aois i ON i.id = a.aoi_id WHERE 1=1", []
        for col, val in (("a.status", status), ("a.aoi_id", aoi_id), ("a.category", category)):
            if val:
                q += f" AND {col} = ?"
                args.append(val)
        if since:
            q += " AND a.last_seen >= ?"
            args.append(since)
        return self.conn.execute(q + " ORDER BY a.last_seen DESC, a.created_at DESC", args).fetchall()

    def events(self, alert_id: str) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM alert_events WHERE alert_id=? ORDER BY id", (alert_id,)).fetchall()

    def export_csv(self, **filters) -> str:
        rows = self.list(**filters)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["alert_id", "aoi_id", "aoi_name", "status", "category", "score", "confidence", "confidence_label",
                    "first_seen", "last_seen", "evidence", "preliminary_indicator"])
        for r in rows:
            w.writerow([r["id"], r["aoi_id"], r["aoi_name"], r["status"], r["category"], r["score"], r["confidence"],
                        r["confidence_label"], r["first_seen"], r["last_seen"], " | ".join(json.loads(r["evidence_json"])), "yes"])
        return buf.getvalue()

    def export_geojson(self, **filters) -> dict:
        features = []
        for r in self.list(**filters):
            geom = json.loads(self.conn.execute("SELECT geometry_geojson FROM aois WHERE id=?", (r["aoi_id"],)).fetchone()[0])
            features.append({"type": "Feature", "id": r["id"], "geometry": geom, "properties": {
                "aoi_id": r["aoi_id"], "aoi_name": r["aoi_name"], "status": r["status"], "category": r["category"],
                "score": r["score"], "confidence": r["confidence"], "first_seen": r["first_seen"], "last_seen": r["last_seen"],
                "evidence": json.loads(r["evidence_json"]), "preliminary_indicator": True}})
        return {"type": "FeatureCollection", "features": features}

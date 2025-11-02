from flask import Blueprint, jsonify
from storage import list_entries
from services.analysis import compute_analysis_for_text
from datetime import datetime, timedelta

bp = Blueprint('dashboard', __name__)


def _parse_iso_datetime(ts: str):
    if not ts:
        return None
    try:
        if ts.endswith('Z'):
            ts2 = ts[:-1]
        else:
            ts2 = ts
        return datetime.fromisoformat(ts2)
    except Exception:
        try:
            return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")
        except Exception:
            return None


@bp.route('/api/dashboard', methods=['GET'])
def get_dashboard():
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_sums = {i: 0.0 for i in range(7)}
    weekday_counts = {i: 0 for i in range(7)}
    freq = {}
    total_insights = 0
    recent = []

    entries = list_entries()
    for e in entries.values():
        created_at = e.get('createdAt')
        dt = _parse_iso_datetime(created_at)
        score = None
        if 'score' in e:
            try:
                score = float(e.get('score'))
            except Exception:
                score = None
        if dt:
            w = dt.weekday()
            if score is not None:
                weekday_sums[w] += score
                weekday_counts[w] += 1
        analysis = e.get('analysis')
        if analysis and isinstance(analysis, dict):
            dlist = analysis.get('distortions', [])
            for d in dlist:
                name = d.get('type') or 'Unknown'
                freq[name] = freq.get(name, 0) + 1
                total_insights += 1
        recent.append((e.get('createdAt'), e.get('id'), e.get('summary') or (e.get('text')[:120] if e.get('text') else '')))

    weeklyProgress = []
    for i, label in enumerate(labels):
        cnt = weekday_counts.get(i, 0)
        avg = int(round(weekday_sums.get(i, 0) / cnt)) if cnt > 0 else 0
        weeklyProgress.append({"day": label, "score": avg})

    distortionFrequency = sorted([{"name": k, "count": v} for k, v in freq.items()], key=lambda x: x['count'], reverse=True)

    num_entries = len(entries)
    score_vals = [float(e.get('score')) for e in entries.values() if e.get('score') is not None]
    current_score = int(round(sum(score_vals) / len(score_vals))) if score_vals else 0

    stats = [
        {"label": "Journal Entries", "value": str(num_entries), "trend": ""},
        {"label": "Current Score", "value": f"{current_score}%", "trend": ""},
        {"label": "Insights Gained", "value": str(total_insights), "trend": ""},
        {"label": "Weekly Streak", "value": "0 days", "trend": ""},
    ]

    try:
        today = datetime.utcnow().date()
        days_with_entry = set()
        for e in entries.values():
            dt = _parse_iso_datetime(e.get('createdAt'))
            if dt:
                days_with_entry.add(dt.date())
        streak = 0
        cur = today
        while cur in days_with_entry:
            streak += 1
            cur = cur - timedelta(days=1)
        stats[3]['value'] = f"{streak} days"
    except Exception:
        pass

    recent_sorted = sorted([r for r in recent if r[0]], key=lambda x: x[0], reverse=True)[:5]
    recentNotes = []
    for createdAt, id_, excerpt in recent_sorted:
        recentNotes.append({"id": id_, "excerpt": excerpt[:120], "createdAt": createdAt})

    return jsonify({
        "weeklyProgress": weeklyProgress,
        "distortionFrequency": distortionFrequency,
        "stats": stats,
        "recentNotes": recentNotes
    })

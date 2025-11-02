from flask import Blueprint, request, jsonify
import uuid
from datetime import datetime
from services.analysis import summarize_text, compute_analysis_for_text
from storage import create_entry_record, get_entry, list_entries, set_entry_analysis
from threading import Thread

bp = Blueprint('entries', __name__)


@bp.route('/api/entries', methods=['POST'])
def create_entry():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    if not payload or 'text' not in payload or not isinstance(payload['text'], str):
        return jsonify({"error": "Request must be JSON with a 'text' string field"}), 400

    text = payload['text'].strip()
    mood = payload.get('mood')
    meta = payload.get('meta') if isinstance(payload.get('meta'), dict) else None

    entry_id = uuid.uuid4().hex
    created_at = datetime.utcnow().isoformat() + 'Z'
    summary = summarize_text(text)

    entry = {
        "id": entry_id,
        "text": text,
        "mood": mood,
        "meta": meta,
        "createdAt": created_at,
        "summary": summary,
    }

    create_entry_record(entry)

    # optionally kick off background analysis
    def _bg(eid, t):
        analysis = compute_analysis_for_text(t)
        set_entry_analysis(eid, analysis)

    Thread(target=_bg, args=(entry_id, text), daemon=True).start()

    response = {"id": entry_id, "text": text, "createdAt": created_at}
    if summary:
        response['summary'] = summary
    return jsonify(response), 201


@bp.route('/api/entries', methods=['GET'])
def list_entries_route():
    entries = list_entries()
    items = []
    for e in entries.values():
        item = {"id": e.get('id'), "text": e.get('text'), "createdAt": e.get('createdAt')}
        if 'score' in e:
            item['score'] = e.get('score')
        if 'summary' in e:
            item['summary'] = e.get('summary')
        items.append(item)
    items.sort(key=lambda x: x.get('createdAt', ''), reverse=True)
    return jsonify(items)


@bp.route('/api/analysis/<entry_id>', methods=['GET'])
def get_analysis_for_entry(entry_id: str):
    entry = get_entry(entry_id)
    if not entry:
        return jsonify({"error": "Entry not found"}), 404
    if 'analysis' in entry:
        return jsonify(entry['analysis'])

    # start background analysis and return pending
    Thread(target=lambda: set_entry_analysis(entry_id, compute_analysis_for_text(entry.get('text', ''))), daemon=True).start()
    return jsonify({"status": "pending"}), 202

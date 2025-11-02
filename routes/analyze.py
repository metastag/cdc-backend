from flask import Blueprint, request, jsonify
from services.analysis import compute_analysis_for_text

bp = Blueprint('analyze', __name__)


@bp.route('/api/analyze', methods=['POST'])
def analyze_text_route():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    if not payload or 'text' not in payload or not isinstance(payload['text'], str):
        return jsonify({"error": "Request must be JSON with a 'text' string field"}), 400

    text = payload['text'].strip()
    if text == "":
        return jsonify({"distortions": [], "overallScore": 100, "positivePatterns": []})

    analysis = compute_analysis_for_text(text)
    return jsonify(analysis)

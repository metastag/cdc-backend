import re
from typing import List, Dict, Any


def match_distortions(text: str) -> List[Dict[str, Any]]:
    text_lower = text.lower()
    matches: List[Dict[str, Any]] = []

    patterns = [
        (
            "All-or-Nothing Thinking",
            re.compile(r"\b(always|never|everyone|no one)\b", re.IGNORECASE),
            "high",
            "This statement generalizes events in absolute terms (always/never), which is a black-and-white cognitive distortion.",
            "There are times when things go well and times when they don't; try focusing on specific instances instead of absolutes."
        ),
        (
            "Overgeneralization",
            re.compile(r"\b(i always|i never|i can't|i cant|i will always)\b", re.IGNORECASE),
            "high",
            "A broad conclusion is being drawn from a single or few events.",
            "One or a few setbacks don't define all future outcomes."
        ),
        (
            "Catastrophizing",
            re.compile(r"\b(disaster|ruin|impossible|can't handle|can't cope|can't cope)\b", re.IGNORECASE),
            "medium",
            "This predicts the worst possible outcome without solid evidence.",
            "Consider more likely, less extreme outcomes and coping strategies."
        ),
        (
            "Labeling",
            re.compile(r"\b(i am worthless|i'm worthless|i am a failure|i'm a failure|i am stupid|i'm stupid)\b", re.IGNORECASE),
            "high",
            "Using global negative labels about the self rather than describing a specific behavior.",
            "Describe the behavior or situation rather than labeling yourself; focus on what can be changed."
        ),
        (
            "Should Statements",
            re.compile(r"\bshould\b", re.IGNORECASE),
            "low",
            "Using 'should' can create unrealistic expectations and guilt.",
            "Replace 'should' with preferences or concrete, achievable actions."
        ),
        (
            "Mind Reading",
            re.compile(r"\b(they think|they must think|everyone thinks|people think)\b", re.IGNORECASE),
            "medium",
            "Assuming you know what others are thinking without evidence.",
            "Check assumptions — ask, or look for concrete evidence before concluding others' thoughts."
        ),
    ]

    for dtype, patt, severity, explanation, reframe in patterns:
        m = patt.search(text)
        if m:
            excerpt = m.group(0)
            confidence = min(95, 60 + len(excerpt) * 3)
            matches.append({
                "type": dtype,
                "severity": severity,
                "confidence": confidence,
                "excerpt": excerpt.strip(),
                "explanation": explanation,
                "reframe": reframe,
            })

    negatives = ["i always fail", "i fail at everything", "i can't do anything"]
    for neg in negatives:
        if neg in text_lower:
            matches.append({
                "type": "All-or-Nothing Thinking",
                "severity": "high",
                "confidence": 90,
                "excerpt": neg,
                "explanation": "This uses absolute language to generalize failure across situations.",
                "reframe": "I've had setbacks but also successes; I can learn from specific experiences."
            })

    # dedupe
    seen = set()
    dedup = []
    for m in matches:
        key = (m['type'], m['excerpt'])
        if key not in seen:
            seen.add(key)
            dedup.append(m)
    return dedup


def detect_positive_patterns(text: str) -> List[str]:
    text_lower = text.lower()
    positives = []
    if re.search(r"\bi feel\b|\bi'm feeling\b|\bi am feeling\b", text_lower):
        positives.append("Shows self-awareness about feelings")
    if re.search(r"\b(i want to|i'm learning|i am learning|i will try|i will work)\b", text_lower):
        positives.append("Indicates willingness to change or learn")
    if re.search(r"\b(thankful|grateful|appreciate)\b", text_lower):
        positives.append("Expresses gratitude or positive appraisal")
    return positives


def summarize_text(text: str, max_len: int = 240) -> str:
    text = text.strip()
    if not text:
        return ""
    for sep in ('.', '!', '?'):
        parts = text.split(sep)
        if parts and parts[0].strip():
            sent = parts[0].strip()
            if len(sent) <= max_len:
                return sent
    return (text[:max_len] + '...') if len(text) > max_len else text


def compute_analysis_for_text(text: str) -> Dict[str, Any]:
    distortions = match_distortions(text)
    positive_patterns = detect_positive_patterns(text)
    severity_map = {"low": 10, "medium": 20, "high": 30}
    penalty = 0.0
    for d in distortions:
        sev_val = severity_map.get(d.get('severity', 'low'), 10)
        conf = float(d.get('confidence', 50)) / 100.0
        penalty += sev_val * conf
    overall = max(0, int(round(100 - penalty)))
    return {"distortions": distortions, "overallScore": overall, "positivePatterns": positive_patterns}

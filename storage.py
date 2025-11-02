import os
import json
from threading import Lock
from typing import Dict, Any, Optional

DATA_FILE = os.path.join(os.path.dirname(__file__), 'entries.json')
_entries_lock = Lock()
ENTRIES: Dict[str, Dict[str, Any]] = {}


def _load_entries() -> None:
    global ENTRIES
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    ENTRIES = data
        except Exception:
            ENTRIES = {}


def _save_entries() -> None:
    try:
        with _entries_lock:
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(ENTRIES, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_entry(entry_id: str) -> Optional[Dict[str, Any]]:
    return ENTRIES.get(entry_id)


def create_entry_record(entry: Dict[str, Any]) -> None:
    with _entries_lock:
        ENTRIES[entry['id']] = entry
    _save_entries()


def list_entries() -> Dict[str, Dict[str, Any]]:
    return ENTRIES


def set_entry_analysis(entry_id: str, analysis: Dict[str, Any]) -> None:
    with _entries_lock:
        e = ENTRIES.get(entry_id)
        if not e:
            return
        e['analysis'] = analysis
        e['score'] = analysis.get('overallScore')
        ENTRIES[entry_id] = e
    _save_entries()


# load on import
_load_entries()

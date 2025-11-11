from flask import Flask, jsonify
from flask_cors import CORS
import os

# Warm model on startup
from model import get_analyzer

# register blueprints
from routes.analyze import bp as analyze_bp
from routes.entries import bp as entries_bp
from routes.dashboard import bp as dashboard_bp

app = Flask(__name__)
# Allow CORS from the frontend running on localhost:8080
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})
app.register_blueprint(analyze_bp)
app.register_blueprint(entries_bp)
app.register_blueprint(dashboard_bp)

# Load the ML analyzer at process startup so the model is downloaded/loaded
# once per process instead of on first user request. This may take time on
# the very first run while the model downloads; subsequent restarts will
# load from the local cache.
try:
    if os.environ.get("WARM_MODEL_ON_STARTUP", "1") != "0":
        print("Warming ML analyzer at startup (this may take a while on first run)...")
        get_analyzer()
        print("ML analyzer warmed and ready.")
except Exception as _e:
    # Don't crash the app if model warmup fails; requests will still work
    # with the rule-based fallback.
    print(f"Warning: ML analyzer warmup failed: {_e}")


@app.route('/')
def index():
    return jsonify({"message": "Hello from Flask", "status": "ok"})



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
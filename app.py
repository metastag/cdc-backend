from flask import Flask, jsonify

# register blueprints
from routes.analyze import bp as analyze_bp
from routes.entries import bp as entries_bp
from routes.dashboard import bp as dashboard_bp

app = Flask(__name__)
app.register_blueprint(analyze_bp)
app.register_blueprint(entries_bp)
app.register_blueprint(dashboard_bp)


@app.route('/')
def index():
    return jsonify({"message": "Hello from Flask", "status": "ok"})



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

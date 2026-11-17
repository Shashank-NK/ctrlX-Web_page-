from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from agent import analyze

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="")


@app.get("/")
def home():
    return send_from_directory(FRONTEND, "index.html")


@app.post("/api/analyze")
def api_analyze():
    f = request.files.get("image")
    if f is None:
        return jsonify({"error": "No image uploaded"}), 400
    try:
        return jsonify(analyze(f.read()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
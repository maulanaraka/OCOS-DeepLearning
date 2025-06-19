from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rag_engine import query_with_rag
import os

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("main.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"answer": "Pertanyaan kosong."}), 400
    answer = query_with_rag(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
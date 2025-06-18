from flask import Flask, request, render_template_string, jsonify
from flask_cors import CORS
from google import genai

app = Flask(__name__)
CORS(app)

gemini_client = genai.Client(api_key="AIzaSyBGfprRY9PW9PED_C_XcqPAZwHatxfMkbo")

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Ask Gemini AI</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        #answer { margin-top: 20px; padding: 10px; background: #f0f0f0; border-radius: 5px; }
    </style>
</head>
<body>
    <h2>Ask Gemini AI</h2>
    <form id="ask-form">
        <input type="text" id="question" name="question" placeholder="Type your question..." required style="width:300px;">
        <button type="submit">Ask</button>
    </form>
    <div id="answer"></div>
    <script>
        document.getElementById('ask-form').onsubmit = async function(e) {
            e.preventDefault();
            const question = document.getElementById('question').value;
            document.getElementById('answer').innerText = 'Loading...';
            const res = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });
            const data = await res.json();
            document.getElementById('answer').innerText = data.answer;
        };
    </script>
</body>
</html>
'''

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"answer": "No question provided."})
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=question,
        )
        answer = response.text
    except Exception as e:
        answer = f"Error: {str(e)}"
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)

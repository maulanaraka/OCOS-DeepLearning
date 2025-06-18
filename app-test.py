from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
import os

app = Flask(__name__)
CORS(app)

# --- Konfigurasi Kunci API ---
# Catatan Keamanan: Sangat disarankan untuk tidak menulis kunci API langsung di dalam kode.
# Gunakan variabel lingkungan untuk keamanan.
# Contoh: impoort os; api_key = os.getenv("GEMINI_API_KEY")
# Pastikan untuk mengatur variabel lingkungan GEMINI_API_KEY di sistem Anda.
try:
    api_key = "AIzaSyBGfprRY9PW9PED_C_XcqPAZwHatxfMkbo" # GANTI DENGAN KUNCI API ANDA
    if not api_key:
        raise ValueError("Kunci API Gemini tidak ditemukan. Harap atur dalam kode atau sebagai variabel lingkungan.")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error saat menginisialisasi klien Gemini: {e}")
    gemini_model = None

@app.route("/")
def index():
    """Menyajikan antarmuka pengguna utama dari file main.html."""
    return render_template('main.html')

@app.route("/ask", methods=["POST"])
def ask():
    """Menerima pertanyaan, mengirimkannya ke Gemini AI, dan mengembalikan jawabannya."""
    if not gemini_model:
        return jsonify({"answer": "Error: Klien Gemini AI tidak terinisialisasi. Periksa kunci API Anda."}), 500

    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"answer": "Tidak ada pertanyaan yang diberikan."}), 400

    try:
        # Menghasilkan konten menggunakan model Gemini
        response = gemini_model.generate_content(question)
        answer = response.text
    except Exception as e:
        # Menangani kemungkinan kesalahan dari API
        answer = f"Terjadi kesalahan saat menghubungi Gemini API: {str(e)}"
        return jsonify({"answer": answer}), 500
        
    return jsonify({"answer": answer})

if __name__ == "__main__":
    # Menjalankan aplikasi Flask dalam mode debug di port 5000
    app.run(debug=True, port=5000)
import google.generativeai as genai
import os

# Setup Gemini
genai.configure(api_key="AIzaSyBGfprRY9PW9PED_C_XcqPAZwHatxfMkbo")
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

def generate_answer(prompt: str):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[ERROR Gemini] {str(e)}"
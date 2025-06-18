
from google import genai

client = genai.Client(api_key="AIzaSyBGfprRY9PW9PED_C_XcqPAZwHatxfMkbo")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Lu kenal damar ga?",
)

print(response.text)
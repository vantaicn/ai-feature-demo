from google import genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Mở file audio (wav, mp3, m4a...)
with open("google_tts_output_vi.wav", "rb") as f:
    audio_file = f.read()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        {
            "role": "user",
            "parts": [
                {
                    "text": "Chuyển đoạn audio sau thành văn bản. Nhận xét cách phát âm, ngữ điệu, tốc độ nói của người trong audio so với người bản xứ."
                },
                {
                    "inline_data": {
                        "mime_type": "audio/wav",  # hoặc "audio/mp3", "audio/m4a"
                        "data": audio_file,
                    }
                },
            ],
        }
    ],
)

print(response.text)

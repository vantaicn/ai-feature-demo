from google import genai
from google.genai import types
import wave
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Set up the wave file to save the output:
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-tts",
    contents="My name is Nguyen Van Tai. I am a software developer. I love programming. I also love music. I love to travel and explore new places. I enjoy reading books and watching movies in my free time. I am passionate about learning new technologies and improving my skills. I believe in continuous growth and self-improvement. I am excited about the future and the opportunities it holds.",
    config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Kore",
                )
            )
        ),
    ),
)

data = response.candidates[0].content.parts[0].inline_data.data

file_name = "gemini-tts-output.wav"
wave_file(file_name, data)  # Saves the file to current directory

from google.cloud import speech
from dotenv import load_dotenv

load_dotenv()


def transcribe_audio(path):
    client = speech.SpeechClient()

    with open(path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=24000,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        print("Result:", result)
        print("Transcript:", result.alternatives[0].transcript)


if __name__ == "__main__":
    transcribe_audio("google_tts_output_vi.wav")

"""
  Result: alternatives {
    transcript: "My name is nwin vanti. I am a software developer. I love programming. I also love music. I love to travel and explore new places. I enjoy reading books and watching movies in my free time, I am passionate about learning, new technologies and improving my skills. I believe in continuous growth and self-improvement. I am excited about the future and the opportunities it holds."
    confidence: 0.969163179
  }
  result_end_time {
    seconds: 22
    nanos: 500000000
  }
  language_code: "en-us"
"""

"""
Result: alternatives {
  transcript: "Me name is Wyn Vonnie. I am a software developer. I love programming. I also love music. I love to travel and explore new places. I enjoy reading book and watching movie in me. Free time. I am passionate about learning, new technology and improving my skill. You believe in continuous growth and self-improvement. I am excited about the future and the opportunity it help."
  confidence: 0.740091443
}
result_end_time {
  seconds: 26
  nanos: 730000000
}
language_code: "en-us"
"""

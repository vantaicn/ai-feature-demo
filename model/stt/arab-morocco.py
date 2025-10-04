from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="smerchi/Arabic-Morocco-Speech_To_Text")
result = pipe("google_tts_output_vi.wav")
print("Result:", result)
print("Transcription:", result["text"])
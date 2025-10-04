from kittentts import KittenTTS
import soundfile as sf

m = KittenTTS("KittenML/kitten-tts-nano-0.2")

audio = m.generate(
  "My name is Nguyen Le Anh Thu. I am a software developer. I love programming. I also love music. I love to travel and explore new places. I enjoy reading books and watching movies in my free time. I am passionate about learning new technologies and improving my skills. I believe in continuous growth and self-improvement. I am excited about the future and the opportunities it holds.",
  speed=1.5,
  voice='expr-voice-3-f'
)

sf.write("output.wav", audio, 24000)

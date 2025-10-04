from google.cloud import texttospeech


def text_to_speech(text, output_file, key_path):
    # Khởi tạo client từ API key JSON
    client = texttospeech.TextToSpeechClient.from_service_account_file(key_path)

    # Nội dung text tiếng Anh
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Chọn giọng đọc tiếng Anh
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",  # English (US)
        name="en-US-Wavenet-A",  # giọng nam tự nhiên
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )

    # Cấu hình định dạng file audio
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16  # WAV
    )

    # Gọi API sinh giọng nói
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Ghi ra file
    with open(output_file, "wb") as out:
        out.write(response.audio_content)
        print(f"✅ Audio content written to file: {output_file}")


if __name__ == "__main__":
    text = """
      Uhm, hello. Today I want to talk about the importance of learning a second language. I think learning a new language is very, very important for everyone for some reasons.
      First, it is good for your job. You can have… uhm… more chance to get a good job. Many company now… they need people who can speak English or other language. So, if you know a second language, you can… you can find a job easy and earn more money. It is very useful.
      Second, learning a new language help you when you travel. For example, when I go to another country, if I know their language, I can talk to local people. I can understand more about… uhm… their culture. It make the trip more interesting and you don't get lost.
      And… uhm… final, it is good for your brain. I read on the internet that when you learn new language, your memory is better. You can think faster. It also open your mind to new ideas and new ways of thinking.
      So, because of these reasons, I believe everyone should try to learn a second language. It is very good. Thank you.
    """

    key_path = "C:\\Users\\vanta\\Downloads\\quickclipaicreator-0b3f1d6a5b48.json"
    output_file = "vstep_speaking_part3.wav"

    text_to_speech(text, output_file, key_path)

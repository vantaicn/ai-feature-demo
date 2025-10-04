import os
import json
from dotenv import load_dotenv

from google import genai
from google.cloud import texttospeech

load_dotenv()


def generate_listening_content(topic: str, num_blanks: int, target_level: str) -> dict:
    """
    Sử dụng Gemini để tạo nội dung cho bài tập nghe.
    """
    try:
        client = genai.Client()

        prompt_text = f"""
            Bạn là một chuyên gia tạo học liệu tiếng Anh, chuyên thiết kế các bài tập nghe cho người học Việt Nam.

            **Bối cảnh:**
            - Trình độ của người học: "{target_level}"
            - Chủ đề bài nghe: "{topic}"
            - Số lượng từ cần điền vào chỗ trống: {num_blanks}

            **Yêu cầu:**
            1.  Viết một đoạn văn ngắn (khoảng 3-5 câu) về chủ đề "{topic}", phù hợp trình độ "{target_level}".
            2.  Từ đoạn văn đó, chọn ra chính xác {num_blanks} từ vựng quan trọng để làm đáp án.
            3.  Tạo ra phiên bản "đề bài" của đoạn văn bằng cách thay thế các từ đã chọn bằng '___'.
            4.  Cung cấp toàn bộ kết quả dưới dạng một đối tượng JSON duy nhất. **Không thêm bất kỳ văn bản giải thích nào bên ngoài đối tượng JSON này.**

            **Cấu trúc JSON đầu ra bắt buộc:**
            {{
              "topic": "{topic}",
              "level": "{target_level}",
              "full_text": "<Đoạn văn gốc đầy đủ để tạo audio>",
              "exercise_text": "<Đoạn văn đã được đục lỗ>",
              "answers": ["<từ thứ nhất>", "<từ thứ hai>"]
            }}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{"role": "user", "parts": [{"text": prompt_text}]}],
        )

        if not response.text:
            return {"error": "Gemini không trả về kết quả."}

        cleaned_response = (
            response.text.strip().replace("```json", "").replace("```", "").strip()
        )
        return json.loads(cleaned_response)

    except Exception as e:
        print(f"Lỗi Gemini: {e}")
        return {"error": "Không thể tạo nội dung bài nghe."}


def create_audio_from_text(text: str, output_file: str) -> str | None:
    """
    Sử dụng Google Cloud Text-to-Speech để tạo file audio.
    """
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", name="en-US-Wavenet-A"
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        with open(output_file, "wb") as out:
            out.write(response.audio_content)
        print(f"✅ Audio content written to file: {output_file}")
        return output_file
    except Exception as e:
        print(f"Lỗi Google TTS: {e}")
        return None


if __name__ == "__main__":
    TOPIC = "A weekend picnic"
    NUM_BLANKS = 4
    LEVEL = "FLYERS"

    OUTPUT_AUDIO_FILE = "listening_exercise.wav"

    print(f"Đang tạo nội dung bài nghe về '{TOPIC}'...")
    content_data = generate_listening_content(
        topic=TOPIC, num_blanks=NUM_BLANKS, target_level=LEVEL
    )

    if "error" in content_data:
        print(f"Không thể tạo nội dung: {content_data['error']}")
    else:
        print("✅ Nội dung đề bài đã được tạo thành công.")
        print("-" * 20)

        text_for_audio = content_data.get("full_text")
        if text_for_audio:
            print(f"Đang tạo file audio '{OUTPUT_AUDIO_FILE}'...")
            create_audio_from_text(text_for_audio, OUTPUT_AUDIO_FILE)
            print("-" * 20)

        print("--- ĐỀ BÀI HOÀN CHỈNH ---")
        print(json.dumps(content_data, indent=2, ensure_ascii=False))


""" 
*Input:
  - topic: str - Chủ đề bài nghe (ví dụ: "A weekend picnic")
  - num_blanks: int - Số lượng từ cần điền vào chỗ trống
  - target_level: str - Trình độ của người học (ví dụ: "STARTERS", "MOVERS", "FLYERS")
*Output:
  - dict - Đối tượng JSON với cấu trúc:
            {
              "topic": "<chủ đề bài nghe>",
              "level": "<trình độ người học>",
              "full_text": "<Đoạn văn gốc đầy đủ để tạo audio>",
              "exercise_text": "<Đoạn văn đã được đục lỗ>",
              "answers": ["<từ thứ nhất>", "<từ thứ hai>"]
            }
"""

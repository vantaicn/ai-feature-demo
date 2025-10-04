import os
import json
from dotenv import load_dotenv

from google.cloud import speech
from google import genai

load_dotenv()
CONFIDENCE_THRESHOLD = 0.9


def transcribe_with_sst(audio_path: str) -> tuple[str | None, float]:
    try:
        client = speech.SpeechClient()
        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000,
            language_code="en-US",
            enable_automatic_punctuation=True,
            enable_word_confidence=True,
        )

        response = client.recognize(config=config, audio=audio)

        if not response or not response.results:
            return None, 0.0

        result = response.results[0].alternatives[0]
        return result.transcript, result.confidence

    except Exception as e:
        print(f"Lỗi STT: {e}")
        return None, 0.0


def analyze_with_gemini(audio_path: str, target_word: str, user_level: str) -> dict:
    try:
        client = genai.Client()
        with open(audio_path, "rb") as f:
            audio_file_data = f.read()

        prompt_text = f"""
            Bạn là một chuyên gia huấn luyện phát âm tiếng Anh giọng Mỹ (American English) cho người Việt. Nhiệm vụ của bạn là lắng nghe đoạn âm thanh do người học cung cấp và đưa ra nhận xét chi tiết, hữu ích.

            **Bối cảnh:**
            - Từ cần phát âm: "{target_word}"
            - Trình độ của người học: "{user_level}"

            **Yêu cầu:**
            1.  Nghe kỹ file âm thanh đính kèm.
            2.  So sánh cách phát âm của người học với cách phát âm chuẩn của từ "{target_word}".
            3.  Cung cấp phản hồi dưới dạng một đối tượng JSON. **Không thêm bất kỳ giải thích nào bên ngoài đối tượng JSON này.**

            **Cấu trúc JSON đầu ra bắt buộc:**
            {{
              "overall_score": <một số nguyên từ 0 đến 100>,
              "transcribed_text": "<văn bản bạn nghe được từ audio>",
              "positive_feedback": "<Một câu nhận xét về những điểm làm tốt>",
              "points_to_improve": [
                {{
                  "phoneme": "<âm vị hoặc trọng âm bị lỗi>",
                  "error_description": "<mô tả lỗi sai một cách đơn giản>",
                  "suggestion": "<hướng dẫn sửa lỗi cụ thể>"
                }}
              ]
            }}
        """

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt_text},
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": audio_file_data,
                            }
                        },
                    ],
                }
            ],
        )

        if not response.text:
            return {"error": "Gemini không trả về kết quả."}

        cleaned_response = (
            response.text.strip().replace("```json", "").replace("```", "").strip()
        )
        return json.loads(cleaned_response)

    except Exception as e:
        print(f"Lỗi Gemini: {e}")
        return {"error": "Không thể phân tích phát âm bằng Gemini."}


def get_pronunciation_feedback(audio_path: str, target_word: str, user_level: str):
    transcript, confidence = transcribe_with_sst(audio_path)

    if transcript is None:
        return {"feedback_type": "error", "message": "Không thể nhận dạng giọng nói."}

    normalized_transcript = transcript.lower().strip().replace(".", "")
    normalized_target_word = target_word.lower().strip()

    if normalized_transcript != normalized_target_word:
        return {
            "feedback_type": "simple_mistake",
            "message": f"Phát âm chưa đúng. Hệ thống nghe được '{transcript}'.",
        }

    if confidence >= CONFIDENCE_THRESHOLD:
        return {"feedback_type": "simple_correct", "message": "Phát âm rất tốt!"}

    print(
        f"Confidence thấp ({confidence}), chuyển sang phân tích chi tiết với Gemini..."
    )

    return {
        "feedback_type": "detailed_analysis",
        "data": analyze_with_gemini(audio_path, target_word, user_level),
    }


if __name__ == "__main__":
    AUDIO_FILE_PATH = "pronunciation.wav"
    TARGET_WORD = "pronunciation"
    USER_LEVEL = "VSTEP B1"

    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"Lỗi: Không tìm thấy file {AUDIO_FILE_PATH}")
    else:
        final_feedback = get_pronunciation_feedback(
            AUDIO_FILE_PATH, TARGET_WORD, USER_LEVEL
        )
        print(json.dumps(final_feedback, indent=2, ensure_ascii=False))


""" 
* Input: 
    - audio_path: đường dẫn file âm thanh người học phát âm từ "pronunciation"
    - target_word: "pronunciation"
    - user_level: trình độ người học, ví dụ "VSTEP B1"
* Output: 
    - Nếu phát âm đúng và confidence cao: {"feedback_type": "simple_correct", "message": "Phát âm rất tốt!"}
    - Nếu phát âm sai: {"feedback_type": "simple_mistake", "message": "Phát âm chưa đúng. Hệ thống nghe được '...'"}
    - Nếu phát âm đúng nhưng confidence thấp: {"feedback_type": "detailed_analysis", "data": {...kết quả phân tích chi tiết từ Gemini...}}
* Prompt Gemini:
    Bạn là một chuyên gia huấn luyện phát âm tiếng Anh giọng Mỹ (American English) cho người Việt. Nhiệm vụ của bạn là lắng nghe đoạn âm thanh do người học cung cấp và đưa ra nhận xét chi tiết, hữu ích.

    **Bối cảnh:**
    - Từ cần phát âm: "{target_word}"
    - Trình độ của người học: "{user_level}" (Ví dụ: FLYERS, VSTEP B1)

    **Yêu cầu:**
    1.  Nghe kỹ file âm thanh đính kèm.
    2.  So sánh cách phát âm của người học với cách phát âm chuẩn của từ "{target_word}".
    3.  Cung cấp phản hồi dưới dạng một đối tượng JSON. **Không thêm bất kỳ giải thích nào bên ngoài đối tượng JSON này.**

    **Cấu trúc JSON đầu ra bắt buộc:**
    {
      "overall_score": <một số nguyên từ 0 đến 100, đánh giá tổng quan>,
      "transcribed_text": "<văn bản bạn nghe được từ audio>",
      "positive_feedback": "<Một câu nhận xét về những điểm làm tốt, dù là nhỏ nhất. Ví dụ: 'Bạn đã nhấn trọng âm đúng vào âm tiết thứ hai.'>",
      "points_to_improve": [
        {
          "phoneme": "<âm vị bị lỗi, ví dụ: '/s/'>",
          "error_description": "<mô tả lỗi sai một cách đơn giản, ví dụ: 'Âm /s/ của bạn nghe hơi giống âm /sh/ (nặng)'>",
          "suggestion": "<hướng dẫn sửa lỗi cụ thể, ví dụ: 'Để phát âm đúng, hãy mỉm cười nhẹ, để đầu lưỡi gần với phần nướu phía sau răng cửa trên và đẩy hơi ra.'>"
        },
        {
          "phoneme": "<trọng âm>",
          "error_description": "<mô tả lỗi sai, ví dụ: 'Bạn đang nhấn trọng âm vào âm tiết đầu tiên thay vì âm tiết thứ hai.'>",
          "suggestion": "<hướng dẫn sửa lỗi, ví dụ: 'Từ này có trọng âm rơi vào âm tiết thứ hai: pro-NUN-ci-a-tion. Hãy thử nhấn mạnh vào âm 'NUN' nhé.'>"
        }
      ]
    }
    **Lưu ý quan trọng:**
    - Nếu không có điểm nào cần cải thiện, hãy để mảng "points_to_improve" rỗng ([]).
    - Phản hồi cần ngắn gọn, tích cực và dễ hiểu cho người có trình độ "{user_level}".
"""

import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()


def generate_pronunciation_exercise(
    pronunciation_focus: str, exercise_type: str, num_sentences: int, target_level: str
) -> dict:
    """
    Sử dụng Gemini để tạo bài tập phát âm theo một chủ điểm cụ thể.
    """
    try:
        client = genai.Client()

        prompt_text = f"""
            Bạn là một chuyên gia ngữ âm và huấn luyện viên phát âm tiếng Anh, chuyên thiết kế bài tập thực hành cho người học Việt Nam.

            **Bối cảnh:**
            - Trình độ của người học: "{target_level}"
            - Chủ điểm phát âm cần luyện tập: "{pronunciation_focus}"
            - Dạng bài tập (câu hoặc đoạn văn): "{exercise_type}"
            - Số lượng câu bài tập cần tạo: {num_sentences}

            **Yêu cầu:**
            1.  Tạo chính xác {num_sentences} bài tập tiếng Anh khác nhau theo dạng "{exercise_type}".
            2.  Mỗi bài tập phải được thiết kế đặc biệt để người học có cơ hội thực hành "{pronunciation_focus}".
            3.  Với mỗi bài tập, hãy cung cấp phiên âm quốc tế (IPA) đầy đủ.
            4.  Với mỗi bài tập, hãy cung cấp một "mẹo" ngắn gọn bằng tiếng Việt để giúp người học phát âm đúng.
            5.  Từ vựng và cấu trúc câu phải phù hợp với trình độ "{target_level}".
            6.  Cung cấp toàn bộ phản hồi dưới dạng một đối tượng JSON duy nhất. **Không thêm bất kỳ văn bản giải thích nào bên ngoài đối tượng JSON này.**

            **Cấu trúc JSON đầu ra bắt buộc:**
            {{
              "pronunciation_focus": "{pronunciation_focus}",
              "level": "{target_level}",
              "practice_sentences": [
                {{
                  "id": 1,
                  "sentence": "<Câu tiếng Anh thứ nhất>",
                  "ipa_transcription": "<Phiên âm IPA đầy đủ của câu đó>",
                  "tip_vi": "<Một mẹo ngắn bằng tiếng Việt.>"
                }}
              ]
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
        return {"error": "Không thể tạo bài tập bằng Gemini."}


if __name__ == "__main__":
    focus = "Phân biệt âm /ʃ/ (sh) và /s/ (s)"
    type_of_exercise = "đoạn văn"
    num = 2
    level = "FLYERS"

    print(f"Đang tạo {num} câu bài tập để luyện tập: '{focus}' cho trình độ {level}...")
    print("-" * 20)

    exercise_data = generate_pronunciation_exercise(
        pronunciation_focus=focus,
        exercise_type=type_of_exercise,
        num_sentences=num,
        target_level=level,
    )

    print(json.dumps(exercise_data, indent=2, ensure_ascii=False))


""" 
*Input:
  - pronunciation_focus: str - Chủ điểm phát âm cần luyện tập (ví dụ: "Phân biệt âm /ʃ/ (sh) và /s/ (s)")
  - exercise_type: str - Dạng bài tập ("câu" hoặc "đoạn văn")
  - num_sentences: int - Số lượng câu bài tập cần tạo
  - target_level: str - Trình độ của người học (ví dụ: "STARTERS", "MOVERS", "FLYERS")
*Output:
  - dict - Đối tượng JSON với cấu trúc:
            {
              "pronunciation_focus": "<chủ điểm phát âm>",
              "level": "<trình độ người học>",
              "practice_sentences": [
                {
                  "id": <số nguyên, thứ tự câu>,
                  "sentence": "<câu tiếng Anh>",
                  "ipa_transcription": "<phiên âm IPA đầy đủ của câu đó>",
                  "tip_vi": "<một mẹo ngắn bằng tiếng Việt để giúp phát âm đúng>"
                }
              ]
            }
"""

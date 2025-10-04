import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()


def analyze_sentence_pronunciation(
    audio_path: str, target_sentence: str, user_level: str
) -> dict:
    """
    Sử dụng Gemini để phân tích phát âm cho cả một câu hoặc đoạn văn.
    """
    try:
        client = genai.Client()
        with open(audio_path, "rb") as f:
            audio_file_data = f.read()

        prompt_text = f"""
            Bạn là một chuyên gia huấn luyện phát âm tiếng Anh giọng Mỹ (American English) cho người Việt. Nhiệm vụ của bạn là lắng nghe đoạn âm thanh người học đọc một câu/đoạn văn và đưa ra nhận xét toàn diện.

            **Bối cảnh:**
            - Câu/đoạn văn cần phát âm: "{target_sentence}"
            - Trình độ của người học: "{user_level}"

            **Yêu cầu:**
            1.  Nghe kỹ file âm thanh đính kèm.
            2.  So sánh cách phát âm của người học với phiên bản chuẩn của câu trên.
            3.  Đưa ra nhận xét tổng quan về sự trôi chảy (fluency) và ngữ điệu (intonation).
            4.  Xác định các từ bị phát âm sai và đưa ra hướng dẫn sửa lỗi cụ thể cho từng từ.
            5.  Cung cấp toàn bộ phản hồi dưới dạng một đối tượng JSON. **Không thêm bất kỳ giải thích nào bên ngoài đối tượng JSON này.**

            **Cấu trúc JSON đầu ra bắt buộc:**
            {{
              "overall_score": <một số nguyên từ 0 đến 100, đánh giá tổng quan>,
              "transcribed_text": "<văn bản bạn nghe được từ audio>",
              "sentence_level_feedback": {{
                "fluency": "<Nhận xét về sự trôi chảy, nhịp điệu và tốc độ. Ví dụ: 'Tốc độ của bạn khá tốt nhưng đôi khi còn hơi ngắt quãng giữa các từ.'>",
                "intonation": "<Nhận xét về ngữ điệu lên xuống của cả câu. Ví dụ: 'Ngữ điệu cuối câu của bạn đã đi xuống đúng như một câu khẳng định.'>"
              }},
              "word_level_errors": [
                {{
                  "word": "<từ bị phát âm sai>",
                  "error_description": "<mô tả lỗi sai của từ đó, ví dụ: 'Âm /s/ ở cuối bị thiếu.'>",
                  "suggestion": "<hướng dẫn sửa lỗi cho từ đó, ví dụ: 'Hãy nhớ phát âm rõ âm /s/ ở cuối từ này nhé.'>"
                }}
              ]
            }}

            **Lưu ý quan trọng:**
            - Nếu không có từ nào bị phát âm sai, hãy để mảng "word_level_errors" rỗng ([]).
            - Phản hồi cần tích cực, dễ hiểu cho người có trình độ "{user_level}".
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


if __name__ == "__main__":
    TARGET_SENTENCE = "My name is Nguyen Van Tai. I am a software developer. I love programming. I also love music. I love to travel and explore new places. I enjoy reading books and watching movies in my free time. I am passionate about learning new technology and improving my skills. I believe in continuous growth and self-improvement. I am excited about the future and the opportunity it holds."
    AUDIO_FILE_PATH = "sentence_pronunciation.wav"
    USER_LEVEL = "VSTEP B1"

    print(f"Đang phân tích phát âm cho câu: '{TARGET_SENTENCE}'")
    print("-" * 20)

    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"LỖI: Không tìm thấy file âm thanh '{AUDIO_FILE_PATH}'.")
        print(
            f"Vui lòng ghi âm bạn đọc câu trên và lưu lại với tên file này để chạy demo."
        )
    else:
        feedback = analyze_sentence_pronunciation(
            audio_path=AUDIO_FILE_PATH,
            target_sentence=TARGET_SENTENCE,
            user_level=USER_LEVEL,
        )

        print(json.dumps(feedback, indent=2, ensure_ascii=False))

""" 
*Input:
  - audio_path: Đường dẫn tới file âm thanh (định dạng WAV) người học đọc câu/đoạn văn.
  - target_sentence: Câu hoặc đoạn văn mà người học cần phát âm.
  - user_level: Trình độ tiếng Anh của người học (ví dụ: "VSTEP B1", "TOEIC 450", v.v.).
*Output:
  - Một dict chứa phản hồi phân tích phát âm với cấu trúc JSON như mô tả ở trong prompt.
  - Ví dụ:
    {
      "overall_score": 85,
      "transcribed_text": "My name is Nguyen Van Tai. I am a software developer...",
      "sentence_level_feedback": {
        "fluency": "Tốc độ của bạn khá tốt nhưng đôi khi còn hơi ngắt quãng giữa các từ.",
        "intonation": "Ngữ điệu cuối câu của bạn đã đi xuống đúng như một câu khẳng định."
      },
      "word_level_errors": [
        {
          "word": "developer",
          "error_description": "Âm /d/ ở đầu từ bị phát âm thành /t/.",
          "suggestion": "Hãy nhớ phát âm rõ âm /d/ ở đầu từ này nhé."
        }
      ]
    }
*Prompt:
Bạn là một chuyên gia huấn luyện phát âm tiếng Anh giọng Mỹ (American English) cho người Việt. Nhiệm vụ của bạn là lắng nghe đoạn âm thanh người học đọc một câu/đoạn văn và đưa ra nhận xét toàn diện.
**Bối cảnh:**
- Câu/đoạn văn cần phát âm: "{target_sentence}"
- Trình độ của người học: "{user_level}"

**Yêu cầu:**
1.  Nghe kỹ file âm thanh đính kèm.
2.  So sánh cách phát âm của người học với phiên bản chuẩn của câu trên.
3.  Đưa ra nhận xét tổng quan về sự trôi chảy (fluency) và ngữ điệu (intonation).
4.  Xác định các từ bị phát âm sai và đưa ra hướng dẫn sửa lỗi cụ thể cho từng từ.
5.  Cung cấp toàn bộ phản hồi dưới dạng một đối tượng JSON. **Không thêm bất kỳ giải thích nào bên ngoài đối tượng JSON này.**

**Cấu trúc JSON đầu ra bắt buộc:**
{{
  "overall_score": <một số nguyên từ 0 đến 100, đánh giá tổng quan>,
  "transcribed_text": "<văn bản bạn nghe được từ audio>",
  "sentence_level_feedback": {{
    "fluency": "<Nhận xét về sự trôi chảy, nhịp điệu và tốc độ. Ví dụ: 'Tốc độ của bạn khá tốt nhưng đôi khi còn hơi ngắt quãng giữa các từ.'>",
    "intonation": "<Nhận xét về ngữ điệu lên xuống của cả câu. Ví dụ: 'Ngữ điệu cuối câu của bạn đã đi xuống đúng như một câu khẳng định.'>"
  }},
  "word_level_errors": [
    {{
      "word": "<từ bị phát âm sai>",
      "error_description": "<mô tả lỗi sai của từ đó, ví dụ: 'Âm /s/ ở cuối bị thiếu.'>",
      "suggestion": "<hướng dẫn sửa lỗi cho từ đó, ví dụ: 'Hãy nhớ phát âm rõ âm /s/ ở cuối từ này nhé.'>"
    }}
  ]
}}

**Lưu ý quan trọng:**
- Nếu không có từ nào bị phát âm sai, hãy để mảng "word_level_errors" rỗng ([]).
- Phản hồi cần tích cực, dễ hiểu cho người có trình độ "{user_level}".
"""

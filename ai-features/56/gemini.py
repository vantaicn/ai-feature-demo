import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()


def evaluate_vstep_speaking(audio_path: str, exam_part: str, exam_prompt: str) -> dict:
    """
    Sử dụng Gemini để chấm điểm và nhận xét bài thi VSTEP Speaking.
    """
    try:
        client = genai.Client()
        with open(audio_path, "rb") as f:
            audio_file_data = f.read()

        prompt_text = f"""
            Bạn là một giám khảo chấm thi VSTEP Speaking có nhiều năm kinh nghiệm, với khả năng nghe và phân tích ngôn ngữ cực kỳ chính xác.

            **Bối cảnh:**
            - Phần thi: "{exam_part}"
            - Đề bài/Câu hỏi: "{exam_prompt}"
            - File ghi âm bài nói của thí sinh được đính kèm.

            **Yêu cầu:**
            1.  Nghe kỹ file ghi âm.
            2.  Đầu tiên, hãy phiên âm (transcribe) lại toàn bộ những gì thí sinh nói một cách chính xác nhất.
            3.  Sau đó, chấm điểm bài nói dựa trên 4 tiêu chí của VSTEP Speaking. Với mỗi tiêu chí, hãy cho điểm trên thang 10 và đưa ra nhận xét.
                - **Grammar (Ngữ pháp):** Đánh giá sự đa dạng và chính xác của cấu trúc câu (đơn, ghép, phức). Ghi nhận các lỗi ngữ pháp hệ thống.
                - **Vocabulary (Từ vựng):** Đánh giá phạm vi từ vựng, khả năng sử dụng collocations và paraphrasing.
                - **Fluency and Coherence (Trôi chảy và Mạch lạc):** Đánh giá tốc độ nói, mức độ ngập ngừng, khả năng sắp xếp ý và sử dụng từ nối logic.
                - **Pronunciation (Phát âm):** Đánh giá độ rõ ràng của các âm, trọng âm từ/câu và ngữ điệu tự nhiên.
            4.  Tính điểm tổng kết bằng cách lấy trung bình cộng của 4 điểm tiêu chí, làm tròn đến 0.5.
            5.  Xác định các lỗi cụ thể về phát âm, ngữ pháp hoặc dùng từ. Trích dẫn phần nói bị lỗi và đưa ra gợi ý sửa chữa.
            6.  Cung cấp toàn bộ kết quả dưới dạng một đối tượng JSON duy nhất. **Tuyệt đối không thêm bất kỳ văn bản giải thích nào bên ngoài đối tượng JSON này.**

            **Cấu trúc JSON đầu ra bắt buộc:**
            {{
              "overall_score": <Điểm tổng kết, ví dụ: 7.0>,
              "full_transcript": "<Bản phiên âm đầy đủ bài nói của thí sinh>",
              "criteria_breakdown": {{
                "grammar": {{"score": <Điểm>, "strengths": "<Điểm mạnh>", "weaknesses": "<Điểm yếu>"}},
                "vocabulary": {{"score": <Điểm>, "strengths": "<Điểm mạnh>", "weaknesses": "<Điểm yếu>"}},
                "fluency_coherence": {{"score": <Điểm>, "strengths": "<Điểm mạnh>", "weaknesses": "<Điểm yếu>"}},
                "pronunciation": {{"score": <Điểm>, "strengths": "<Điểm mạnh>", "weaknesses": "<Điểm yếu>"}}
              }},
              "actionable_feedback": [
                {{
                  "error_quote": "<Trích dẫn câu/cụm từ bị lỗi từ transcript>",
                  "issue_type": "<Loại lỗi: 'Grammar', 'Vocabulary', hoặc 'Pronunciation'>",
                  "suggestion": "<Gợi ý sửa lỗi cụ thể, bằng tiếng Việt>"
                }}
              ]
            }}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
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
        return {"error": "Không thể đánh giá bài nói bằng Gemini."}


if __name__ == "__main__":
    EXAM_PART = "Phần 3: Phát triển chủ đề"
    EXAM_PROMPT = "Topic: The importance of learning a second language."
    AUDIO_FILE_PATH = "vstep_speaking_part3.wav"

    print(f"Đang đánh giá bài nói VSTEP: {EXAM_PART}...")
    print(f"Chủ đề: {EXAM_PROMPT}")
    print("-" * 20)

    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"LỖI: Không tìm thấy file âm thanh '{AUDIO_FILE_PATH}'.")
        print(
            f"Vui lòng ghi âm bạn trả lời chủ đề trên và lưu lại với tên file này để chạy demo."
        )
    else:
        evaluation_result = evaluate_vstep_speaking(
            audio_path=AUDIO_FILE_PATH, exam_part=EXAM_PART, exam_prompt=EXAM_PROMPT
        )

        print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))


""" 
*Input:
    - exam_part: str - Phần thi VSTEP Speaking (e.g., "Phần 3: Phát triển chủ đề")
    - exam_prompt: str - Đề bài/Câu hỏi của phần thi
    - audio_path: str - Đường dẫn đến file âm thanh ghi âm bài nói của thí sinh
*Output:
    - dict - Kết quả đánh giá bao gồm điểm tổng kết, bản phiên âm, điểm chi tiết theo tiêu chí và phản hồi cụ thể
    {
      "overall_score": 8.0,
      "full_transcript": "<Bản phiên âm đầy đủ bài nói của thí sinh>",
      "criteria_breakdown": {
        "grammar": {"score": 7.5, "strengths": "<Điểm mạnh>", "weaknesses": "<Điểm yếu>"},
        "vocabulary": {"score": 8.0, "strengths": "<Điểm mạnh>", "weaknesses": "<Điểm yếu>"},
        "fluency_coherence": {"score": 8.5, "strengths": "<Điểm mạnh>", "weaknesses": "<Điểm yếu>"},
        "pronunciation": {"score": 7.0, "strengths": "<Điểm mạnh>", "weaknesses": "<Điểm yếu>"}
      },
      "actionable_feedback": [
        {
          "error_quote": "<Trích dẫn câu/cụm từ bị lỗi từ transcript>",
          "issue_type": "<Loại lỗi: 'Grammar', 'Vocabulary', hoặc 'Pronunciation'>",
          "suggestion": "<Gợi ý sửa lỗi cụ thể, bằng tiếng Việt>"
        }
      ]
    }
"""

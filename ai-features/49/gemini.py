import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()


def evaluate_vstep_writing(
    task_type: str, exam_prompt: str, user_submission: str
) -> dict:
    """
    Sử dụng Gemini để chấm điểm và nhận xét bài thi VSTEP Writing.
    """
    try:
        client = genai.Client()

        prompt_text = f"""
        Bạn là một giám khảo chấm thi VSTEP Writing giàu kinh nghiệm. Nhiệm vụ của bạn là phân tích, chấm điểm và đưa ra nhận xét chi tiết cho bài viết của thí sinh một cách khách quan và mang tính xây dựng.

        **Bối cảnh:**
        - Loại bài thi: "{task_type}"
        - Đề bài gốc: "{exam_prompt}"
        - Bài viết của thí sinh:
        ---
        {user_submission}
        ---

        **Yêu cầu:**
        1.  Đọc kỹ đề bài và bài viết của thí sinh.
        2.  Chấm điểm bài viết dựa trên 4 tiêu chí chính của VSTEP Writing. Với mỗi tiêu chí, hãy cho điểm trên thang 10 và đưa ra nhận xét cụ thể.
            - **Task Fulfillment (Mức độ hoàn thành yêu cầu):** Bài viết có trả lời đủ các ý của đề bài không? Độ dài có đạt yêu cầu? Văn phong (formal/informal) có phù hợp không? Các ý chính có được phát triển không?
            - **Organization (Tổ chức bài viết):** Bố cục có rõ ràng (Mở-Thân-Kết) không? Việc chia đoạn có logic không? Các từ nối được sử dụng có đa dạng và hiệu quả không?
            - **Vocabulary (Từ vựng):** Phạm vi từ vựng có rộng không? Có sử dụng từ đúng ngữ cảnh, collocations không? Có lỗi chính tả không?
            - **Grammar (Ngữ pháp):** Cấu trúc câu có đa dạng (đơn, ghép, phức) không? Có các lỗi ngữ pháp về thì, S-V agreement, mạo từ, giới từ không? Dấu câu có chính xác không?
        3.  Tính điểm tổng kết cho bài viết này (thang điểm 10, làm tròn 0.5).
        4.  Cung cấp một phiên bản bài viết đã được sửa lỗi và gợi ý cách diễn đạt tốt hơn.
        5.  Cung cấp toàn bộ kết quả dưới dạng một đối tượng JSON duy nhất. **Tuyệt đối không thêm bất kỳ văn bản giải thích nào bên ngoài đối tượng JSON này.**

        **Lưu ý về JSON:**
        - Các trường "score" phải là số (float), ví dụ: 7.5
        - Các trường văn bản phải là chuỗi (string), bọc trong dấu ngoặc kép.
        - JSON phải hợp lệ tuyệt đối.

        **Cấu trúc JSON đầu ra bắt buộc:**
        {{
          "overall_score": 7.5,
          "summary_feedback_vi": "Một đoạn nhận xét chung ngắn gọn bằng tiếng Việt",
          "criteria_breakdown": {{
            "task_fulfillment": {{
              "score": 8.0,
              "strengths": "Những điểm làm tốt của tiêu chí này, bằng tiếng Việt",
              "weaknesses": "Những điểm cần cải thiện của tiêu chí này, bằng tiếng Việt"
            }},
            "organization": {{
              "score": 7.0,
              "strengths": "Những điểm làm tốt của tiêu chí này, bằng tiếng Việt",
              "weaknesses": "Những điểm cần cải thiện của tiêu chí này, bằng tiếng Việt"
            }},
            "vocabulary": {{
              "score": 7.5,
              "strengths": "Những điểm làm tốt của tiêu chí này, bằng tiếng Việt",
              "weaknesses": "Những điểm cần cải thiện của tiêu chí này, bằng tiếng Việt"
            }},
            "grammar": {{
              "score": 6.5,
              "strengths": "Những điểm làm tốt của tiêu chí này, bằng tiếng Việt",
              "weaknesses": "Những điểm cần cải thiện của tiêu chí này, bằng tiếng Việt"
            }}
          }},
          "corrected_text": "Toàn bộ bài viết của người dùng nhưng đã được sửa lỗi và tối ưu hóa. Đánh dấu các thay đổi nếu có thể."
        }}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{"role": "user", "parts": [{"text": prompt_text}]}],
            # Cân nhắc thêm safety_settings nếu gặp vấn đề bị chặn
            # safety_settings={'HARASSMENT': 'block_none', 'HATE_SPEECH': 'block_none', 'SEXUAL': 'block_none', 'DANGEROUS': 'block_none'}
        )

        if not response.text:
            return {"error": "Gemini không trả về kết quả."}

        cleaned_response = (
            response.text.strip().replace("```json", "").replace("```", "").strip()
        )
        return json.loads(cleaned_response)

    except Exception as e:
        print(f"Lỗi Gemini: {e}")
        return {"error": "Không thể đánh giá bài viết bằng Gemini."}


if __name__ == "__main__":
    TASK_TYPE = "Bài 2 (Viết luận)"

    EXAM_PROMPT = "Online learning is becoming more and more popular. What are the advantages and disadvantages of online learning? Give your opinion."

    USER_SUBMISSION = """
    In recent year, the world has see a big change in education, with online learning become very common. This way of learning have both good points and bad points.

    One of the main advantage of online learning is its flexible. Student can learn from anywhere and at anytime they want. This is very good for people who have jobs or live far from school. Furthermore, online courses is often cheaper than traditional classes, which help people save money. For example, they dont need to pay for transportation or campus accommodation.

    However, there are also some disadvantage. The biggest problem is the lack of face-to-face interact. Student may feel lonely and find it hard to ask teacher question directly. Another drawback is that it require strong self-discipline. If a student is not motivated, they can easy fall behind with their studies. It also depend on good internet connection, which not everyone have.

    In conclusion, I belief that online learning offer great benefit in term of flexibility and cost. Despite its drawbacks like the need for self-discipline and less interaction, I think the advantages are more important, specialy in our modern world.
    """

    print(f"Đang đánh giá bài viết VSTEP: {TASK_TYPE}...")
    print("-" * 20)

    evaluation_result = evaluate_vstep_writing(
        task_type=TASK_TYPE, exam_prompt=EXAM_PROMPT, user_submission=USER_SUBMISSION
    )

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))

""" 
*Input:
  - task_type: str - Loại bài thi VSTEP Writing (ví dụ: "Bài 1 (Thư từ)", "Bài 2 (Viết luận)")
  - exam_prompt: str - Đề bài gốc của bài thi VSTEP Writing
  - user_submission: str - Bài viết của thí sinh cần được chấm điểm và nhận xét
*Output:
  - dict - Kết quả đánh giá bao gồm điểm tổng kết, nhận xét chung, điểm và nhận xét chi tiết cho từng tiêu chí, và bài viết đã được sửa lỗi dưới dạng JSON:
    {
      "overall_score": <Điểm tổng kết, là số, ví dụ: 7.5>,
      "summary_feedback_vi": "<Một đoạn nhận xét chung ngắn gọn bằng tiếng Việt>",
      "criteria_breakdown": {
        "task_fulfillment": {
          "score": <Điểm cho tiêu chí này, ví dụ: 8.0>,
          "strengths": "<Những điểm làm tốt của tiêu chí này, bằng tiếng Việt>",
          "weaknesses": "<Những điểm cần cải thiện của tiêu chí này, bằng tiếng Việt>"
        },
        "organization": {
          "score": <Điểm cho tiêu chí này, ví dụ: 7.0>,
          "strengths": "<Những điểm làm tốt của tiêu chí này, bằng tiếng Việt>",
          "weaknesses": "<Những điểm cần cải thiện của tiêu chí này, bằng tiếng Việt>"
        },
        "vocabulary": {
          "score": <Điểm cho tiêu chí này, ví dụ: 7.5>,
          "strengths": "<Những điểm làm tốt của tiêu chí này, bằng tiếng Việt>",
          "weaknesses": "<Những điểm cần cải thiện của tiêu chí này, bằng tiếng Việt>"
        },
        "grammar": {
          "score": <Điểm cho tiêu chí này, ví dụ: 6.5>,
          "strengths": "<Những điểm làm tốt của tiêu chí này, bằng tiếng Việt>",
          "weaknesses": "<Những điểm cần cải thiện của tiêu chí này, bằng tiếng Việt>"
        }
      },
      "corrected_text": "<Toàn bộ bài viết của người dùng nhưng đã được sửa lỗi và tối ưu hóa. Đánh dấu các thay đổi nếu có thể.>"
    }
"""

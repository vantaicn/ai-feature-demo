import json
from dotenv import load_dotenv
from google import genai

load_dotenv()


def generate_grammar_exercise(
    grammar_structures: list, num_questions: int, target_level: str, exercise_type: str
) -> dict:
    """
    Sử dụng Gemini để tạo bài tập ngữ pháp theo yêu cầu.
    """
    try:
        grammar_str = ", ".join(f'"{g}"' for g in grammar_structures)

        client = genai.Client()

        prompt_text = f"""
            Bạn là một chuyên gia tạo nội dung học liệu tiếng Anh, chuyên thiết kế bài tập ngữ pháp cho người học Việt Nam.

            **Bối cảnh:**
            - Trình độ của người học: "{target_level}"
            - Các chủ điểm ngữ pháp cần kiểm tra: [{grammar_str}]
            - Loại bài tập: "{exercise_type}"
            - Số lượng câu hỏi: {num_questions}

            **Yêu cầu:**
            1.  Tạo chính xác {num_questions} câu hỏi thuộc loại "{exercise_type}".
            2.  Các câu hỏi phải kiểm tra trực tiếp và chỉ xoay quanh các chủ điểm ngữ pháp đã liệt kê.
            3.  Từ vựng và độ phức tạp của câu phải phù hợp với trình độ "{target_level}".
            4.  Đối với câu hỏi trắc nghiệm, các lựa chọn sai (distractors) phải hợp lý và có tính gây nhiễu.
            5.  Cung cấp phản hồi dưới dạng một đối tượng JSON duy nhất. **Tuyệt đối không thêm bất kỳ văn bản giải thích nào bên ngoài đối tượng JSON này.**

            **Cấu trúc JSON đầu ra bắt buộc:**
            {{
              "level": "{target_level}",
              "grammar_focus": [{grammar_str}],
              "exercises": [
                {{
                  "id": 1,
                  "question_text": "<Nội dung câu hỏi. Sử dụng '___' cho chỗ trống cần điền>",
                  "options": ["<Lựa chọn A>", "<Lựa chọn B>", "<Lựa chọn C>", "<Lựa chọn D>"],
                  "correct_answer": "<Đáp án đúng (phải khớp chính xác với một trong các options)>",
                  "explanation": "<Giải thích chi tiết bằng tiếng Việt tại sao đáp án đó đúng, có đề cập đến quy tắc ngữ pháp tương ứng.>"
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
    target_grammar = ["Thì Hiện tại Hoàn thành", "Thì Quá khứ Đơn"]
    number_of_q = 5
    level = "VSTEP B1"
    q_type = "Trắc nghiệm"

    print(f"Đang tạo {number_of_q} câu hỏi {q_type} cho trình độ {level}...")
    print(f"Chủ điểm: {', '.join(target_grammar)}")
    print("-" * 20)

    exercise_data = generate_grammar_exercise(
        grammar_structures=target_grammar,
        num_questions=number_of_q,
        target_level=level,
        exercise_type=q_type,
    )

    print(json.dumps(exercise_data, indent=2, ensure_ascii=False))


""" 
* Input:
    - grammar_structures: danh sách các cấu trúc ngữ pháp cần kiểm tra
    - num_questions: số lượng câu hỏi cần tạo
    - target_level: trình độ người học
    - exercise_type: loại bài tập (ví dụ: "Trắc nghiệm", "Điền từ", ...)
* Output:
    - Một đối tượng JSON chứa thông tin về bài tập ngữ pháp đã tạo:
    {
      "level": "<trình độ người học>",
      "grammar_focus": ["<cấu trúc ngữ pháp 1>", "<cấu trúc ngữ pháp 2>", ...],
      "exercises": [
        {
          "id": <số thứ tự câu hỏi>,
          "question_text": "<nội dung câu hỏi>",
          "options": ["<lựa chọn A>", "<lựa chọn B>", "<lựa chọn C>", "<lựa chọn D>"],  # Chỉ có cho câu hỏi trắc nghiệm
          "correct_answer": "<đáp án đúng>",
          "explanation": "<giải thích đáp án đúng>"
        },
        ...
      ]
    }}
* Prompt Gemini:
    Bạn là một chuyên gia tạo nội dung học liệu tiếng Anh, chuyên thiết kế bài tập ngữ pháp cho người học Việt Nam.

    **Bối cảnh:**
    - Trình độ của người học: "{target_level}"
    - Các chủ điểm ngữ pháp cần kiểm tra: [{grammar_str}]
    - Loại bài tập: "{exercise_type}"
    - Số lượng câu hỏi: {num_questions}

    **Yêu cầu:**
    1.  Tạo chính xác {num_questions} câu hỏi thuộc loại "{exercise_type}".
    2.  Các câu hỏi phải kiểm tra trực tiếp và chỉ xoay quanh các chủ điểm ngữ pháp đã liệt kê.
    3.  Từ vựng và độ phức tạp của câu phải phù hợp với trình độ "{target_level}".
    4.  Đối với câu hỏi trắc nghiệm, các lựa chọn sai (distractors) phải hợp lý và có tính gây nhiễu.
    5.  Cung cấp phản hồi dưới dạng một đối tượng JSON duy nhất. **Tuyệt đối không thêm bất kỳ văn bản giải thích nào bên ngoài đối tượng JSON này.**
    **Cấu trúc JSON đầu ra bắt buộc:**
    {{
      "level": "{target_level}",
      "grammar_focus": [{grammar_str}],
      "exercises": [
        {{
          "id": 1,
          "question_text": "<Nội dung câu hỏi. Sử dụng '___' cho chỗ trống cần điền>",
          "options": ["<Lựa chọn A>", "<Lựa chọn B>", "<Lựa chọn C>", "<Lựa chọn D>"],
          "correct_answer": "<Đáp án đúng (phải khớp chính xác với một trong các options)>",
          "explanation": "<Giải thích chi tiết bằng tiếng Việt tại sao đáp án đó đúng, có đề cập đến quy tắc ngữ pháp tương ứng.>"
        }}
      ]
    }}
"""

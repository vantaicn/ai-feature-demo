import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()


def analyze_grammar_structures(
    selected_sentence: str, paragraph_context: str, target_level: str
) -> dict:
    """
    Sử dụng Gemini để phân tích các cấu trúc ngữ pháp có trong câu được chọn.
    """
    try:
        client = genai.Client()

        prompt_text = f"""
            Bạn là một chuyên gia ngữ pháp tiếng Anh và giảng dạy ESL, chuyên về việc phân tích cấu trúc ngữ pháp cho người học Việt Nam.

            **Bối cảnh:**
            - Câu được chọn: "{selected_sentence}"
            - Đoạn văn ngữ cảnh: "{paragraph_context}"
            - Trình độ người học: "{target_level}"

            **Yêu cầu:**
            1. Phân tích và xác định TẤT CẢ các cấu trúc ngữ pháp có trong câu "{selected_sentence}".
            2. Với mỗi cấu trúc, giải thích nghĩa và chức năng trong ngữ cảnh của đoạn văn.
            3. Cung cấp công thức/pattern của từng cấu trúc.
            4. Đưa ra 3 ví dụ mẫu cho mỗi cấu trúc với cùng pattern.
            5. Giải thích khi nào và tại sao sử dụng cấu trúc đó.
            6. Cung cấp toàn bộ kết quả dưới dạng một đối tượng JSON duy nhất. **Không thêm bất kỳ văn bản giải thích nào bên ngoài đối tượng JSON này.**

            **Cấu trúc JSON đầu ra bắt buộc:**
            {{
              "selected_sentence": "{selected_sentence}",
              "paragraph_context": "{paragraph_context}",
              "level": "{target_level}",
              "sentence_analysis": {{
                "sentence_type": "<Loại câu: Simple/Compound/Complex/Compound-Complex>",
                "main_tense": "<Thì chính của câu>",
                "sentence_function": "<Chức năng: Statement/Question/Command/Exclamation>"
              }},
              "grammar_structures": [
                {{
                  "structure_id": 1,
                  "structure_name": "<Tên cấu trúc ngữ pháp>",
                  "pattern": "<Công thức/Pattern của cấu trúc>",
                  "highlighted_part": "<Phần trong câu gốc thể hiện cấu trúc này>",
                  "contextual_meaning": {{
                    "vietnamese": "<Ý nghĩa của cấu trúc trong ngữ cảnh này>",
                    "function": "<Chức năng của cấu trúc trong câu>"
                  }},
                  "detailed_explanation": {{
                    "usage_rules": "<Quy tắc sử dụng cấu trúc này>",
                    "when_to_use": "<Khi nào sử dụng cấu trúc này>",
                    "common_situations": "<Các tình huống thường dùng>",
                    "grammar_notes": "<Ghi chú ngữ pháp đặc biệt>"
                  }},
                  "examples": [
                    {{
                      "sentence": "<Ví dụ thứ nhất sử dụng cùng cấu trúc>",
                      "translation": "<Bản dịch tiếng Việt>",
                      "explanation": "<Giải thích cách cấu trúc hoạt động trong ví dụ này>"
                    }},
                    {{
                      "sentence": "<Ví dụ thứ hai sử dụng cùng cấu trúc>",
                      "translation": "<Bản dịch tiếng Việt>",
                      "explanation": "<Giải thích cách cấu trúc hoạt động trong ví dụ này>"
                    }},
                    {{
                      "sentence": "<Ví dụ thứ ba sử dụng cùng cấu trúc>",
                      "translation": "<Bản dịch tiếng Việt>",
                      "explanation": "<Giải thích cách cấu trúc hoạt động trong ví dụ này>"
                    }}
                  ],
                  "common_mistakes": [
                    {{
                      "mistake": "<Lỗi thường gặp khi sử dụng cấu trúc này>",
                      "correction": "<Cách sửa lỗi>",
                      "explanation": "<Giải thích tại sao bị lỗi>"
                    }}
                  ],
                  "related_structures": [
                    "<Cấu trúc ngữ pháp liên quan thứ nhất>",
                    "<Cấu trúc ngữ pháp liên quan thứ hai>"
                  ]
                }}
              ],
              "contextual_analysis": {{
                "paragraph_theme": "<Chủ đề chính của đoạn văn>",
                "sentence_role": "<Vai trò của câu này trong đoạn văn>",
                "discourse_markers": "<Các từ nối/liên kết có trong câu>",
                "register": "<Văn phong: formal/informal/academic/conversational>"
              }},
              "learning_suggestions": [
                "<Gợi ý học tập thứ nhất cho cấu trúc này>",
                "<Gợi ý học tập thứ hai cho cấu trúc này>",
                "<Gợi ý học tập thứ ba cho cấu trúc này>"
              ]
            }}

            **Lưu ý quan trọng:**
            - Phân tích TẤT CẢ cấu trúc ngữ pháp trong câu, từ cơ bản đến phức tạp.
            - Giải thích phù hợp với trình độ "{target_level}".
            - Tập trung vào cách cấu trúc hoạt động trong ngữ cảnh cụ thể.
            - Ví dụ phải đa dạng và thực tế.
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
        return {"error": "Không thể phân tích cấu trúc ngữ pháp bằng Gemini."}


def get_structure_details(structure_name: str, target_level: str) -> dict:
    """
    Lấy thông tin chi tiết về một cấu trúc ngữ pháp cụ thể.
    """
    try:
        client = genai.Client()

        prompt_text = f"""
            Bạn là một chuyên gia ngữ pháp tiếng Anh, chuyên về việc giải thích cấu trúc ngữ pháp chi tiết cho người học Việt Nam.

            **Yêu cầu:**
            Cung cấp thông tin đầy đủ về cấu trúc ngữ pháp "{structure_name}" phù hợp với trình độ "{target_level}".

            **Cấu trúc JSON đầu ra bắt buộc:**
            {{
              "structure_name": "{structure_name}",
              "level": "{target_level}",
              "comprehensive_info": {{
                "definition": "<Định nghĩa cấu trúc bằng tiếng Việt>",
                "pattern": "<Công thức/Pattern chính xác>",
                "variations": ["<Biến thể 1>", "<Biến thể 2>", "<Biến thể 3>"],
                "formation_rules": "<Quy tắc tạo thành cấu trúc>"
              }},
              "usage_contexts": {{
                "when_to_use": "<Khi nào sử dụng cấu trúc này>",
                "common_situations": [
                  "<Tình huống sử dụng 1>",
                  "<Tình huống sử dụng 2>", 
                  "<Tình huống sử dụng 3>"
                ],
                "register": "<Văn phong thích hợp: formal/informal/both>",
                "frequency": "<Mức độ phổ biến: very common/common/less common>"
              }},
              "detailed_examples": [
                {{
                  "category": "<Loại ví dụ: Basic/Intermediate/Advanced>",
                  "sentence": "<Câu ví dụ>",
                  "translation": "<Bản dịch tiếng Việt>",
                  "breakdown": "<Phân tích từng phần của cấu trúc>",
                  "context": "<Ngữ cảnh sử dụng ví dụ này>"
                }},
                {{
                  "category": "<Loại ví dụ: Basic/Intermediate/Advanced>",
                  "sentence": "<Câu ví dụ>",
                  "translation": "<Bản dịch tiếng Việt>",
                  "breakdown": "<Phân tích từng phần của cấu trúc>",
                  "context": "<Ngữ cảnh sử dụng ví dụ này>"
                }},
                {{
                  "category": "<Loại ví dụ: Basic/Intermediate/Advanced>",
                  "sentence": "<Câu ví dụ>",
                  "translation": "<Bản dịch tiếng Việt>",
                  "breakdown": "<Phân tích từng phần của cấu trúc>",
                  "context": "<Ngữ cảnh sử dụng ví dụ này>"  
                }}
              ],
              "comparison_with_similar": [
                {{
                  "similar_structure": "<Cấu trúc tương tự>",
                  "difference": "<Sự khác biệt chính>",
                  "example_comparison": "<Ví dụ so sánh>"
                }}
              ],
              "common_errors": [
                {{
                  "error_type": "<Loại lỗi>",
                  "wrong_example": "<Ví dụ sai>",
                  "correct_example": "<Ví dụ đúng>",
                  "explanation": "<Giải thích lỗi>",
                  "prevention_tip": "<Mẹo tránh lỗi>"
                }}
              ],
              "practice_exercises": [
                {{
                  "exercise_type": "<Loại bài tập: Fill in blanks/Transform/Choose correct>",
                  "question": "<Câu hỏi bài tập>",
                  "answer": "<Đáp án>",
                  "explanation": "<Giải thích đáp án>"
                }},
                {{
                  "exercise_type": "<Loại bài tập>",
                  "question": "<Câu hỏi bài tập>",
                  "answer": "<Đáp án>",
                  "explanation": "<Giải thích đáp án>"
                }}
              ],
              "learning_progression": {{
                "prerequisite_knowledge": ["<Kiến thức cần có trước>"],
                "next_level_structures": ["<Cấu trúc nâng cao tiếp theo>"],
                "practice_recommendations": "<Gợi ý luyện tập>"
              }}
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
        return {"error": "Không thể lấy thông tin chi tiết cấu trúc ngữ pháp."}


def process_grammar_analysis(
    selected_sentence: str, paragraph_context: str, target_level: str
) -> dict:
    """
    Hàm chính xử lý phân tích cấu trúc ngữ pháp.
    """
    print(f"Đang phân tích cấu trúc ngữ pháp trong câu...")
    print(f"Câu: '{selected_sentence}'")

    # Phân tích các cấu trúc trong câu
    analysis_result = analyze_grammar_structures(
        selected_sentence=selected_sentence,
        paragraph_context=paragraph_context,
        target_level=target_level,
    )

    if "error" in analysis_result:
        return analysis_result

    # Thêm thông tin về số lượng cấu trúc tìm được
    if "grammar_structures" in analysis_result:
        analysis_result["structures_count"] = len(analysis_result["grammar_structures"])

    return analysis_result


if __name__ == "__main__":
    # Ví dụ demo
    SELECTED_SENTENCE = "If I had studied harder, I would have passed the exam easily."
    PARAGRAPH_CONTEXT = """
    Education is very important for everyone's future. Many students struggle with their studies because they don't have good study habits. If I had studied harder, I would have passed the exam easily. Now I realize that consistent effort is the key to academic success. Students should develop effective study strategies from an early age.
    """
    TARGET_LEVEL = "VSTEP B2"

    print(f"TÍNH NĂNG 170: PHÂN TÍCH CẤU TRÚC NGỮ PHÁP TRONG CÂU")
    print(f"Câu được chọn: '{SELECTED_SENTENCE}'")
    print(f"Trình độ: {TARGET_LEVEL}")
    print("=" * 80)

    # Phân tích cấu trúc ngữ pháp
    result = process_grammar_analysis(
        selected_sentence=SELECTED_SENTENCE,
        paragraph_context=PARAGRAPH_CONTEXT,
        target_level=TARGET_LEVEL,
    )

    if "error" in result:
        print(f"Lỗi: {result['error']}")
    else:
        print("✅ Phân tích cấu trúc ngữ pháp hoàn tất!")
        print(f"📊 Tìm thấy {result.get('structures_count', 0)} cấu trúc ngữ pháp")
        print()
        print("--- KẾT QUẢ PHÂN TÍCH ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Lưu kết quả vào file
        with open("grammar_analysis_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n📄 Kết quả đã được lưu vào 'grammar_analysis_result.json'")

    print("\n" + "=" * 80)
    print("🔍 DEMO: XEM CHI TIẾT MỘT CẤU TRÚC CỤ THỂ")

    # Demo xem chi tiết một cấu trúc
    STRUCTURE_NAME = (
        "Third Conditional (If + Past Perfect, would have + Past Participle)"
    )
    print(f"Cấu trúc: {STRUCTURE_NAME}")
    print("-" * 60)

    detailed_info = get_structure_details(STRUCTURE_NAME, TARGET_LEVEL)

    if "error" in detailed_info:
        print(f"Lỗi: {detailed_info['error']}")
    else:
        print("✅ Lấy thông tin chi tiết hoàn tất!")
        print()
        print("--- THÔNG TIN CHI TIẾT CẤU TRÚC ---")
        print(json.dumps(detailed_info, indent=2, ensure_ascii=False))

        # Lưu thông tin chi tiết vào file
        with open("structure_details.json", "w", encoding="utf-8") as f:
            json.dump(detailed_info, f, ensure_ascii=False, indent=2)
        print(f"\n📄 Thông tin chi tiết đã được lưu vào 'structure_details.json'")


""" 
*Input:
  - selected_sentence: str - Câu được người dùng bôi đen để phân tích (ví dụ: "If I had studied harder, I would have passed the exam easily.")
  - paragraph_context: str - Đoạn văn chứa câu đó để hiểu ngữ cảnh
  - target_level: str - Trình độ của người học (ví dụ: "VSTEP B2", "TOEIC 700", "IELTS 6.5")

*Output:
  - dict - Kết quả phân tích cấu trúc ngữ pháp với cấu trúc JSON:
    {
      "selected_sentence": "<câu được chọn>",
      "paragraph_context": "<đoạn văn ngữ cảnh>",
      "level": "<trình độ người học>",
      "structures_count": <số lượng cấu trúc tìm được>,
      "sentence_analysis": {
        "sentence_type": "<loại câu>",
        "main_tense": "<thì chính>",
        "sentence_function": "<chức năng câu>"
      },
      "grammar_structures": [
        {
          "structure_id": <ID>,
          "structure_name": "<tên cấu trúc>",
          "pattern": "<công thức>",
          "highlighted_part": "<phần thể hiện cấu trúc>",
          "contextual_meaning": {
            "vietnamese": "<ý nghĩa tiếng Việt>",
            "function": "<chức năng trong câu>"
          },
          "detailed_explanation": {
            "usage_rules": "<quy tắc sử dụng>",
            "when_to_use": "<khi nào dùng>",
            "common_situations": "<tình huống thường dùng>",
            "grammar_notes": "<ghi chú ngữ pháp>"
          },
          "examples": [<3 ví dụ với giải thích>],
          "common_mistakes": [<lỗi thường gặp>],
          "related_structures": [<cấu trúc liên quan>]
        }
      ],
      "contextual_analysis": {
        "paragraph_theme": "<chủ đề đoạn văn>",
        "sentence_role": "<vai trò câu trong đoạn văn>",
        "discourse_markers": "<từ nối>",
        "register": "<văn phong>"
      },
      "learning_suggestions": [<gợi ý học tập>]
    }

*Tính năng chính:
  - Phân tích TẤT CẢ cấu trúc ngữ pháp trong câu được chọn
  - Giải thích ý nghĩa và chức năng trong ngữ cảnh đoạn văn
  - Cung cấp pattern/công thức cho từng cấu trúc
  - 3 ví dụ mẫu với cùng pattern cho mỗi cấu trúc
  - Hướng dẫn khi nào và tại sao sử dụng
  - Phân tích lỗi thường gặp và cách tránh
  - Xem chi tiết đầy đủ về từng cấu trúc cụ thể

*Hàm bổ sung:
  - get_structure_details(): Lấy thông tin chi tiết về một cấu trúc cụ thể
  - Bao gồm: định nghĩa, biến thể, quy tắc, ví dụ phân tích, so sánh với cấu trúc tương tự, bài tập thực hành

*Cách sử dụng:
  1. Người dùng bôi đen một câu trong đoạn văn
  2. Hệ thống gọi process_grammar_analysis()
  3. Hiển thị danh sách các cấu trúc tìm được
  4. Người dùng có thể click vào cấu trúc để xem chi tiết
  5. Gọi get_structure_details() để lấy thông tin đầy đủ
"""

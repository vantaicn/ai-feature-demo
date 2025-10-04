import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()


def evaluate_flyers_speaking(
    audio_path: str, exam_part: str, exam_prompt: str, additional_context: str = ""
) -> dict:
    """
    Sử dụng Gemini để đánh giá chi tiết bài thi Cambridge Flyers Speaking.

    Args:
        audio_path: Đường dẫn đến file âm thanh bài nói của học sinh
        exam_part: Phần thi (Part 1, Part 2, Part 3, hoặc Part 4)
        exam_prompt: Mô tả đề bài/yêu cầu cụ thể
        additional_context: Thông tin bổ sung (ví dụ: mô tả tranh ảnh)

    Returns:
        dict: Kết quả đánh giá chi tiết theo tiêu chí Cambridge YLE
    """
    try:
        client = genai.Client()

        with open(audio_path, "rb") as f:
            audio_file_data = f.read()

        prompt_text = f"""
            Bạn là một giám khảo chấm thi Cambridge YLE Flyers Speaking có nhiều năm kinh nghiệm, được chứng nhận bởi Cambridge Assessment English. Bạn có khả năng đánh giá chính xác trình độ tiếng Anh của trẻ em theo tiêu chuẩn quốc tế.

            **BỐI CẢNH BÀI THI:**
            - Kỳ thi: Cambridge Young Learners English (YLE) - Flyers Level
            - Phần thi: "{exam_part}"
            - Đề bài/Yêu cầu: "{exam_prompt}"
            {f"- Thông tin bổ sung: {additional_context}" if additional_context else ""}
            - Độ tuổi học sinh: 9-12 tuổi (trình độ tương đương A2 theo CEFR)

            **TIÊU CHÍ CHẤM ĐIỂM CAMBRIDGE FLYERS SPEAKING:**
            
            **1. Grammar & Vocabulary (25%):**
            - Sử dụng đúng các cấu trúc ngữ pháp cơ bản (Present Simple, Past Simple, Present Continuous, Future Simple)
            - Từ vựng phù hợp với chủ đề và độ tuổi (khoảng 600-700 từ)
            - Khả năng diễn đạt ý tưởng với ngôn ngữ đơn giản nhưng chính xác

            **2. Pronunciation (25%):**
            - Phát âm rõ ràng, dễ hiểu cho người nghe
            - Trọng âm từ và câu cơ bản đúng
            - Ngữ điệu tự nhiên phù hợp với ngữ cảnh
            
            **3. Discourse Management (25%):**  
            - Khả năng tổ chức ý tưởng logic, mạch lạc
            - Sử dụng từ nối đơn giản (and, but, because, then, first, next...)
            - Duy trì chủ đề và phát triển ý tưởng phù hợp với yêu cầu đề bài

            **4. Interactive Communication (25%):**
            - Khả năng giao tiếp tự nhiên, không quá cứng nhắc
            - Phản ứng phù hợp với câu hỏi/tình huống
            - Sự tự tin và sẵn sàng trong giao tiếp

            **YÊU CẦU ĐÁNH GIÁ:**
            1. Nghe kỹ file ghi âm đính kèm và transcribe chính xác những gì học sinh nói.
            2. Đánh giá theo 4 tiêu chí trên, cho điểm mỗi tiêu chí từ 0-5 (theo thang điểm Cambridge):
               - 0: Không đạt
               - 1: Yếu  
               - 2: Khá yếu
               - 3: Trung bình
               - 4: Khá tốt
               - 5: Xuất sắc (mức A2+ cho trẻ em)
            3. Tính điểm tổng kết (trung bình 4 điểm tiêu chí).
            4. Đưa ra nhận xét chi tiết, khuyến khích và gợi ý cải thiện phù hợp với độ tuổi.
            5. Xác định những lỗi cụ thể và đưa ra lời khuyên thực tế.

            **LƯUY Ý ĐẶC BIỆT:**
            - Đánh giá phù hợp với trình độ A2 cho trẻ em (không quá khắt khe)
            - Nhận xét phải tích cực, khuyến khích tinh thần học tập
            - Gợi ý cải thiện phải cụ thể và dễ thực hiện cho trẻ em
            - Sử dụng ngôn ngữ đơn giản, dễ hiểu

            **CẤU TRÚC JSON ĐẦU RA BẮT BUỘC:**
            {{
              "overall_score": <điểm tổng kết, ví dụ: 3.5>,
              "level_assessment": "<đánh giá trình độ: 'Pre-A1', 'A1', 'A2' hoặc 'Above A2'>",
              "full_transcript": "<bản phiên âm đầy đủ bài nói của học sinh>",
              "criteria_scores": {{
                "grammar_vocabulary": {{
                  "score": <điểm 0-5>,
                  "strengths": "<những điểm làm tốt, bằng tiếng Việt>",
                  "areas_for_improvement": "<những điểm cần cải thiện, bằng tiếng Việt>"
                }},
                "pronunciation": {{
                  "score": <điểm 0-5>,
                  "strengths": "<những điểm làm tốt, bằng tiếng Việt>", 
                  "areas_for_improvement": "<những điểm cần cải thiện, bằng tiếng Việt>"
                }},
                "discourse_management": {{
                  "score": <điểm 0-5>,
                  "strengths": "<những điểm làm tốt, bằng tiếng Việt>",
                  "areas_for_improvement": "<những điểm cần cải thiện, bằng tiếng Việt>"
                }},
                "interactive_communication": {{
                  "score": <điểm 0-5>,
                  "strengths": "<những điểm làm tốt, bằng tiếng Việt>",
                  "areas_for_improvement": "<những điểm cần cải thiện, bằng tiếng Việt>"
                }}
              }},
              "detailed_feedback": {{
                "positive_highlights": [
                  "<điểm tích cực thứ nhất, bằng tiếng Việt>",
                  "<điểm tích cực thứ hai, bằng tiếng Việt>"
                ],
                "specific_errors": [
                  {{
                    "error_quote": "<trích dẫn lỗi từ transcript>",
                    "error_type": "<loại lỗi: Grammar/Vocabulary/Pronunciation>", 
                    "correction": "<cách sửa đúng>",
                    "explanation": "<giải thích ngắn gọn bằng tiếng Việt>"
                  }}
                ],
                "improvement_suggestions": [
                  "<gợi ý cải thiện thứ nhất, cụ thể và dễ thực hiện>",
                  "<gợi ý cải thiện thứ hai, cụ thể và dễ thực hiện>"
                ]
              }},
              "next_steps": "<khuyến nghị về bước học tiếp theo phù hợp với trình độ hiện tại>"
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
        return {"error": f"Không thể đánh giá bài nói bằng Gemini: {str(e)}"}


def create_sample_evaluation_report(evaluation_result: dict) -> str:
    """
    Tạo báo cáo đánh giá dễ đọc từ kết quả JSON.
    """
    if "error" in evaluation_result:
        return f"Lỗi: {evaluation_result['error']}"

    report = f"""
BÁO CÁO ĐÁNH GIÁ FLYERS SPEAKING
{'='*50}

ĐIỂM TỔNG KẾT: {evaluation_result.get('overall_score', 'N/A')}/5.0
TRÌNH ĐỘ ĐÁNH GIÁ: {evaluation_result.get('level_assessment', 'N/A')}

TRANSCRIPT:
{evaluation_result.get('full_transcript', 'Không có transcript')}

CHI TIẾT ĐIỂM SỐ:
"""

    criteria = evaluation_result.get("criteria_scores", {})
    for criterion_name, criterion_data in criteria.items():
        criterion_display = {
            "grammar_vocabulary": "Ngữ pháp & Từ vựng",
            "pronunciation": "Phát âm",
            "discourse_management": "Quản lý diễn ngôn",
            "interactive_communication": "Giao tiếp tương tác",
        }.get(criterion_name, criterion_name)

        report += f"""
{criterion_display}: {criterion_data.get('score', 'N/A')}/5
  Điểm mạnh: {criterion_data.get('strengths', 'N/A')}
  🔸 Cần cải thiện: {criterion_data.get('areas_for_improvement', 'N/A')}"""

    feedback = evaluation_result.get("detailed_feedback", {})

    report += f"""

ĐIỂM NỔI BẬT:
"""
    for highlight in feedback.get("positive_highlights", []):
        report += f"• {highlight}\n"

    if feedback.get("specific_errors"):
        report += f"""
LỖI CỤ THỂ VÀ CÁCH SỬA:
"""
        for error in feedback.get("specific_errors", []):
            report += f"""• Lỗi: "{error.get('error_quote', '')}"
  Sửa thành: "{error.get('correction', '')}"
  Giải thích: {error.get('explanation', '')}
"""

    report += f"""
GỢI Ý CẢI THIỆN:
"""
    for suggestion in feedback.get("improvement_suggestions", []):
        report += f"• {suggestion}\n"

    report += f"""
BƯỚC TIẾP THEO:
{evaluation_result.get('next_steps', 'Không có gợi ý')}
"""

    return report


if __name__ == "__main__":
    # Ví dụ sử dụng cho Part 2 - Story telling
    EXAM_PART = "Part 2 - Story Telling"
    EXAM_PROMPT = "Look at the pictures and tell me the story. You can see a girl named Emma and her day at the zoo."
    ADDITIONAL_CONTEXT = "4 pictures showing: 1) Emma arriving at zoo entrance, 2) Emma feeding giraffes, 3) Emma watching monkeys play, 4) Emma buying souvenirs at gift shop"
    AUDIO_FILE_PATH = "flyers_speaking_part2.wav"  # File âm thanh demo

    print(f"ĐÁNH GIÁ FLYERS SPEAKING")
    print(f"Phần thi: {EXAM_PART}")
    print(f"Đề bài: {EXAM_PROMPT}")
    if ADDITIONAL_CONTEXT:
        print(f"Bối cảnh: {ADDITIONAL_CONTEXT}")
    print("=" * 60)

    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"THÔNG BÁO: Không tìm thấy file âm thanh '{AUDIO_FILE_PATH}'.")
        print("Để test tính năng này, hãy:")
        print("   1. Ghi âm học sinh kể chuyện theo 4 bức tranh")
        print("   2. Lưu file với tên 'flyers_speaking_part2.wav'")
        print("   3. Chạy lại script này")
        print()

        print("🔄 Đang tạo báo cáo mẫu...")
        sample_result = {
            "overall_score": 3.5,
            "level_assessment": "A2",
            "full_transcript": "Emma go to zoo. She very happy. First she see big giraffe and give food. Then she watch funny monkeys. They jumping and playing. After that Emma buy presents for family. She buy toy elephant and postcard. Emma have good day at zoo.",
            "criteria_scores": {
                "grammar_vocabulary": {
                    "score": 3,
                    "strengths": "Sử dụng từ vựng về động vật chính xác, có thể mô tả các hoạt động cơ bản",
                    "areas_for_improvement": "Cần chú ý thì quá khứ (went thay vì go), động từ to-be (is/was)",
                },
                "pronunciation": {
                    "score": 4,
                    "strengths": "Phát âm rõ ràng, dễ hiểu, trọng âm từ khá chính xác",
                    "areas_for_improvement": "Luyện thêm âm cuối /s/ và /ed/",
                },
                "discourse_management": {
                    "score": 4,
                    "strengths": "Kể chuyện theo trình tự logic, sử dụng từ nối đơn giản (first, then, after that)",
                    "areas_for_improvement": "Có thể mở rộng ý tưởng với nhiều chi tiết hơn",
                },
                "interactive_communication": {
                    "score": 3,
                    "strengths": "Tự tin kể chuyện, giọng nói rõ ràng",
                    "areas_for_improvement": "Có thể thêm cảm xúc vào câu chuyện",
                },
            },
            "detailed_feedback": {
                "positive_highlights": [
                    "Kể chuyện theo đúng trình tự thời gian, rất logic!",
                    "Sử dụng tốt từ vựng về động vật và hoạt động",
                ],
                "specific_errors": [
                    {
                        "error_quote": "Emma go to zoo",
                        "error_type": "Grammar",
                        "correction": "Emma went to zoo",
                        "explanation": "Cần dùng thì quá khứ 'went' thay vì 'go'",
                    },
                    {
                        "error_quote": "She very happy",
                        "error_type": "Grammar",
                        "correction": "She was very happy",
                        "explanation": "Cần thêm động từ 'was' trước tính từ",
                    },
                ],
                "improvement_suggestions": [
                    "Luyện tập thì quá khứ với các động từ bất quy tắc (go-went, buy-bought)",
                    "Thêm tính từ miêu tả (big giraffe, funny monkeys, beautiful postcard)",
                ],
            },
            "next_steps": "Tiếp tục luyện tập kể chuyện với nhiều thì khác nhau và mở rộng từ vựng miêu tả cảm xúc.",
        }

        print(create_sample_evaluation_report(sample_result))
    else:
        evaluation_result = evaluate_flyers_speaking(
            audio_path=AUDIO_FILE_PATH,
            exam_part=EXAM_PART,
            exam_prompt=EXAM_PROMPT,
            additional_context=ADDITIONAL_CONTEXT,
        )

        if "error" in evaluation_result:
            print(f"Lỗi: {evaluation_result['error']}")
        else:
            print("Đánh giá hoàn tất!")
            print()
            print(create_sample_evaluation_report(evaluation_result))

            with open("flyers_evaluation_result.json", "w", encoding="utf-8") as f:
                json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
            print(f"Kết quả chi tiết đã được lưu vào 'flyers_evaluation_result.json'")


""" 
*Input:
  - audio_path: str - Đường dẫn đến file âm thanh ghi âm bài nói (định dạng WAV)
  - exam_part: str - Phần thi Flyers (Part 1, Part 2, Part 3, Part 4)
  - exam_prompt: str - Mô tả đề bài/yêu cầu cụ thể
  - additional_context: str - Thông tin bổ sung (mô tả tranh, ngữ cảnh...)

*Output:
  - dict - Kết quả đánh giá chi tiết theo tiêu chuẩn Cambridge YLE Flyers:
    {
      "overall_score": <điểm tổng kết 0-5>,
      "level_assessment": "<trình độ: Pre-A1/A1/A2/Above A2>",
      "full_transcript": "<bản ghi âm văn bản đầy đủ>",
      "criteria_scores": {
        "grammar_vocabulary": {"score": <0-5>, "strengths": "...", "areas_for_improvement": "..."},
        "pronunciation": {"score": <0-5>, "strengths": "...", "areas_for_improvement": "..."},
        "discourse_management": {"score": <0-5>, "strengths": "...", "areas_for_improvement": "..."},
        "interactive_communication": {"score": <0-5>, "strengths": "...", "areas_for_improvement": "..."}
      },
      "detailed_feedback": {
        "positive_highlights": ["...", "..."],
        "specific_errors": [{"error_quote": "...", "error_type": "...", "correction": "...", "explanation": "..."}],
        "improvement_suggestions": ["...", "..."]
      },
      "next_steps": "<gợi ý bước học tiếp theo>"
    }

*Các phần thi Flyers Speaking:
  - Part 1: Find the differences (Tìm khác biệt giữa 2 tranh)
  - Part 2: Story telling (Kể chuyện từ 4 tranh theo trình tự)  
  - Part 3: Personal questions (Câu hỏi cá nhân)
  - Part 4: Discussion (Thảo luận chủ đề)

*Tiêu chí chấm điểm (mỗi tiêu chí 0-5 điểm):
  - Grammar & Vocabulary: Ngữ pháp và từ vựng phù hợp trình độ A2
  - Pronunciation: Phát âm rõ ràng, trọng âm và ngữ điệu
  - Discourse Management: Tổ chức ý tưởng, sử dụng từ nối
  - Interactive Communication: Khả năng giao tiếp tự nhiên

*Đặc điểm đánh giá cho trẻ em:
  - Khuyến khích tích cực, không quá khắt khe
  - Phù hợp với tâm lý và khả năng nhận thức của trẻ 9-12 tuổi
  - Gợi ý cải thiện cụ thể và dễ thực hiện
  - Đánh giá theo chuẩn A2 cho Young Learners
"""

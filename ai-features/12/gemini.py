from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

with open("pronunciation.wav", "rb") as f:
    audio_file = f.read()

target_word = "pronunciation"
user_level = "VSTEP B1"

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        {
            "role": "user",
            "parts": [
                {
                    "text": """
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
                },
                {
                    "inline_data": {
                        "mime_type": "audio/wav",  # hoặc "audio/mp3", "audio/m4a"
                        "data": audio_file,
                    }
                },
            ],
        }
    ],
)

print(response.text)

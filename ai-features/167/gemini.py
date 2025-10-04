import os
import json
from dotenv import load_dotenv
from google import genai
from google.cloud import texttospeech

load_dotenv()


def translate_vocabulary_with_context(
    selected_word: str, context_sentence: str, target_level: str
) -> dict:
    """
    Sử dụng Gemini để dịch từ vựng theo ngữ cảnh với đầy đủ thông tin phát âm và ví dụ.
    """
    try:
        client = genai.Client()

        prompt_text = f"""
            Bạn là một chuyên gia ngôn ngữ học và giảng dạy tiếng Anh, chuyên về việc giải thích từ vựng theo ngữ cảnh cho người học Việt Nam.

            **Bối cảnh:**
            - Từ vựng được chọn: "{selected_word}"
            - Câu chứa từ vựng (ngữ cảnh): "{context_sentence}"
            - Trình độ người học: "{target_level}"

            **Yêu cầu:**
            1. Phân tích từ "{selected_word}" trong ngữ cảnh câu "{context_sentence}".
            2. Cung cấp nghĩa chính xác của từ trong ngữ cảnh này (không phải nghĩa chung).
            3. Giải thích chi tiết cách sử dụng từ này trong tình huống cụ thể.
            4. Cung cấp phiên âm quốc tế (IPA) đầy đủ và chính xác.
            5. Tạo 3 ví dụ mẫu khác sử dụng từ này với nghĩa tương tự.
            6. Cung cấp toàn bộ kết quả dưới dạng một đối tượng JSON duy nhất. **Không thêm bất kỳ văn bản giải thích nào bên ngoài đối tượng JSON này.**

            **Cấu trúc JSON đầu ra bắt buộc:**
            {{
              "word": "{selected_word}",
              "context": "{context_sentence}",
              "level": "{target_level}",
              "contextual_meaning": {{
                "vietnamese": "<Nghĩa tiếng Việt chính xác trong ngữ cảnh này>",
                "english_definition": "<Định nghĩa bằng tiếng Anh đơn giản phù hợp trình độ>",
                "word_class": "<Từ loại: noun/verb/adjective/adverb/...>"
              }},
              "pronunciation": {{
                "ipa": "<Phiên âm IPA đầy đủ và chính xác>",
                "phonetic_spelling": "<Cách đọc đơn giản bằng tiếng Việt>",
                "stress_pattern": "<Mô tả trọng âm, ví dụ: 'Trọng âm rơi vào âm tiết thứ 2'>"
              }},
              "detailed_explanation": {{
                "usage_in_context": "<Giải thích cách từ này hoạt động trong câu gốc>",
                "grammar_notes": "<Ghi chú ngữ pháp nếu có (cấu trúc, collocations, v.v.)>",
                "common_mistakes": "<Lỗi thường gặp khi sử dụng từ này>"
              }},
              "example_sentences": [
                {{
                  "sentence": "<Ví dụ tiếng Anh thứ nhất>",
                  "translation": "<Bản dịch tiếng Việt>",
                  "explanation": "<Giải thích ngắn gọn về cách sử dụng trong ví dụ này>"
                }},
                {{
                  "sentence": "<Ví dụ tiếng Anh thứ hai>", 
                  "translation": "<Bản dịch tiếng Việt>",
                  "explanation": "<Giải thích ngắn gọn về cách sử dụng trong ví dụ này>"
                }},
                {{
                  "sentence": "<Ví dụ tiếng Anh thứ ba>",
                  "translation": "<Bản dịch tiếng Việt>", 
                  "explanation": "<Giải thích ngắn gọn về cách sử dụng trong ví dụ này>"
                }}
              ],
              "related_vocabulary": {{
                "synonyms": ["<từ đồng nghĩa 1>", "<từ đồng nghĩa 2>"],
                "antonyms": ["<từ trái nghĩa 1>", "<từ trái nghĩa 2>"],
                "collocations": ["<cụm từ thường đi kèm 1>", "<cụm từ thường đi kèm 2>"]
              }},
              "audio_text": "<Từ vựng để tạo file phát âm>"
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
        return {"error": "Không thể dịch từ vựng bằng Gemini."}


def create_pronunciation_audio(text: str, output_file: str) -> str | None:
    """
    Sử dụng Google Cloud Text-to-Speech để tạo file phát âm.
    """
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-D",  # Giọng nữ tự nhiên
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=0.9,  # Chậm hơn một chút để dễ nghe
        )

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        with open(output_file, "wb") as out:
            out.write(response.audio_content)
        print(f"✅ File phát âm đã được tạo: {output_file}")
        return output_file

    except Exception as e:
        print(f"Lỗi Google TTS: {e}")
        return None


def process_vocabulary_translation(
    selected_word: str,
    context_sentence: str,
    target_level: str,
    create_audio: bool = True,
) -> dict:
    """
    Hàm chính xử lý dịch từ vựng với đầy đủ tính năng.
    """
    print(f"Đang phân tích từ '{selected_word}' trong ngữ cảnh...")

    # Bước 1: Dịch và phân tích từ vựng
    translation_result = translate_vocabulary_with_context(
        selected_word=selected_word,
        context_sentence=context_sentence,
        target_level=target_level,
    )

    if "error" in translation_result:
        return translation_result

    # Bước 2: Tạo file phát âm nếu được yêu cầu
    if create_audio and "audio_text" in translation_result:
        audio_filename = f"pronunciation_{selected_word.lower().replace(' ', '_')}.wav"
        audio_file = create_pronunciation_audio(
            translation_result["audio_text"], audio_filename
        )
        if audio_file:
            translation_result["pronunciation_audio"] = audio_file

    return translation_result


if __name__ == "__main__":
    # Ví dụ demo
    SELECTED_WORD = "score"
    CONTEXT_SENTENCE = "The student received a high score on the VSTEP exam."
    TARGET_LEVEL = "VSTEP B1"

    print(f"TÍNH NĂNG 167: DỊCH TỪ VỰNG THEO NGỮ CẢNH")
    print(f"Từ được chọn: '{SELECTED_WORD}'")
    print(f"Ngữ cảnh: '{CONTEXT_SENTENCE}'")
    print(f"Trình độ: {TARGET_LEVEL}")
    print("=" * 60)

    result = process_vocabulary_translation(
        selected_word=SELECTED_WORD,
        context_sentence=CONTEXT_SENTENCE,
        target_level=TARGET_LEVEL,
        create_audio=True,
    )

    if "error" in result:
        print(f"Lỗi: {result['error']}")
    else:
        print("✅ Phân tích từ vựng hoàn tất!")
        print()
        print("--- KẾT QUẢ CHI TIẾT ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Lưu kết quả vào file để tham khảo
        with open(
            f"vocabulary_analysis_{SELECTED_WORD}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(
            f"\n📄 Kết quả đã được lưu vào 'vocabulary_analysis_{SELECTED_WORD}.json'"
        )


""" 
*Input:
  - selected_word: str - Từ vựng được người dùng bôi đen/chọn (ví dụ: "score")
  - context_sentence: str - Câu chứa từ vựng đó để hiểu ngữ cảnh (ví dụ: "The student received a high score on the exam.")
  - target_level: str - Trình độ của người học (ví dụ: "VSTEP B1", "TOEIC 500", "FLYERS")
  - create_audio: bool - Có tạo file phát âm không (mặc định True)

*Output:
  - dict - Kết quả phân tích từ vựng với cấu trúc JSON:
    {
      "word": "<từ vựng được chọn>",
      "context": "<câu ngữ cảnh>", 
      "level": "<trình độ người học>",
      "contextual_meaning": {
        "vietnamese": "<nghĩa tiếng Việt chính xác trong ngữ cảnh>",
        "english_definition": "<định nghĩa tiếng Anh đơn giản>",
        "word_class": "<từ loại>"
      },
      "pronunciation": {
        "ipa": "<phiên âm IPA>",
        "phonetic_spelling": "<cách đọc bằng tiếng Việt>", 
        "stress_pattern": "<mô tả trọng âm>"
      },
      "detailed_explanation": {
        "usage_in_context": "<giải thích cách sử dụng trong câu gốc>",
        "grammar_notes": "<ghi chú ngữ pháp>",
        "common_mistakes": "<lỗi thường gặp>"
      },
      "example_sentences": [
        {
          "sentence": "<ví dụ tiếng Anh>",
          "translation": "<bản dịch tiếng Việt>",
          "explanation": "<giải thích cách sử dụng>"
        }
      ],
      "related_vocabulary": {
        "synonyms": ["<từ đồng nghĩa>"],
        "antonyms": ["<từ trái nghĩa>"], 
        "collocations": ["<cụm từ thường đi kèm>"]
      },
      "audio_text": "<text để tạo audio>",
      "pronunciation_audio": "<đường dẫn file audio nếu được tạo>"
    }

*Tính năng chính:
  - Dịch từ vựng theo ngữ cảnh cụ thể (không phải nghĩa chung)
  - Cung cấp phiên âm IPA chính xác và cách đọc tiếng Việt
  - Giải thích chi tiết cách sử dụng với ví dụ mẫu
  - Tạo file phát âm bằng Google Cloud TTS
  - Gợi ý từ vựng liên quan (đồng nghĩa, trái nghĩa, collocations)
  - Phù hợp với trình độ người học

*Cách sử dụng:
  1. Người dùng bôi đen từ vựng trong một câu
  2. Hệ thống gọi hàm process_vocabulary_translation()
  3. Nhận được phân tích chi tiết và file phát âm
  4. Có thể lưu vào sổ tay từ vựng cá nhân
"""

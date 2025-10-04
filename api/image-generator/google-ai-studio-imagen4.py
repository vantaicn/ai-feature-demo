from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_images(
    model="imagen-4.0-generate-preview-06-06",
    prompt="Candid portrait photo of a young woman with visibly flushed, red cheeks. She is covering her face with both hands but peeking through her fingers with wide, awkward eyes, conveying a strong sense of embarrassment. The background is a softly blurred, warm-toned classroom. Soft, natural window light illuminates her face, highlighting the blush. Hyper-realistic, high detail, shallow depth of field. --ar 3:4",
    config=types.GenerateImagesConfig(
        number_of_images=1,
    ),
)
for generated_image in response.generated_images:
    image_bytes = generated_image.image.image_bytes
    unique_id = uuid.uuid4().hex
    file_name = f"./images/generated_image_{unique_id}.png"

    with open(file_name, "wb") as f:
        f.write(image_bytes)

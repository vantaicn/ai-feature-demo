from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY"))

contents = (
    "A surreal scene of a floating island with a giant tree in the center, "
    "waterfalls cascading off the edges, and a vibrant sunset sky, digital art."
)

response = client.models.generate_content(
    model="gemini-2.0-flash-preview-image-generation",
    contents=contents,
    config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
)

for part in response.candidates[0].content.parts:
    if part.text is not None:
        print(part.text)
    elif part.inline_data is not None:
        image = Image.open(BytesIO((part.inline_data.data)))
        unique_id = uuid.uuid4().hex
        image.save(f"gemini-native-image_{unique_id}.png")

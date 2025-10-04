from together import Together
import base64
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = Together(api_key=os.getenv("TOGETHER_AI_API_KEY"))
response = client.images.generate(
    prompt="A fantasy landscape with castles and dragons, vibrant colors, highly detailed, digital art",
    model="black-forest-labs/FLUX.1-schnell",
    steps=4,
    n=1,
    response_format="base64",
)

response_data = response.data

for idx, image in enumerate(response_data, start=1):
    image_base64 = image.b64_json
    unique_id = uuid.uuid4().hex
    filename = f"./images/together_schnell_{unique_id}.png"

    with open(filename, "wb") as f:
        f.write(base64.b64decode(image_base64))

    print(f"Saved {filename}")

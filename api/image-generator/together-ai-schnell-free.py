from together import Together
import base64
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = Together(api_key=os.getenv("TOGETHER_AI_API_KEY"))
response = client.images.generate(
    prompt="A realistic photo of a cat wearing a spacesuit, floating in space with Earth in the background, high resolution.",
    model="black-forest-labs/FLUX.1-schnell-Free",
    steps=4,
    n=2,
    response_format="base64",
)

response_data = response.data

for idx, image in enumerate(response_data, start=1):
    image_base64 = image.b64_json
    unique_id = uuid.uuid4().hex
    filename = f"./images/together_schnell[free]_{unique_id}.png"

    with open(filename, "wb") as f:
        f.write(base64.b64decode(image_base64))

    print(f"Saved {filename}")

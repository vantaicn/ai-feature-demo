import requests
import base64
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

url = "https://api.freepik.com/v1/ai/text-to-image"

payload = {
    "prompt": "A picture to illustrate the vocabulary 'score', digital art.",
    "negative_prompt": "b&w, earth, cartoon, ugly",
    "guidance_scale": 2,
    "seed": 42,
    "num_images": 1,
    "image": {"size": "square_1_1"},
    "styling": {
        "style": "anime",
        "effects": {"color": "pastel", "lightning": "warm", "framing": "portrait"},
        "colors": [
            {"color": "#FF5733", "weight": 1},
            {"color": "#33FF57", "weight": 1},
        ],
    },
    "filter_nsfw": True,
}
headers = {
    "x-freepik-api-key": os.getenv("FREEPIK_API_KEY"),
    "Content-Type": "application/json",
}

response = requests.post(url, json=payload, headers=headers)

response_data = response.json()

for idx, image_data in enumerate(response_data["data"], start=1):
    image_base64 = image_data["base64"]
    unique_id = uuid.uuid4().hex
    filename = f"./images/freepik_image_{unique_id}.png"

    with open(filename, "wb") as f:
        f.write(base64.b64decode(image_base64))

    print(f"Saved {filename}")

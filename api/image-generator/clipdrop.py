import requests
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

r = requests.post(
    "https://clipdrop-api.co/text-to-image/v1",
    files={"prompt": (None, "shot of vaporwave fashion dog in miami", "text/plain")},
    headers={"x-api-key": os.getenv("CLIPDROP_API_KEY")},
)
if r.ok:
    # r.content contains the bytes of the returned image
    unique_id = uuid.uuid4().hex
    with open(f"clipdrop_{unique_id}.png", "wb") as f:
        f.write(r.content)
else:
    r.raise_for_status()

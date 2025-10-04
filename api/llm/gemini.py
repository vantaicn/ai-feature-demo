from google import genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Nêu cấu trúc thì Hiện tại đơn. Và cách dùng nó."
)
print(response.text)

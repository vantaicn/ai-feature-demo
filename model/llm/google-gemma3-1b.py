from huggingface_hub import login
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Login with Hugging Face token from environment
login(os.getenv("HUGGINGFACE_TOKEN"), new_session=False)

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="google/gemma-3-1b-it")
messages = [
    {
        "role": "user",
        "content": "Bạn hãy nêu cấu trúc thì Hiện tại hoàn thành? Cách dùng nó ra sao?",
    },
]
print(pipe(messages))

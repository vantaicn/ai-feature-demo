from vertexai.preview.vision_models import ImageGenerationModel
import vertexai
import os

path_to_service_account = r"C:\Users\vanta\Downloads\videocreatorbackend-872809953a56.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path_to_service_account

vertexai.init(project="videocreatorbackend", location="us-central1")

generation_model = ImageGenerationModel.from_pretrained("imagen-4.0-generate-preview-06-06")

images = generation_model.generate_images(
    prompt="A picture to illustrate the vocabulary 'of'",
    number_of_images=1,
    aspect_ratio="1:1",
    negative_prompt="",
    person_generation="allow_all",
    safety_filter_level="block_few",
    add_watermark=True,
)

# Save images
import uuid

for i, image in enumerate(images):
    unique_id = uuid.uuid4().hex
    file_name = f"./images/google_cloud_{unique_id}.png"
    image_bytes = image._image_bytes
    
    with open(file_name, "wb") as f:
        f.write(image_bytes)
    
    print(f"Saved {file_name}")
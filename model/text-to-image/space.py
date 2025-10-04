from gradio_client import Client

client = Client("haduykien/graduation-project")
result = client.predict(
    prompt="An astronaut riding a horse on Mars, hd, dramatic lighting",
    api_name="/predict"
)
print(result)
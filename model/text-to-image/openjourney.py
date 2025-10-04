from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("prompthero/openjourney")
prompt = "A fantasy landscape with castles and dragons, vibrant colors, highly detailed, digital art"
image = pipe(prompt).images[0]
image.save("openjourney.png")

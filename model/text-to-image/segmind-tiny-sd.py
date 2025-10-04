from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("segmind/tiny-sd")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]
image.save("segmind-tiny-sd1.png")



import torch
from diffusers import StableDiffusion3Pipeline

for dump in range(110):
    try:
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
        print("success download model")
        break
    except Exception as e:
        print(f"try: {dump} times: {e}")

if dump == 110:
    print("fail to download model")

# pipe = pipe.to("mps")
images = pipe(
    "A cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]
# image
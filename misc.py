from vid import slerp, new_embeds
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("dreamlike-art/dreamlike-photoreal-2.0")

new_embeds('A horse galloping on the streets','A man swimming alone in a large pool', pipe)